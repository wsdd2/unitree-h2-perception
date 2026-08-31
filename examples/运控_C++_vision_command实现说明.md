# 运控 C++：自动发 requested_class_names + 回报 3D 坐标

视觉侧已就绪，运控只需实现一个「请求–等待–回报」节点（可挂在现有 `h2_arm_service_node` 上）。

## 1. 数据流

```text
运控 C++ 节点
  │  发布  InspectionCommand
  │  topic: /robot/inspection_command
  │  字段: requested_class_names / active_target_... / timeout_sec ...
  ▼
视觉感知 (start_perception_h2.sh)
  │  发布  Object3DArray
  │  topic: /detector/objects
  │  每项: detection.class_name + point_target (通常 torso_link)
  ▼
运控 C++ 在回调里过滤 class → 稳定判定 → 得到 xyz
  （可选再调 /h2_arm/singlearmjoint 动臂）
```

环境必须一致：

```bash
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
# 能找到 detector_msgs（source Foxy_ROS 或把 msg 编进运控工作空间）
source ~/MscapeTech/Foxy_ROS/install/setup.bash
```

消息包：`detector_msgs`  
- `InspectionCommand`  
- `Object3DArray` / `Object3D`

## 2. 推荐接口形态（与你们现有习惯一致）

对外仍用 **Trigger service**（自动化脚本好调）：

```text
/h2_arm/vision_command   std_srvs/srv/Trigger
```

服务回调内部做完整一轮：

1. 读 YAML（或读内存里的 stage 配置）→ 得到 `requested_class_names`
2. `command_id = prefix + "_" + timestamp_ms`（每次调用唯一）
3. 发布 `/robot/inspection_command`（本轮内建议 5Hz 重发同一 `command_id`）
4. 订阅 `/detector/objects`，只保留 `requested_class_names` 里的类
5. 满足 `min_confidence` + `required_stable_frames` + `max_position_std_m`，或直到 `timeout_sec`
6. `response.success` + `response.message`（建议 JSON 字符串带回各类 xyz）

YAML 示例字段（与 `InspectionCommand` 对齐）：

```yaml
command_id: "observe_lock_handle"
stage_id: 1
stage_name: "locate_lock_and_handle"
requested_action: "observe_targets"
requested_class_names:
  - "lock point"
  - "black cabinet door handle"
active_target_class_name: ""
min_confidence: 0.30
required_stable_frames: 3
max_position_std_m: 0.005
timeout_sec: 20.0
```

改类名 = 改 YAML（或改 stage 表），再 `service call` 一次即可。

## 3. C++ 节点骨架（可直接贴进 h2_arm）

```cpp
// 依赖: rclcpp, std_srvs, detector_msgs, yaml-cpp (或你们已有 YAML 库)

#include <chrono>
#include <cmath>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "detector_msgs/msg/inspection_command.hpp"
#include "detector_msgs/msg/object3_d_array.hpp"

using InspectionCommand = detector_msgs::msg::InspectionCommand;
using Object3DArray = detector_msgs::msg::Object3DArray;

struct ClassSample {
  double x{0}, y{0}, z{0};
  float conf{0};
  std::string frame;
};

class VisionCommandNode : public rclcpp::Node {
public:
  VisionCommandNode() : Node("h2_arm_vision_command") {
    cmd_pub_ = create_publisher<InspectionCommand>("/robot/inspection_command", 10);
    obj_sub_ = create_subscription<Object3DArray>(
      "/detector/objects", 10,
      std::bind(&VisionCommandNode::onObjects, this, std::placeholders::_1));

    srv_ = create_service<std_srvs::srv::Trigger>(
      "/h2_arm/vision_command",
      std::bind(&VisionCommandNode::onTrigger, this,
                std::placeholders::_1, std::placeholders::_2));

    // 本轮激活时 5Hz 重发同一 command
    timer_ = create_wall_timer(
      std::chrono::milliseconds(200),
      std::bind(&VisionCommandNode::republish, this));
  }

private:
  void onTrigger(
      const std::shared_ptr<std_srvs::srv::Trigger::Request> /*req*/,
      std::shared_ptr<std_srvs::srv::Trigger::Response> res)
  {
    std::unique_lock<std::mutex> lk(mu_);
    if (busy_) {
      res->success = false;
      res->message = "busy";
      return;
    }
    busy_ = true;
    lk.unlock();

    // 1) 读 YAML → 填 cfg_（略：用你们现有 loader）
    // cfg_.requested_class_names = {"lock point", "black cabinet door handle"};
    // cfg_.min_confidence = 0.30; cfg_.required_stable_frames = 3; ...

    InspectionCommand cmd;
    cmd.header.frame_id = "pelvis";
    cmd.command_id = cfg_.command_id_prefix + "_" +
        std::to_string(now_ms());  // 每次 Trigger 唯一
    cmd.stage_id = cfg_.stage_id;
    cmd.stage_name = cfg_.stage_name;
    cmd.requested_action = cfg_.requested_action;  // "observe_targets"
    cmd.requested_class_names = cfg_.requested_class_names;
    cmd.active_target_class_name = cfg_.active_target_class_name;
    cmd.selection_policy = "highest_confidence";
    cmd.min_confidence = cfg_.min_confidence;
    cmd.required_stable_frames = cfg_.required_stable_frames;
    cmd.max_position_std_m = cfg_.max_position_std_m;
    cmd.timeout_sec = cfg_.timeout_sec;

    {
      std::lock_guard<std::mutex> g(mu_);
      active_cmd_ = cmd;
      hist_.clear();
      for (const auto& n : cfg_.requested_class_names) hist_[to_lower(n)] = {};
      done_ = false;
      result_json_.clear();
    }
    cmd_pub_->publish(cmd);

    // 2) 等到齐或超时
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::duration<double>(cfg_.timeout_sec);
    while (rclcpp::ok()) {
      {
        std::lock_guard<std::mutex> g(mu_);
        if (done_) break;
      }
      if (std::chrono::steady_clock::now() > deadline) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      // 若用 MultiThreadedExecutor，对象回调会并行更新 hist_
    }

    std::lock_guard<std::mutex> g(mu_);
    busy_ = false;
    active_cmd_ = std::nullopt;
    if (!done_) {
      res->success = false;
      res->message = result_json_.empty() ? "timeout" : result_json_;
      return;
    }
    res->success = true;
    res->message = result_json_;  // JSON: 各类 xyz / frame / conf
  }

  void republish() {
    std::lock_guard<std::mutex> g(mu_);
    if (!busy_ || !active_cmd_) return;
    active_cmd_->header.stamp = now();
    cmd_pub_->publish(*active_cmd_);
  }

  void onObjects(const Object3DArray::SharedPtr msg) {
    std::lock_guard<std::mutex> g(mu_);
    if (!busy_ || !active_cmd_) return;

    // 本帧每类取最高置信度
    std::unordered_map<std::string, ClassSample> best;
    for (const auto& obj : msg->objects) {
      if (!obj.valid) continue;
      const std::string name = obj.detection.class_name;
      const std::string key = to_lower(name);
      if (!hist_.count(key)) continue;
      if (obj.detection.confidence < cfg_.min_confidence) continue;

      ClassSample s;
      s.x = obj.point_target.x;
      s.y = obj.point_target.y;
      s.z = obj.point_target.z;
      s.conf = obj.detection.confidence;
      s.frame = obj.target_frame;
      auto it = best.find(key);
      if (it == best.end() || s.conf > it->second.conf) best[key] = s;
    }

    for (auto& [key, s] : best) {
      auto& q = hist_[key];
      q.push_back(s);
      while (static_cast<int>(q.size()) > cfg_.required_stable_frames) q.erase(q.begin());
    }

    // 全部 requested 类都达到稳定帧且位置标准差达标 → 成功
    if (all_stable_locked(g_unused)) {
      result_json_ = build_json_from_hist();  // 自写：{"lock point":{"x":..,"y":..,"z":..},...}
      done_ = true;
    }
  }

  // --- 成员略：cfg_, hist_, busy_, done_, active_cmd_, pubs/subs ---
};

// main: MultiThreadedExecutor（Trigger 阻塞等待时，objects 回调仍能进）
```

要点：

- 必须用 **多线程 executor**，否则 Trigger 里 `sleep` 会堵死订阅回调。
- 坐标用 `object.point_target`（`target_frame` 一般是 `torso_link`）。
- `observe_targets` 时 `active_target_class_name` 可空；要跟单个锁定点可再订 `/detector/target_point`。

## 4. 自动化怎么跑

```bash
# 终端A：视觉
bash ~/MscapeTech/Foxy_ROS/scripts/start_perception_h2.sh

# 终端B：运控节点（提供 /h2_arm/vision_command）
cd ~/h2/unitree_sdk2/build/bin
./h2_arm_service_node

# 终端C：一键请求（改 YAML 里的 requested_class_names 即换目标）
ros2 service call /h2_arm/vision_command std_srvs/srv/Trigger
# 成功 → response.message 里是坐标 JSON
# 之后如需动臂：
ros2 service call /h2_arm/singlearmjoint std_srvs/srv/Trigger
```

多阶段自动化：每个 stage 一份 YAML（或一张表），脚本按序 `service call` 即可：

```bash
# stage1 观察锁+把手
cp stage1.yaml current_vision_command.yaml
ros2 service call /h2_arm/vision_command std_srvs/srv/Trigger

# stage2 观察按钮
cp stage2.yaml current_vision_command.yaml
ros2 service call /h2_arm/vision_command std_srvs/srv/Trigger
```

或在 C++ 里做状态机：stage 完成 → 自动换成下一组 `requested_class_names` 再发一轮（不必每次手动 call）。

## 5. 和视觉的责任边界

| 谁 | 做什么 |
|----|--------|
| 运控 | 发 `InspectionCommand`；收 `Object3DArray`；稳定判定；动臂 |
| 视觉 | 听 command；持续发 `/detector/objects`（及可选 `target_point`） |
| 视觉 **不提供** | `/h2_arm/vision_command` Trigger（这是运控自己的封装） |

运控 `package.xml` / `CMakeLists` 需依赖 `detector_msgs`，并能 `source` 到该消息包。
