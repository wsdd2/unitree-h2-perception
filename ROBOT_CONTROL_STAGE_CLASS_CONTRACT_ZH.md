# 机器人运控阶段与视觉类别接口合同

本文档面向机器人运控程序，规定以下两个接口的使用方式：

```text
/robot/inspection_status   detector_msgs/msg/RobotInspectionStatus
/robot/inspection_command  detector_msgs/msg/InspectionCommand
```

基本规则：

1. `InspectionStatus`表示机器人当前状态。
2. `InspectionCommand`表示本阶段希望视觉查找和执行的目标。
3. 两个消息的`stage_id`必须一致。
4. 建议两个消息均以5Hz持续发布。
5. 同一请求重复发布时必须保持相同的`command_id`。
6. 只有新请求才能生成新的`command_id`。

## 一、机器人可以请求的静态class_name

机器人程序必须使用以下最终发布名称：

```text
lock point
black cabinet door handle

red push button
green push button
yellow push button
black push button

square analog ammeter
square analog voltmeter
digital panel meter
multi-color indicator panel
round pressure gauge

black rotary selector switch
black rocker switch
yellow toggle switch
red toggle switch
white toggle switch

green work tag
red hang cord
work tag grasp point
cabinet hang hook
```

机器人程序不得请求以下仅供YOLOE使用的原始Prompt：

```text
green square work here safety tag
red twisted hanging cord rope
white adhesive cabinet wall hook
```

它们会在视觉节点内部归一化成：

```text
green work tag
red hang cord
cabinet hang hook
```

## 二、OCR动态语义

OCR不会修改静态`class_name`，而是为外观相同的控件增加实例语义。

示例：

```text
class_name: white toggle switch
label_text: 备用
label_confidence: 0.91
semantic_name: white toggle switch/备用
control_id: white_toggle_switch/备用
```

请求某一类的所有实例时使用：

```text
requested_class_names
active_target_class_name
```

请求带有指定文字标签的实例时使用：

```text
requested_semantic_names
active_target_semantic_name
```

例如：

```text
requested_semantic_names:
  - white toggle switch/备用
active_target_semantic_name: white toggle switch/备用
```

如果OCR为空、置信度不足或尚未稳定，不允许自动匹配到另一个外观相同的开关。

## 三、requested_action定义

当前支持：

```text
observe_targets
```

只观察和返回目标，不产生机械臂动作。

```text
move
```

移动到目标预接近点。

```text
press
```

仅允许用于`lock point`和四色按钮。

```text
grasp_rotate
door
```

目前仅允许用于`black cabinet door handle`开门序列。

```text
abort
```

取消当前自动化动作。

以下物理技能尚未实现：

- 旋钮旋转
- Rocker拨动
- Toggle拨动
- 工作牌自动抓取
- 工作牌自动挂钩
- 表计读数

这些目标目前只能使用`observe_targets`。

## 四、Stage 0：空闲

Status：

```yaml
stage_id: 0
stage_name: idle
current_action: idle
motion_active: false
```

不需要发送InspectionCommand。

## 五、Stage 1：同时定位Lock和Handle

Status：

```yaml
stage_id: 1
stage_name: locate_lock_and_handle
current_action: observe_targets
motion_active: false
target_reachable: true
```

Command：

```yaml
command_id: open-door-observe-001
stage_id: 1
stage_name: locate_lock_and_handle
requested_action: observe_targets
requested_class_names:
  - lock point
  - black cabinet door handle
active_target_class_name: ""
selection_policy: highest_confidence
lock_target: false
min_confidence: 0.30
required_stable_frames: 5
max_position_std_m: 0.003
timeout_sec: 10.0
```

预期输出：

```text
lock point:
  force_point_target_m

black cabinet door handle:
  handle_image_left_target_m
  handle_image_right_target_m
  handle_center_target_m
  handle_width_m
```

## 六、Stage 2：戳按Lock point

Status：

```yaml
stage_id: 2
stage_name: press_lock_point
current_action: press_lock_point
target_id: lock_point_1
motion_active: false
target_reachable: true
```

Command：

```yaml
command_id: press-lock-001
stage_id: 2
stage_name: press_lock_point
requested_action: press
requested_class_names:
  - lock point
active_target_class_name: lock point
target_id: lock_point_1
lock_target: true
selection_policy: highest_confidence
min_confidence: 0.30
required_stable_frames: 5
max_position_std_m: 0.003
timeout_sec: 10.0
```

当前已支持物理执行。

## 七、Stage 3：抓取并打开电柜门

Status：

```yaml
stage_id: 3
stage_name: grasp_or_pull_handle
current_action: open_cabinet_door
target_id: cabinet_handle_1
motion_active: false
target_reachable: true
```

Command：

```yaml
command_id: open-handle-001
stage_id: 3
stage_name: grasp_or_pull_handle
requested_action: grasp_rotate
requested_class_names:
  - black cabinet door handle
active_target_class_name: black cabinet door handle
target_id: cabinet_handle_1
lock_target: true
selection_policy: highest_confidence
min_confidence: 0.30
required_stable_frames: 5
max_position_std_m: 0.003
timeout_sec: 10.0
```

当前已支持五阶段开门动作。

## 八、Stage 4：柜门已打开

Status：

```yaml
stage_id: 4
stage_name: door_opened
current_action: door_opened
motion_active: false
progress: 1.0
target_reachable: true
```

不需要发送新Command。

## 九、Stage 5：恢复或中止

Status：

```yaml
stage_id: 5
stage_name: recover_or_abort
current_action: abort
motion_active: false
has_error: true
target_reachable: false
```

Command：

```yaml
command_id: abort-001
stage_id: 5
stage_name: recover_or_abort
requested_action: abort
```

收到后自动化Worker中止当前动作。

## 十、Stage 6：定位/抓取工作牌

Status：

```yaml
stage_id: 6
stage_name: pick_work_tag
current_action: locate_work_tag_grasp
target_reachable: true
```

定位Command：

```yaml
requested_action: observe_targets
requested_class_names:
  - green work tag
  - red hang cord
  - work tag grasp point
```

最终抓取着力点：

```text
work tag grasp point
```

自动夹取动作尚未实现。

## 十一、Stage 7：定位挂钩

Status：

```yaml
stage_id: 7
stage_name: hang_work_tag
current_action: locate_cabinet_hook
target_reachable: true
```

Command：

```yaml
requested_action: observe_targets
requested_class_names:
  - cabinet hang hook
```

自动挂钩动作尚未实现。

## 十二、Stage 8和9

保留编号，当前禁止使用。

## 十三、Stage 10：扫描柜内控件

第一阶段红框区域的实际布局：

```text
第一排左侧四个：red/green push button
第一排右侧两个：black rotary selector switch
第二排三个：black rotary selector switch
第三排：yellow/red/white toggle switch
```

Status：

```yaml
stage_id: 10
stage_name: inspect_inside_panel
current_action: observe_inside_controls
target_reachable: true
```

Command示例：

```yaml
requested_action: observe_targets
requested_class_names:
  - red push button
  - green push button
  - black rotary selector switch
  - yellow toggle switch
  - red toggle switch
  - white toggle switch
```

只应填写当前画面预期可见的类别。所有请求类别均满足稳定条件后，观察请求才会完成。

## 十四、Stage 11：定位/读取表计

Status：

```yaml
stage_id: 11
stage_name: inspect_meter
current_action: observe_meter
target_reachable: true
```

每次请求一个表计类别：

```text
square analog ammeter
square analog voltmeter
digital panel meter
round pressure gauge
```

使用：

```text
requested_action: observe_targets
```

当前仅输出表计3D位置，尚未实现模拟表指针读数和数字表OCR读数。

## 十五、Stage 12：戳按柜内按钮

允许执行的类别：

```text
red push button
green push button
yellow push button
black push button
```

Status示例：

```yaml
stage_id: 12
stage_name: press_inside_button
current_action: press_red_button
target_id: red_button_1
target_reachable: true
```

Command示例：

```yaml
command_id: press-red-001
stage_id: 12
stage_name: press_inside_button
requested_action: press
requested_class_names:
  - red push button
active_target_class_name: red push button
target_id: red_button_1
lock_target: true
```

当前已支持物理戳按。

按OCR语义选择时：

```yaml
requested_semantic_names:
  - red push button/停止
active_target_semantic_name: red push button/停止
```

## 十六、Stage 13：旋钮

Status：

```yaml
stage_id: 13
stage_name: inspect_rotary_selector
current_action: observe_rotary_selector
target_reachable: true
```

Command：

```yaml
requested_action: observe_targets
requested_class_names:
  - black rotary selector switch
```

第一排右侧两个旋钮使用二次语义：

```text
black rotary selector switch/top_row/with_tag
```

第二排三个旋钮使用：

```text
black rotary selector switch/middle_row/with_tag
```

OCR稳定后，`with_tag`替换为实际标签文字，例如：

```text
black rotary selector switch/top_row/远方
black rotary selector switch/middle_row/手动
```

物理旋转技能尚未实现。禁止对旋钮发送`grasp_rotate`，该动作当前仅用于电柜门把手。

## 十七、Stage 14：Rocker和Toggle开关

允许定位的类别：

```text
black rocker switch
yellow toggle switch
red toggle switch
white toggle switch
```

Status：

```yaml
stage_id: 14
stage_name: inspect_toggle_switch
current_action: observe_toggle_switch
target_reachable: true
```

使用：

```text
requested_action: observe_targets
```

物理拨动技能尚未实现。

可使用OCR语义区分相同外观实例：

```text
white toggle switch/备用
white toggle switch/远方
white toggle switch/就地
```

## 十八、相机与目标手臂路由

多相机版本中，InspectionCommand还应指定：

```text
primary_camera_source
secondary_camera_sources
target_arm
fusion_policy
```

推荐任务路由：

```text
开门：
  primary_camera_source: right_wrist
  secondary_camera_sources: [torso]
  target_arm: right

柜内操作：
  primary_camera_source: torso
  secondary_camera_sources: [right_wrist]
  target_arm: right

关门：
  primary_camera_source: torso
  secondary_camera_sources: []
  target_arm: left
```

这些多相机字段尚未加入当前InspectionCommand消息，加入后才能由运控程序正式发送。

## 十九、安全要求

Status必须满足：

```yaml
has_error: false
emergency_stop: false
target_reachable: true
```

Command建议默认值：

```yaml
selection_policy: highest_confidence
min_confidence: 0.30
required_stable_frames: 5
max_position_std_m: 0.003
timeout_sec: 10.0
```

执行规则：

1. Command与Status的`stage_id`必须一致。
2. 新动作必须使用新的`command_id`。
3. OCR语义不稳定时不得执行。
4. 深度无效时不得执行。
5. 多相机结果差异超过阈值时不得直接取平均。
6. 未实现的物理技能只能定位，不能执行。
