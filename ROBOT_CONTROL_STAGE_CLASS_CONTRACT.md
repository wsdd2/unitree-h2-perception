# 机器人运控 Stage 与 Class 接口合同

本文档面向机器人运控端，规定以下接口的使用方式：

```text
/robot/inspection_status   detector_msgs/msg/RobotInspectionStatus
/robot/inspection_command  detector_msgs/msg/InspectionCommand
```

两个消息必须使用相同的 `stage_id`。Status 和 Command 建议以 5 Hz
持续发布。同一请求重复发布时必须保持相同的 `command_id`，只有新请求才生成新 ID。

## 可请求的静态 class names

机器人运控必须使用以下最终发布名称：

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

不要请求以下仅供 YOLOE 使用的 Prompt 别名：

```text
green square work here safety tag
red twisted hanging cord rope
white adhesive cabinet wall hook
```

它们会在视觉节点内部归一化为上方稳定的工作牌类别名称。

## OCR 动态语义名称

OCR 不会替换 `class_name`，而是为具体实例增加语义身份：

```text
class_name: white toggle switch
label_text: 备用
semantic_name: white toggle switch/备用
control_id: white_toggle_switch/备用
```

使用以下字段匹配某个视觉类别的全部实例：

```text
requested_class_names
active_target_class_name
```

使用以下字段匹配一个由 OCR 文字区分的具体实例：

```text
requested_semantic_names
active_target_semantic_name
```

如果 OCR 为空或尚未稳定，语义匹配不得回退到另一个外观相同的开关。

## 支持的 requested_action

```text
observe_targets   仅观察，不产生物理动作
move              移动到目标预接近点
press             仅允许 lock point 或 push button
grasp_rotate      目前仅允许电柜门把手
door              电柜门把手开门序列别名
abort             取消当前自动化动作
```

旋钮、Rocker 和 Toggle 的物理操作技能尚未实现，目前只能使用
`observe_targets` 进行定位。

## Stage 0：空闲

Status：

```text
stage_id: 0
stage_name: idle
current_action: idle
motion_active: false
```

不需要发送 InspectionCommand。

## Stage 1：同时定位 Lock 和 Handle

Status：

```text
stage_id: 1
stage_name: locate_lock_and_handle
current_action: observe_targets
```

Command：

```text
requested_action: observe_targets
requested_class_names:
  - lock point
  - black cabinet door handle
active_target_class_name: ""
lock_target: false
```

预期结果：

```text
lock point:
  force_point_target_m

black cabinet door handle:
  handle_image_left_target_m
  handle_image_right_target_m
  handle_center_target_m
  handle_width_m
```

## Stage 2：戳按 lock point

Status：

```text
stage_id: 2
stage_name: press_lock_point
current_action: press_lock_point
target_id: lock_point_1
```

Command：

```text
requested_action: press
requested_class_names:
  - lock point
active_target_class_name: lock point
target_id: lock_point_1
lock_target: true
```

当前已支持物理执行。

## Stage 3：抓取、旋转并拉动电柜门把手

Status：

```text
stage_id: 3
stage_name: grasp_or_pull_handle
current_action: open_cabinet_door
target_id: cabinet_handle_1
```

Command：

```text
requested_action: grasp_rotate
requested_class_names:
  - black cabinet door handle
active_target_class_name: black cabinet door handle
target_id: cabinet_handle_1
lock_target: true
```

当前已支持五阶段开门动作。

## Stage 4：柜门已打开

Status：

```text
stage_id: 4
stage_name: door_opened
current_action: door_opened
motion_active: false
progress: 1.0
```

不需要发送新的 Command。

## Stage 5：恢复或中止

Status：

```text
stage_id: 5
stage_name: recover_or_abort
current_action: abort
has_error: true
target_reachable: false
```

Command：

```text
requested_action: abort
```

收到后将中止当前 Worker 动作。

## Stage 6：定位或抓取工作牌

Status：

```text
stage_id: 6
stage_name: pick_work_tag
current_action: locate_work_tag_grasp
```

定位 Command：

```text
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

Worker 尚未实现自动夹取动作。

## Stage 7：定位电柜挂钩

Status：

```text
stage_id: 7
stage_name: hang_work_tag
current_action: locate_cabinet_hook
```

Command：

```text
requested_action: observe_targets
requested_class_names:
  - cabinet hang hook
```

自动挂钩动作尚未实现。

## Stage 8-9

保留编号，正式分配前机器人运控不得使用。

## Stage 10：扫描柜内控件

红框第一排左侧四个为红/绿按钮，右侧两个为黑色旋转切换开关；
第二排三个为黑色旋转选择开关；第三排为黄/红/白 Toggle 开关。

Status：

```text
stage_id: 10
stage_name: inspect_inside_panel
current_action: observe_inside_controls
```

Command 示例：

```text
requested_action: observe_targets
requested_class_names:
  - red push button
  - green push button
  - black rotary selector switch
  - yellow toggle switch
  - red toggle switch
  - white toggle switch
```

只应填写当前画面预期可见的类别。所有请求类别都满足稳定性要求后，
本次观察请求才算完成。

## Stage 11：定位或读取表计

Status：

```text
stage_id: 11
stage_name: inspect_meter
current_action: observe_meter
```

每次只请求一个表计类别：

```text
square analog ammeter
square analog voltmeter
digital panel meter
round pressure gauge
```

使用 `observe_targets`。当前仅提供表计定位，尚未实现模拟表指针读数和
数字表 OCR 数值读取。

## Stage 12：戳按柜内按钮

Status 示例：

```text
stage_id: 12
stage_name: press_inside_button
current_action: press_red_button
target_id: red_button_1
```

Command 示例：

```text
requested_action: press
requested_class_names:
  - red push button
active_target_class_name: red push button
target_id: red_button_1
lock_target: true
```

允许执行的类别：

```text
red push button
green push button
yellow push button
black push button
```

当前已支持物理戳按。

按 OCR 语义选择时：

```text
requested_semantic_names:
  - red push button/停止
active_target_semantic_name: red push button/停止
```

## Stage 13：旋钮

Status：

```text
stage_id: 13
stage_name: inspect_rotary_selector
current_action: observe_rotary_selector
```

Command：

```text
requested_action: observe_targets
requested_class_names:
  - black rotary selector switch
```

第一排右侧两个旋钮使用：

```text
black rotary selector switch/top_row/with_tag
```

第二排三个旋钮使用：

```text
black rotary selector switch/middle_row/with_tag
```

OCR稳定后，`with_tag`会替换为实际标签文字，例如：

```text
black rotary selector switch/top_row/远方
black rotary selector switch/middle_row/手动
```

物理旋转技能尚未实现。禁止向旋钮发送 `grasp_rotate`，该动作当前仅用于
电柜门把手。

## Stage 14：Rocker / Toggle 开关

Status：

```text
stage_id: 14
stage_name: inspect_toggle_switch
current_action: observe_toggle_switch
```

允许定位的类别：

```text
black rocker switch
yellow toggle switch
red toggle switch
white toggle switch
```

使用 `observe_targets`。物理拨动技能尚未实现。可以使用 OCR 动态语义
区分外观相同的实例：

```text
white toggle switch/备用
white toggle switch/远方
white toggle switch/就地
```

## 必需的安全字段

Status 必须满足：

```text
has_error: false
emergency_stop: false
target_reachable: true
```

Command 建议默认值：

```text
selection_policy: highest_confidence
min_confidence: 0.30
required_stable_frames: 5
max_position_std_m: 0.003
timeout_sec: 10.0
```

Status 与 Command 的 `stage_id` 必须一致。每个新动作必须使用新的
`command_id`。
