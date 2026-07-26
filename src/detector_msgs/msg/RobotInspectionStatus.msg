std_msgs/Header header

# High-level inspection stage reported by robot/motion control.
# Suggested convention:
# 0 idle/unknown
# 1 move_to_handle_front
# 2 press_blue_point
# 3 grasp_or_pull_handle
# 4 door_opened
# 5 recover_or_abort
# 6 pick_work_tag / grasp_work_tag  -> /detector target = work tag grasp point
# 7 hang_work_tag / hang_on_hook    -> /detector target = cabinet hang hook
# 8-9 reserved
# 10 inspect_inside_panel
# 11 inspect/read meter (localization only until reader is implemented)
# 12 press_inside_button
# 13 inspect/operate rotary selector (physical operation not implemented)
# 14 inspect/operate rocker/toggle switch (physical operation not implemented)
int32 stage_id
string stage_name

# Current motion/control state.
string current_action
bool motion_active
float32 progress

# Robot-side health and reachability.
bool has_error
string error_code
string error_message
bool emergency_stop
bool target_reachable
string reachability_message

# Optional target identifier the robot is acting on, e.g. object_id/class_name.
string target_id
