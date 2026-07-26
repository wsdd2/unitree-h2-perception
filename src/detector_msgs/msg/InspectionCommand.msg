std_msgs/Header header

# Unique command identity. A completed/rejected command_id must not be replayed.
string command_id

# High-level inspection stage associated with this request.
int32 stage_id
string stage_name

# observe_targets / move / press / grasp_rotate (handle only) / door / abort
string requested_action

# Classes that the requester wants reported for this stage.
string[] requested_class_names

# The one class allowed to drive physical motion. Empty means observe only.
string active_target_class_name

# Optional OCR-enriched identities, e.g. "white toggle switch/备用".
string[] requested_semantic_names
string active_target_semantic_name

# highest_confidence / nearest / specified_target_id
string selection_policy
string target_id

# Keep one selected detection identity until completion/abort.
bool lock_target

# Per-command target acceptance requirements.
float32 min_confidence
int32 required_stable_frames
float32 max_position_std_m
float32 timeout_sec
