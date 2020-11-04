from absl import flags

# Waypoint visualization flags.
flags.DEFINE_bool('draw_waypoints_on_world', True,
                  'True to enable drawing on the simulator world')
flags.DEFINE_bool('draw_waypoints_on_camera_frames', False,
                  'True to enable drawing on camera frames')
