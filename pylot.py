from absl import app
from absl import flags

import erdos.graph

import pylot.config
import pylot.operator_creator
import pylot.simulation.utils


FLAGS = flags.FLAGS
CENTER_CAMERA_NAME = 'front_rgb_camera'
LEFT_CAMERA_NAME = 'front_left_rgb_camera'
RIGHT_CAMERA_NAME = 'front_right_rgb_camera'
DEPTH_CAMERA_NAME = 'front_depth_camera'
SEGMENTED_CAMERA_NAME = 'front_semantic_camera'


def create_left_right_camera_setups():
    rotation = pylot.simulation.utils.Rotation(0, 0, 0)
    left_loc = pylot.simulation.utils.Location(1.5, -0.4, 1.4)
    right_loc = pylot.simulation.utils.Location(1.5, 0.4, 1.4)
    left_transform = pylot.simulation.utils.Transform(left_loc, rotation)
    right_transform = pylot.simulation.utils.Transform(right_loc, rotation)

    left_camera_setup = pylot.simulation.utils.CameraSetup(
        LEFT_CAMERA_NAME,
        'sensor.camera.rgb',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        left_transform)
    right_camera_setup = pylot.simulation.utils.CameraSetup(
        RIGHT_CAMERA_NAME,
        'sensor.camera.rgb',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        right_transform)
    return [left_camera_setup, right_camera_setup]


def create_camera_setups():
    location = pylot.simulation.utils.Location(1.5, 0.0, 1.4)
    rotation = pylot.simulation.utils.Rotation(0, 0, 0)
    transform = pylot.simulation.utils.Transform(location, rotation)
    bgr_camera_setup = pylot.simulation.utils.CameraSetup(
        CENTER_CAMERA_NAME,
        'sensor.camera.rgb',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform)
    depth_camera_setup = pylot.simulation.utils.CameraSetup(
        DEPTH_CAMERA_NAME,
        'sensor.camera.depth',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform)
    segmented_camera_setup = pylot.simulation.utils.CameraSetup(
        SEGMENTED_CAMERA_NAME,
        'sensor.camera.semantic_segmentation',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform)
    return [bgr_camera_setup, depth_camera_setup, segmented_camera_setup]


def create_lidar_setups():
    if FLAGS.lidar:
        location = pylot.simulation.utils.Location(1.5, 0.0, 1.4)
        rotation = pylot.simulation.utils.Rotation(0, 0, 0)
        lidar_transform = pylot.simulation.utils.Transform(location, rotation)
        return [pylot.simulation.utils.LidarSetup(
            name='front_center_lidar',
            lidar_type='sensor.lidar.ray_cast',
            transform=lidar_transform,
            range=5000,  # in centimers
            rotation_frequency=20,
            channels=32,
            upper_fov=15,
            lower_fov=-30,
            points_per_second=500000)]
    return []


def add_driver_operators(graph, auto_pilot):
    camera_setups = create_camera_setups()
    bgr_camera_setup = camera_setups[0]
    if FLAGS.depth_estimation:
        camera_setups = camera_setups + create_left_right_camera_setups()

    lidar_setups = create_lidar_setups()

    (carla_op,
     camera_ops,
     lidar_ops) = pylot.operator_creator.create_driver_ops(
         graph, camera_setups, lidar_setups, auto_pilot)
    return (bgr_camera_setup, carla_op, camera_ops, lidar_ops)


def add_ground_eval_ops(graph, perfect_det_ops, camera_ops):
    if FLAGS.eval_ground_truth_segmentation:
        eval_ground_seg_op = pylot.operator_creator.create_segmentation_ground_eval_op(
            graph, SEGMENTED_CAMERA_NAME)
        graph.connect(camera_ops, [eval_ground_seg_op])

    # This operator evaluates the temporal decay of the ground truth of
    # object detection across timestamps.
    if FLAGS.eval_ground_truth_object_detection:
        eval_ground_det_op = pylot.operator_creator.create_eval_ground_truth_detector_op(
            graph)
        graph.connect(perfect_det_ops, [eval_ground_det_op])


def add_detection_component(graph, bgr_camera_setup, camera_ops, carla_op):
    obj_detector_ops = []
    if FLAGS.obj_detection:
        obj_detector_ops = pylot.operator_creator.create_detector_ops(graph)
        graph.connect(camera_ops, obj_detector_ops)

        if FLAGS.evaluate_obj_detection:
            perfect_det_op = pylot.operator_creator.create_perfect_detector_op(
                graph, bgr_camera_setup, 'perfect_detector')
            graph.connect([carla_op] + camera_ops, [perfect_det_op])
            obstacle_accuracy_op = pylot.operator_creator.create_obstacle_accuracy_op(
                graph, 'perfect_detector')
            graph.connect(obj_detector_ops + [perfect_det_op],
                          [obstacle_accuracy_op])

        if FLAGS.obj_tracking:
            tracker_op = pylot.operator_creator.create_object_tracking_op(
                graph)
            graph.connect(camera_ops + obj_detector_ops, [tracker_op])

        if FLAGS.fusion:
            (fusion_op,
             fusion_verif_op) = pylot.operator_creator.create_fusion_ops(graph)
            graph.connect(obj_detector_ops + camera_ops + [carla_op],
                          [fusion_op])
            graph.connect([fusion_op, carla_op], [fusion_verif_op])

    # Currently, we keep this separate from the existing trackers, because the
    # perfect tracker returns ego-vehicle (x,y,z) coordinates, while our existing
    # trackers use camera coordinates.
    perfect_tracker_ops = []
    if FLAGS.perfect_tracking:
        perfect_tracker_ops = [pylot.operator_creator.create_perfect_tracking_op(
            graph, 'perfect_tracker')]
        graph.connect([carla_op], perfect_tracker_ops)

    traffic_light_det_ops = []
    if FLAGS.traffic_light_det:
        traffic_light_det_ops.append(
            pylot.operator_creator.create_traffic_light_op(graph))
        graph.connect(camera_ops, traffic_light_det_ops)

    lane_detection_ops = []
    if FLAGS.lane_detection:
        lane_detection_ops.append(
            pylot.operator_creator.create_lane_detection_op(graph))
        graph.connect(camera_ops, lane_detection_ops)
    return (obj_detector_ops, perfect_tracker_ops, traffic_light_det_ops, lane_detection_ops)


def add_segmentation_component(graph, camera_ops):
    segmentation_ops = []
    if FLAGS.segmentation_drn:
        segmentation_op = pylot.operator_creator.create_segmentation_drn_op(
            graph)
        segmentation_ops.append(segmentation_op)

    if FLAGS.segmentation_dla:
        segmentation_op = pylot.operator_creator.create_segmentation_dla_op(
            graph)
        segmentation_ops.append(segmentation_op)
    graph.connect(camera_ops, segmentation_ops)

    if FLAGS.evaluate_segmentation:
        eval_segmentation_op = pylot.operator_creator.create_segmentation_eval_op(
            graph, SEGMENTED_CAMERA_NAME, 'segmented_stream')
        graph.connect(camera_ops + segmentation_ops, [eval_segmentation_op])

    return segmentation_ops


def add_agent_op(graph,
                 carla_op,
                 traffic_light_det_ops,
                 obj_detector_ops,
                 segmentation_ops,
                 lane_detection_ops,
                 bgr_camera_setup):
    agent_op = None
    if FLAGS.ground_agent_operator:
        agent_op = pylot.operator_creator.create_ground_agent_op(graph)
        graph.connect([carla_op], [agent_op])
        graph.connect([agent_op], [carla_op])
    else:
        # TODO(ionel): The ERDOS agent doesn't use obj tracker and fusion.
        agent_op = pylot.operator_creator.create_pylot_agent_op(
            graph, bgr_camera_setup)
        input_ops = [carla_op] + traffic_light_det_ops + obj_detector_ops +\
                    segmentation_ops + lane_detection_ops
        graph.connect(input_ops, [agent_op])
        graph.connect([agent_op], [carla_op])
    return agent_op


def add_planning_component(graph,
                           goal_location,
                           goal_orientation,
                           carla_op,
                           agent_op,
                           city_name='Town01'):
    if '0.8' in FLAGS.carla_version:
        planning_op = pylot.operator_creator.create_legacy_planning_op(
            graph, city_name, goal_location, goal_orientation)
    elif '0.9' in FLAGS.carla_version:
        planning_op = pylot.operator_creator.create_planning_op(
            graph, goal_location)
        if FLAGS.visualize_waypoints:
            waypoint_viz_op = pylot.operator_creator.create_waypoint_visualizer_op(
                graph)
            graph.connect([planning_op], [waypoint_viz_op])
    else:
        raise ValueError('Unexpected Carla version')
    graph.connect([carla_op], [planning_op])
    graph.connect([planning_op], [agent_op])


def add_debugging_component(graph, carla_op, camera_ops, lidar_ops):
    # Add visual operators.
    pylot.operator_creator.add_visualization_operators(
        graph, camera_ops, lidar_ops, CENTER_CAMERA_NAME, DEPTH_CAMERA_NAME)

    # Add recording operators.
    pylot.operator_creator.add_recording_operators(graph,
                                                   camera_ops,
                                                   carla_op,
                                                   lidar_ops,
                                                   CENTER_CAMERA_NAME,
                                                   DEPTH_CAMERA_NAME)
    # Add operator that estimates depth.
    if FLAGS.depth_estimation:
        depth_estimation_op = pylot.operator_creator.create_depth_estimation_op(
            graph, LEFT_CAMERA_NAME, RIGHT_CAMERA_NAME)
        graph.connect(camera_ops + lidar_ops + [carla_op],
                      [depth_estimation_op])


def add_perfect_perception_component(graph,
                                     bgr_camera_setup,
                                     ground_obstacles_stream_name,
                                     lane_detection_stream_name,
                                     carla_op,
                                     camera_ops):
    obj_det_ops = [pylot.operator_creator.create_perfect_detector_op(
        graph, bgr_camera_setup, ground_obstacles_stream_name)]
    graph.connect([carla_op] + camera_ops, obj_det_ops)
    # TODO(ionel): Populate the other types of detectors.
    traffic_light_det_ops = []
    lane_det_ops = [
        pylot.operator_creator.create_perfect_lane_detector_op(
            graph, lane_detection_stream_name)
    ]
    graph.connect([carla_op], lane_det_ops)
    # Get the ground segmented frames from the driver operators.
    segmentation_ops = camera_ops
    return (obj_det_ops, traffic_light_det_ops, lane_det_ops, segmentation_ops)


def main(argv):
    # Define graph
    graph = erdos.graph.get_current_graph()

    # Add camera and lidar driver operators to the data-flow graph.
    (bgr_camera_setup,
     carla_op,
     camera_ops,
     lidar_ops) = add_driver_operators(
         graph, auto_pilot=FLAGS.carla_auto_pilot)

    # Add debugging operators (e.g., visualizers) to the data-flow graph.
    add_debugging_component(graph, carla_op, camera_ops, lidar_ops)

    if FLAGS.use_perfect_perception:
        # Add operators that use ground information.
        (obj_det_ops,
         traffic_light_det_ops,
         lane_det_ops,
         segmentation_ops) = add_perfect_perception_component(
             graph,
             bgr_camera_setup,
             'perfect_detector_output',
             'perfect_lane_detector_output',
             carla_op,
             camera_ops)
    else:
        # Add detectors.
        (obj_det_ops,
         perfect_tracker_ops,
         traffic_light_det_ops,
         lane_det_ops) = add_detection_component(
             graph, bgr_camera_setup, camera_ops, carla_op)

        # Add segmentation operators.
        segmentation_ops = add_segmentation_component(graph, camera_ops)

    add_ground_eval_ops(graph, obj_det_ops, camera_ops)

    # Add the behaviour planning agent operator.
    agent_op = add_agent_op(graph,
                            carla_op,
                            traffic_light_det_ops,
                            obj_det_ops,
                            segmentation_ops,
                            lane_det_ops,
                            bgr_camera_setup)

    # Add planning operators.
    goal_location = (234.269989014, 59.3300170898, 39.4306259155)
    goal_orientation = (1.0, 0.0, 0.22)
    add_planning_component(graph,
                           goal_location,
                           goal_orientation,
                           carla_op,
                           agent_op,
                           city_name='Town{:02d}'.format(FLAGS.carla_town))

    graph.execute(FLAGS.framework)


if __name__ == '__main__':
    app.run(main)
