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
    left_camera_setup = pylot.simulation.utils.CameraSetup(
        LEFT_CAMERA_NAME,
        'sensor.camera.rgb',
        (FLAGS.carla_camera_image_width, FLAGS.carla_camera_image_height),
        (2.0, -0.4, 1.4))
    right_camera_setup = pylot.simulation.utils.CameraSetup(
        RIGHT_CAMERA_NAME,
        'sensor.camera.rgb',
        (FLAGS.carla_camera_image_width, FLAGS.carla_camera_image_height),
        (2.0, 0.4, 1.4))
    return [left_camera_setup, right_camera_setup]


def create_camera_setups(position=(2.0, 0.0, 1.4)):
    rgb_camera_setup = pylot.simulation.utils.CameraSetup(
        CENTER_CAMERA_NAME,
        'sensor.camera.rgb',
        (FLAGS.carla_camera_image_width, FLAGS.carla_camera_image_height),
        position)
    depth_camera_setup = pylot.simulation.utils.CameraSetup(
        DEPTH_CAMERA_NAME,
        'sensor.camera.depth',
        (FLAGS.carla_camera_image_width, FLAGS.carla_camera_image_height),
        position)
    segmented_camera_setup = pylot.simulation.utils.CameraSetup(
        SEGMENTED_CAMERA_NAME,
        'sensor.camera.semantic_segmentation',
        (FLAGS.carla_camera_image_width, FLAGS.carla_camera_image_height),
        position)
    return (rgb_camera_setup, depth_camera_setup, segmented_camera_setup)


def main(argv):
    # Define graph
    graph = erdos.graph.get_current_graph()

    (rgb_camera_setup,
     depth_camera_setup,
     segmented_camera_setup) = create_camera_setups()
    camera_setups = [rgb_camera_setup,
                     depth_camera_setup,
                     segmented_camera_setup]
    if FLAGS.depth_estimation:
        camera_setups = camera_setups + create_left_right_camera_setups()

    lidar_setups = []
    if FLAGS.lidar:
        lidar_setup = pylot.simulation.utils.LidarSetup(
            name='front_center_lidar',
            type='sensor.lidar.ray_cast',
            pos=(2.0, 0.0, 1.4),
            range=5000,  # in centimers
            rotation_frequency=20,
            channels=32,
            upper_fov=15,
            lower_fov=-30,
            points_per_second=500000)
        lidar_setups.append(lidar_setup)

    # Add operators to the graph.
    camera_ops = []
    lidar_ops = []
    if '0.8' in FLAGS.carla_version:
        carla_op = pylot.operator_creator.create_carla_legacy_op(
            graph, camera_setups, lidar_setups)
        camera_ops = [carla_op]
    elif '0.9' in FLAGS.carla_version:
        carla_op = pylot.operator_creator.create_carla_op(graph)
        camera_ops = [pylot.operator_creator.create_camera_driver_op(graph, cs)
                      for cs in camera_setups]
        lidar_ops = [pylot.operator_creator.create_lidar_driver_op(graph, ls)
                     for ls in lidar_setups]
        graph.connect([carla_op], camera_ops + lidar_ops)
    else:
        raise ValueError(
            'Unexpected Carla version {}'.format(FLAGS.carla_version))

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

    if FLAGS.eval_ground_truth_segmentation:
        eval_ground_seg_op = pylot.operator_creator.create_segmentation_ground_eval_op(
            graph, SEGMENTED_CAMERA_NAME)
        graph.connect(camera_ops, [eval_ground_seg_op])

    # This operator evaluates the temporal decay of the ground truth of
    # object detection across timestamps.
    if FLAGS.eval_ground_truth_object_detection:
        eval_ground_det_op = pylot.operator_creator.create_eval_ground_truth_detector_op(
            graph, rgb_camera_setup, DEPTH_CAMERA_NAME)
        graph.connect([carla_op] + camera_ops, [eval_ground_det_op])

    obj_detector_ops = []
    if FLAGS.obj_detection:
        obj_detector_ops = pylot.operator_creator.create_detector_ops(graph)
        graph.connect(camera_ops, obj_detector_ops)

        if FLAGS.evaluate_obj_detection:
            obstacle_accuracy_op = pylot.operator_creator.create_obstacle_accuracy_op(
                graph, rgb_camera_setup, DEPTH_CAMERA_NAME)
            graph.connect(obj_detector_ops + [carla_op] + camera_ops,
                          [obstacle_accuracy_op])

        if FLAGS.obj_tracking:
            tracker_op = pylot.operator_creator.create_object_tracking_op(
                graph)
            graph.connect(camera_ops + obj_detector_ops, [tracker_op])

        if FLAGS.fusion:
            (fusion_op, fusion_verification_op) = pylot.operator_creator.create_fusion_ops(graph)
            graph.connect(obj_detector_ops + camera_ops + [carla_op],
                          [fusion_op])
            graph.connect([fusion_op, carla_op], [fusion_verification_op])

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

    if FLAGS.depth_estimation:
        depth_estimation_op = pylot.operator_creator.create_depth_estimation_op(
            graph, LEFT_CAMERA_NAME, RIGHT_CAMERA_NAME)
        graph.connect(camera_ops + lidar_ops + [carla_op],
                      [depth_estimation_op])

    agent_op = None
    if FLAGS.ground_agent_operator:
        agent_op = pylot.operator_creator.create_ground_agent_op(graph)
        graph.connect([carla_op], [agent_op])
        graph.connect([agent_op], [carla_op])
    else:
        # TODO(ionel): The ERDOS agent doesn't use obj tracker and fusion.
        agent_op = pylot.operator_creator.create_erdos_agent_op(
            graph, DEPTH_CAMERA_NAME)
        input_ops = [carla_op] + traffic_light_det_ops + obj_detector_ops +\
                    segmentation_ops + lane_detection_ops
        graph.connect(input_ops, [agent_op])
        graph.connect([agent_op], [carla_op])

    goal_location = (234.269989014, 59.3300170898, 39.4306259155)
    goal_orientation = (1.0, 0.0, 0.22)

    if '0.8' in FLAGS.carla_version:
        waypointer_op = pylot.operator_creator.create_waypointer_op(
            graph, goal_location, goal_orientation)
        graph.connect([carla_op], [waypointer_op])
        graph.connect([waypointer_op], [agent_op])
    elif '0.9' in FLAGS.carla_version:
        planning_op = pylot.operator_creator.create_planning_op(
            graph, goal_location)
        graph.connect([carla_op], [planning_op])
        graph.connect([planning_op], [agent_op])

        if FLAGS.visualize_waypoints:
            waypoint_viz_op = pylot.operator_creator.create_waypoint_visualizer_op(
                graph)
            graph.connect([planning_op], [waypoint_viz_op])

    graph.execute(FLAGS.framework)


if __name__ == '__main__':
    app.run(main)
