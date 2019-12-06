from absl import app
from absl import flags

import erdos.graph

import pylot.config
import pylot.operator_creator
import pylot.simulation.utils
import pylot.utils

FLAGS = flags.FLAGS

CENTER_CAMERA_LOCATION_X = 1.5
CENTER_CAMERA_LOCATION_Y = 0.0
CENTER_CAMERA_LOCATION_Z = 1.4
CENTER_CAMERA_LOCATION = pylot.simulation.utils.Location(
    CENTER_CAMERA_LOCATION_X,
    CENTER_CAMERA_LOCATION_Y,
    CENTER_CAMERA_LOCATION_Z)


def create_left_right_camera_setups():
    rotation = pylot.simulation.utils.Rotation(0, 0, 0)
    left_loc = pylot.simulation.utils.Location(CENTER_CAMERA_LOCATION_X,
                                               CENTER_CAMERA_LOCATION_Y - 0.4,
                                               CENTER_CAMERA_LOCATION_Z)
    right_loc = pylot.simulation.utils.Location(CENTER_CAMERA_LOCATION_X,
                                                CENTER_CAMERA_LOCATION_Y + 0.4,
                                                CENTER_CAMERA_LOCATION_Z)
    left_transform = pylot.simulation.utils.Transform(left_loc, rotation)
    right_transform = pylot.simulation.utils.Transform(right_loc, rotation)

    left_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.LEFT_CAMERA_NAME,
        'sensor.camera.rgb',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        left_transform)
    right_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.RIGHT_CAMERA_NAME,
        'sensor.camera.rgb',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        right_transform)
    return [left_camera_setup, right_camera_setup]


def create_top_down_segmentation_setups():
    # Height calculation relies on the fact that the camera's FOV is 90.
    location = pylot.simulation.utils.Location(
        CENTER_CAMERA_LOCATION_X,
        CENTER_CAMERA_LOCATION_Y,
        CENTER_CAMERA_LOCATION_Z + FLAGS.top_down_lateral_view)
    rotation = pylot.simulation.utils.Rotation(-90, 0, 0)
    transform = pylot.simulation.utils.Transform(location, rotation)
    top_down_segmented_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.TOP_DOWN_SEGMENTED_CAMERA_NAME,
        'sensor.camera.semantic_segmentation',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform,
        fov=90)
    return top_down_segmented_camera_setup


def create_camera_setups():
    rotation = pylot.simulation.utils.Rotation(0, 0, 0)
    transform = pylot.simulation.utils.Transform(
        CENTER_CAMERA_LOCATION, rotation)
    bgr_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.CENTER_CAMERA_NAME,
        'sensor.camera.rgb',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform)
    depth_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.DEPTH_CAMERA_NAME,
        'sensor.camera.depth',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform)
    front_segmented_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.FRONT_SEGMENTED_CAMERA_NAME,
        'sensor.camera.semantic_segmentation',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        transform)
    return [bgr_camera_setup, depth_camera_setup, front_segmented_camera_setup]


def create_lidar_setups():
    if FLAGS.lidar:
        rotation = pylot.simulation.utils.Rotation(0, 0, 0)
        # Place the lidar in the same position as the camera.
        lidar_transform = pylot.simulation.utils.Transform(
            CENTER_CAMERA_LOCATION, rotation)
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
    top_down_segmentation_setup = None
    if FLAGS.depth_estimation:
        camera_setups = camera_setups + create_left_right_camera_setups()
    if FLAGS.top_down_segmentation or FLAGS.visualize_top_down_tracker_output:
        top_down_segmentation_setup = create_top_down_segmentation_setups()
        camera_setups = camera_setups + [top_down_segmentation_setup]

    lidar_setups = create_lidar_setups()

    (carla_op,
     camera_ops,
     lidar_ops) = pylot.operator_creator.create_driver_ops(
         graph, camera_setups, lidar_setups, auto_pilot)
    return (bgr_camera_setup,
            top_down_segmentation_setup,
            carla_op,
            camera_ops,
            lidar_ops)


def add_ground_eval_ops(graph, perfect_det_ops, camera_ops):
    if FLAGS.eval_ground_truth_segmentation:
        eval_ground_seg_op = pylot.operator_creator.create_segmentation_ground_eval_op(
            graph, pylot.utils.FRONT_SEGMENTED_CAMERA_NAME)
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
    # perfect tracker returns ego-vehicle (x,y,z) coordinates, while our
    # existing trackers use camera coordinates.
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

    graph.connect(camera_ops, segmentation_ops)

    if FLAGS.evaluate_segmentation:
        eval_segmentation_op = pylot.operator_creator.create_segmentation_eval_op(
            graph, pylot.utils.FRONT_SEGMENTED_CAMERA_NAME, 'segmented_stream')
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
    elif FLAGS.mpc_agent_operator:
        agent_op = pylot.operator_creator.create_mpc_agent_op(graph)
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
                           carla_op,
                           agent_op,
                           prediction_op):
    if FLAGS.waypoint_planning_operator:
        planning_op = pylot.operator_creator.create_waypoint_planning_op(graph, goal_location)
    elif FLAGS.rrt_star_planning_operator:
        planning_op = pylot.operator_creator.create_rrt_star_planning_op(graph, goal_location)
    else:
        planning_op = pylot.operator_creator.create_planning_op(
            graph, goal_location)

    if FLAGS.visualize_planning:
        planning_viz_op = pylot.operator_creator.create_planning_visualizer_op(graph)
        graph.connect([planning_op], [planning_viz_op])

    if FLAGS.prediction:
        graph.connect(prediction_op, [planning_op])

    graph.connect([carla_op], [planning_op])
    graph.connect([planning_op], [agent_op])


def add_debugging_component(graph, top_down_camera_setup, carla_op, camera_ops,
                            lidar_ops, perfect_tracker_ops, prediction_ops):
    # Add visual operators.
    pylot.operator_creator.add_visualization_operators(
        graph,
        camera_ops,
        lidar_ops,
        perfect_tracker_ops,
        prediction_ops,
        pylot.utils.CENTER_CAMERA_NAME,
        pylot.utils.DEPTH_CAMERA_NAME,
        pylot.utils.FRONT_SEGMENTED_CAMERA_NAME,
        pylot.utils.TOP_DOWN_SEGMENTED_CAMERA_NAME,
        top_down_camera_setup)

    # Add recording operators.
    pylot.operator_creator.add_recording_operators(
        graph,
        camera_ops,
        carla_op,
        lidar_ops,
        pylot.utils.CENTER_CAMERA_NAME,
        pylot.utils.DEPTH_CAMERA_NAME)

    # Add operator that estimates depth.
    if FLAGS.depth_estimation:
        depth_estimation_op = pylot.operator_creator.create_depth_estimation_op(
            graph, pylot.utils.LEFT_CAMERA_NAME, pylot.utils.RIGHT_CAMERA_NAME)
        graph.connect(camera_ops + lidar_ops + [carla_op],
                      [depth_estimation_op])

    # Add operator that visualizes CanBus
    if FLAGS.visualize_can_bus:
        can_bus_viz_op = pylot.operator_creator.create_can_bus_visualizer(graph)
        graph.connect([carla_op], [can_bus_viz_op])


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


def add_prediction_component(graph,
                             perfect_tracker_ops,
                             prediction_stream_name):
    prediction_ops = []
    if FLAGS.prediction_type == 'linear':
        prediction_ops.append(pylot.operator_creator.create_linear_predictor_op(
            graph, prediction_stream_name))
        graph.connect(perfect_tracker_ops, prediction_ops)

    return prediction_ops


def main(argv):
    # Define graph
    graph = erdos.graph.get_current_graph()

    # Add camera and lidar driver operators to the data-flow graph.
    (bgr_camera_setup,
     top_down_camera_setup,
     carla_op,
     camera_ops,
     lidar_ops) = add_driver_operators(
         graph, auto_pilot=FLAGS.carla_auto_pilot)

    perfect_tracker_ops = []
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

    prediction_ops = []
    if FLAGS.prediction:
        prediction_ops = add_prediction_component(graph, perfect_tracker_ops, 'prediction_output')

    add_ground_eval_ops(graph, obj_det_ops, camera_ops)

    # Add debugging operators (e.g., visualizers) to the data-flow graph.
    add_debugging_component(graph, top_down_camera_setup, carla_op, camera_ops,
                            lidar_ops, perfect_tracker_ops, prediction_ops)

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
    add_planning_component(graph, goal_location, carla_op, agent_op, prediction_ops)

    graph.execute("ros")


if __name__ == '__main__':
    app.run(main)
