from collections import namedtuple

import carla
from absl import app
from absl import flags

import erdos.graph
import pylot.config
import pylot.simulation.utils

from pylot.simulation.camera_driver_operator import CameraDriverOperator
from pylot.simulation.carla_scenario_operator import CarlaScenarioOperator
from pylot.simulation.perfect_pedestrian_detector_operator import \
        PerfectPedestrianDetectorOperator
from pylot.simulation.carla_utils import get_world
from pylot.simulation.mpc_planning_operator import MPCPlanningOperator

FLAGS = flags.FLAGS

Camera = namedtuple("Camera", "camera_setup, instance")


def add_camera_operator(graph):
    """ Adds the RGB and depth camera operator to the given graph and returns
    the setups and the instances of the operators added to the graph.

    Args:
        graph: The erdos.graph instance to add the operator to.

    Returns:
        A tuple containing the RGB and depth camera instances, which consists
        of the setup used to spawn the camera, and the operator instance
        itself.
    """
    # Create the camera setup needed to add the operator to the grpah.
    camera_location = pylot.simulation.utils.Location(1.5, 0.0, 1.4)
    camera_rotation = pylot.simulation.utils.Rotation(0, 0, 0)
    camera_transform = pylot.simulation.utils.Transform(
        camera_location, camera_rotation)

    # Create the RGB camera and add the operator to the graph.
    rgb_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.CENTER_CAMERA_NAME,
        'sensor.camera.rgb',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        camera_transform,
        fov=90)

    rgb_camera_operator = graph.add(
        CameraDriverOperator,
        name=rgb_camera_setup.name,
        init_args={
            'camera_setup': rgb_camera_setup,
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name
        },
        setup_args={'camera_setup': rgb_camera_setup})
    rgb_camera = Camera(camera_setup=rgb_camera_setup,
                        instance=rgb_camera_operator)

    # Create the depth camera and add the operator to the graph.
    depth_camera_setup = pylot.simulation.utils.CameraSetup(
        pylot.utils.DEPTH_CAMERA_NAME,
        'sensor.camera.depth',
        FLAGS.carla_camera_image_width,
        FLAGS.carla_camera_image_height,
        camera_transform,
        fov=90)
    depth_camera_operator = graph.add(
        CameraDriverOperator,
        name=depth_camera_setup.name,
        init_args={
            'camera_setup': depth_camera_setup,
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name
        },
        setup_args={'camera_setup': depth_camera_setup})
    depth_camera = Camera(camera_setup=depth_camera_setup,
                          instance=depth_camera_operator)

    return (rgb_camera, depth_camera)


def add_carla_operator(graph):
    """ Adds the Carla operator to the given graph and returns the graph.

    Args:
        graph: The erdos.graph instance to add the operator to.

    Returns:
        The operator instance depicting the Carla operator returned by the
        graph add method.
    """
    carla_operator = graph.add(CarlaScenarioOperator,
                               name='carla',
                               init_args={
                                   'role_name': 'hero',
                                   'flags': FLAGS,
                                   'log_file_name': FLAGS.log_file_name,
                               })
    return carla_operator


def set_asynchronous_mode(world):
    """ Sets the simulator to asynchronous mode.

    Args:
        world: The world instance of the simulator to set the asynchronous
            mode on.
    """
    settings = world.get_settings()
    settings.synchronous_mode = False
    world.apply_settings(settings)


def add_pedestrian_detector_operator(graph, camera_setup):
    """ Adds the perfect pedestrian detector operator to the graph, and returns
    the added operator.

    Args:
        graph: The erdos.graph instance to add the operator to.
        camera_setup: The camera setup to use for projecting the pedestrians
            onto the view of the camera.
    Returns:
        The operator instance depicting the PerfectPedestrianOperator returned
        by the graph add method.
    """
    pedestrian_detector_operator = graph.add(
        PerfectPedestrianDetectorOperator,
        name='perfect_pedestrian',
        init_args={
            'output_stream_name': 'perfect_pedestrian_bboxes',
            'camera_setup': camera_setup,
            'flags': FLAGS,
            'log_file_name': FLAGS.log_file_name
        },
        setup_args={'output_stream_name': 'perfect_pedestrian_bboxes'})
    return pedestrian_detector_operator


def add_planning_operator(graph, destination):
    mpc_planning_operator = graph.add(MPCPlanningOperator,
                                          name='mpc_planning',
                                          init_args={
                                              'goal': destination,
                                              'flags': FLAGS,
                                              'log_file_name':
                                              FLAGS.log_file_name
                                          })
    return mpc_planning_operator



def main(args):
    # Connect an instance to the simulator to make sure that we can turn the
    # synchronous mode off after the script finishes running.
    client, world = get_world(FLAGS.carla_host, FLAGS.carla_port,
                              FLAGS.carla_timeout)
    if client is None or world is None:
        raise ValueError("There was an issue connecting to the simulator.")

    try:
        # Define the ERDOS graph.
        graph = erdos.graph.get_current_graph()

        # Define the CARLA operator.
        carla_operator = add_carla_operator(graph)

        # Add the camera operator to the data-flow graph.
        rgb_camera, depth_camera = add_camera_operator(graph)
        graph.connect([carla_operator],
                      [rgb_camera.instance, depth_camera.instance])

        # Add a perfect pedestrian detector operator.
        object_detector_operator = add_pedestrian_detector_operator(
            graph, rgb_camera.camera_setup)
        graph.connect([carla_operator, depth_camera.instance],
                      [object_detector_operator])

        # Add a mpc planning operator.
        mpc_planning_operator = add_planning_operator(
            graph, carla.Location(x=17.73, y=327.07, z=0.5))
        graph.connect([carla_operator], [mpc_planning_operator])
        graph.connect([mpc_planning_operator], [carla_operator])

        graph.execute(FLAGS.framework)
    except KeyboardInterrupt:
        set_asynchronous_mode(world)
    except Exception as e:
        set_asynchronous_mode(world)
        raise e


if __name__ == "__main__":
    app.run(main)
