import erdos
import threading
import time

from pylot.perception.messages import FrameMessage, DepthFrameMessage, \
    SegmentedFrameMessage
from pylot.simulation.utils import get_world, set_synchronous_mode
from pylot.perception.segmentation.segmented_frame import SegmentedFrame
from pylot.perception.camera_frame import CameraFrame
from pylot.perception.depth_frame import DepthFrame


class CarlaCameraDriverOperator(erdos.Operator):
    """Publishes images onto the desired stream from a camera.

    This operator attaches a vehicle at the required position with respect to
    the vehicle, registers callback functions to retrieve the images and
    publishes it to downstream operators.

    Args:
        ground_vehicle_id_stream (:py:class:`erdos.ReadStream`): Stream on
            which the operator receives the id of the ego vehicle. It uses this
            id to get a Carla handle to the vehicle.
        camera_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends camera frames.
        name (:obj:`str`): The name of the operator.
        camera_setup (:py:class:`pylot.drivers.sensor_setup.RGBCameraSetup`):
            Setup of the camera.
        flags (absl.flags): Object to be used to access absl flags.
        log_file_name (:obj:`str`, optional): Name of file where log messages
            are written to. If None, then messages are written to stdout.
    """
    def __init__(self,
                 ground_vehicle_id_stream,
                 camera_stream,
                 name,
                 camera_setup,
                 flags,
                 log_file_name=None):
        self._vehicle_id_stream = ground_vehicle_id_stream
        self._camera_stream = camera_stream
        self._name = name
        self._flags = flags
        self._logger = erdos.utils.setup_logging(name, log_file_name)
        self._camera_setup = camera_setup
        # The hero vehicle actor object we obtain from Carla.
        self._vehicle = None
        # The camera sensor actor object we obtain from Carla.
        self._camera = None
        # Lock to ensure that the callbacks do not execute simultaneously.
        self._lock = threading.Lock()

    @staticmethod
    def connect(ground_vehicle_id_stream):
        camera_stream = erdos.WriteStream()
        return [camera_stream]

    def run(self):
        # Read the vehicle id from the vehicle id stream
        vehicle_id_msg = self._vehicle_id_stream.read()
        vehicle_id = vehicle_id_msg.data
        self._logger.debug(
            "The CarlaCameraDriverOperator received the vehicle id: {}".format(
                vehicle_id))

        # Connect to the world. We connect here instead of in the constructor
        # to ensure we're connected to the latest world.
        _, world = get_world(self._flags.carla_host, self._flags.carla_port,
                             self._flags.carla_timeout)
        if world is None:
            raise ValueError("There was an issue connecting to the simulator.")
        if self._flags.carla_synchronous_mode:
            set_synchronous_mode(world, self._flags.carla_fps)

        num_tries = 0
        while self._vehicle is None and num_tries < 30:
            self._vehicle = world.get_actors().find(vehicle_id)
            self._logger.debug(
                "Could not find vehicle. Try {}".format(num_tries))
            time.sleep(1)
            num_tries += 1
        if self._vehicle is None:
            raise ValueError("There was an issue finding the vehicle.")

        # Install the camera.
        camera_blueprint = world.get_blueprint_library().find(
            self._camera_setup.camera_type)

        camera_blueprint.set_attribute('image_size_x',
                                       str(self._camera_setup.width))
        camera_blueprint.set_attribute('image_size_y',
                                       str(self._camera_setup.height))
        camera_blueprint.set_attribute('fov', str(self._camera_setup.fov))

        transform = self._camera_setup.get_transform().as_carla_transform()

        self._logger.debug("Spawning a camera: {}".format(self._camera_setup))

        self._camera = world.spawn_actor(camera_blueprint,
                                         transform,
                                         attach_to=self._vehicle)

        # Register the callback on the camera.
        self._camera.listen(self.process_images)
        # TODO: We might have to loop here to keep hold of the thread so that
        # Carla callbacks are still invoked.
        # while True:
        #     time.sleep(0.01)

    def process_images(self, carla_image):
        """ Invoked when an image is received from the simulator.

        Args:
            carla_image: a carla.Image.
        """
        # Ensure that the code executes serially
        with self._lock:
            game_time = int(carla_image.timestamp * 1000)
            timestamp = erdos.Timestamp(coordinates=[game_time])
            watermark_msg = erdos.WatermarkMessage(timestamp)

            msg = None
            if self._camera_setup.camera_type == 'sensor.camera.rgb':
                msg = FrameMessage(
                    timestamp,
                    CameraFrame.from_carla_frame(carla_image,
                                                 self._camera_setup))
            elif self._camera_setup.camera_type == 'sensor.camera.depth':
                # Include the transform relative to the vehicle.
                # Carla carla_image.transform returns the world transform, but
                # we do not use it directly.
                msg = DepthFrameMessage(
                    timestamp,
                    DepthFrame.from_carla_frame(carla_image,
                                                self._camera_setup))
            elif self._camera_setup.camera_type == \
                 'sensor.camera.semantic_segmentation':
                msg = SegmentedFrameMessage(
                    timestamp,
                    SegmentedFrame.from_carla_image(carla_image,
                                                    self._camera_setup))
            # Send the message containing the frame.
            self._camera_stream.send(msg)
            # Note: The operator is set not to automatically propagate
            # watermark messages received on input streams. Thus, we can
            # issue watermarks only after the Carla callback is invoked.
            self._camera_stream.send(watermark_msg)
