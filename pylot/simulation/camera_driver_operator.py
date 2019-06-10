import threading

# ERDOS specific imports.
from erdos.op import Op
from erdos.utils import setup_logging
from erdos.message import WatermarkMessage
from erdos.timestamp import Timestamp

from pylot.perception.messages import SegmentedFrameMessage
import pylot.utils
from pylot.simulation.carla_utils import get_world, to_carla_transform
from pylot.simulation.utils import depth_to_array, labels_to_array, to_bgra_array, to_erdos_transform


class CameraDriverOperator(Op):
    """ CameraDriverOperator publishes images onto the desired stream from a camera.

    This operator attaches a vehicle at the required position with respect to
    the vehicle, registers callback functions to retrieve the images and
    publishes it to downstream operators.

    Attributes:
        _camera_setup: A CameraSetup tuple.
        _camera: Handle to the camera inside the simulation.
        _vehicle: Handle to the simulated hero vehicle.
    """
    def __init__(self,
                 name,
                 camera_setup,
                 flags,
                 log_file_name=None):
        """ Initializes the camera inside the simulation with the given
        parameters.

        Args:
            name: The unique name of the operator.
            camera_setup: A CameraSetup tuple.
            flags: A handle to the global flags instance to retrieve the
                configuration.
            log_file_name: The file to log the required information to.
        """
        super(CameraDriverOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._camera_setup = camera_setup

        _, self._world = get_world(self._flags.carla_host,
                                   self._flags.carla_port,
                                   self._flags.carla_timeout)
        if self._world is None:
            raise ValueError("There was an issue connecting to the simulator.")

        # The hero vehicle actor object we obtain from Carla.
        self._vehicle = None
        # The camera sensor actor object we obtain from Carla.
        self._camera = None
        # Lock to ensure that the callbacks do not execute simultaneously.
        self._lock = threading.Lock()

    @staticmethod
    def setup_streams(input_streams, camera_setup):
        """ Set up callback functions on the input streams and return the
        output stream that publishes the images.

        Args:
            input_streams: The streams that this operator is connected to.
            camera_setup: A CameraSetup tuple.
        """
        input_streams.filter(pylot.utils.is_ground_vehicle_id_stream)\
                     .add_callback(CameraDriverOperator.on_vehicle_id)
        return [pylot.utils.create_camera_stream(camera_setup)]

    def process_images(self, carla_image):
        """ Invoked when an image is received from the simulator.

        Args:
            carla_image: a carla.Image.
        """
        # Ensure that the code executes serially
        with self._lock:
            game_time = int(carla_image.timestamp * 1000)
            timestamp = Timestamp(coordinates=[game_time])
            watermark_msg = WatermarkMessage(timestamp)

            msg = None
            if self._camera_setup.camera_type == 'sensor.camera.rgb':
                msg = pylot.simulation.messages.FrameMessage(
                    pylot.utils.bgra_to_bgr(to_bgra_array(carla_image)),
                    timestamp)
            elif self._camera_setup.camera_type == 'sensor.camera.depth':
                msg = pylot.simulation.messages.DepthFrameMessage(
                    depth_to_array(carla_image),
                    to_erdos_transform(carla_image.transform),
                    carla_image.fov,
                    timestamp)
            elif self._camera_setup.camera_type == 'sensor.camera.semantic_segmentation':
                frame = labels_to_array(carla_image)
                msg = SegmentedFrameMessage(frame, 0, timestamp)
                # Send the message containing the frame.
            self.get_output_stream(self._camera_setup.name).send(msg)
            # XXX(ionel): The operator does not have to send watermarks.
            # They are automatically propagated from the input data stream.
            # This happens because the driver operator is not truly an input
            # operator. It receives vehicle id from the carla operator.
            #self.get_output_stream(self._camera_setup.name).send(watermark_msg)

    def on_vehicle_id(self, msg):
        """ This function receives the identifier for the vehicle, retrieves
        the handler for the vehicle from the simulation and attaches the
        camera to it.

        Args:
            msg: The identifier for the vehicle to attach the camera to.
        """
        vehicle_id = msg.data
        self._logger.info(
            "The CameraDriverOperator received the vehicle id: {}".format(
                vehicle_id))

        self._vehicle = self._world.get_actors().find(vehicle_id)
        if self._vehicle is None:
            raise ValueError("There was an issue finding the vehicle.")

        # Install the camera.
        camera_blueprint = self._world.get_blueprint_library().find(
                self._camera_setup.camera_type)

        camera_blueprint.set_attribute('image_size_x',
                                       str(self._camera_setup.width))
        camera_blueprint.set_attribute('image_size_y',
                                       str(self._camera_setup.height))

        transform = to_carla_transform(self._camera_setup.get_transform())

        self._logger.info("Spawning a camera: {}".format(self._camera_setup))

        self._camera = self._world.spawn_actor(camera_blueprint,
                                               transform,
                                               attach_to=self._vehicle)

        # Register the callback on the camera.
        self._camera.listen(self.process_images)
