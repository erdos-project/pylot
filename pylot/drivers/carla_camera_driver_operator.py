"""This module implements an operator acts like a camera driver when
using the simulator.

The operator attaches a camera to the ego vehicle, receives camera frames from
the simulator, and sends them on its output stream.
"""

import pickle
import threading

import erdos

from pylot.perception.camera_frame import CameraFrame
from pylot.perception.depth_frame import DepthFrame
from pylot.perception.messages import DepthFrameMessage, FrameMessage, \
    SegmentedFrameMessage
from pylot.perception.segmentation.segmented_frame import SegmentedFrame
from pylot.simulation.utils import get_vehicle_handle, get_world, \
    set_simulation_mode


class CarlaCameraDriverOperator(erdos.Operator):
    """Publishes images onto the desired stream from a camera.

    This operator attaches a vehicle at the required position with respect to
    the vehicle, registers callback functions to retrieve the images and
    publishes it to downstream operators.

    Args:
        ground_vehicle_id_stream (:py:class:`erdos.ReadStream`): Stream on
            which the operator receives the id of the ego vehicle. It uses this
            id to get a simulator handle to the vehicle.
        camera_stream (:py:class:`erdos.WriteStream`): Stream on which the
            operator sends camera frames.
        notify_reading_stream (:py:class:`erdos.WriteStream`): Stream on which
            the operator sends notifications when it receives camera frames.
        camera_setup (:py:class:`pylot.drivers.sensor_setup.RGBCameraSetup`):
            Setup of the camera.
        flags (absl.flags): Object to be used to access absl flags.
    """
    def __init__(self, ground_vehicle_id_stream: erdos.ReadStream,
                 release_sensor_stream: erdos.ReadStream,
                 camera_stream: erdos.WriteStream,
                 notify_reading_stream: erdos.WriteStream, camera_setup,
                 flags):
        erdos.add_watermark_callback([release_sensor_stream], [],
                                     self.release_data)
        self._vehicle_id_stream = ground_vehicle_id_stream
        self._camera_stream = camera_stream
        self._notify_reading_stream = notify_reading_stream
        self._flags = flags
        self._logger = erdos.utils.setup_logging(self.config.name,
                                                 self.config.log_file_name)
        self._camera_setup = camera_setup
        # The hero vehicle actor object we obtain from the simulator.
        self._vehicle = None
        # The camera sensor actor object we obtain from the simulator.
        self._camera = None
        self._pickle_lock = threading.Lock()
        self._pickled_messages = {}
        # Lock to ensure that the callbacks do not execute simultaneously.
        self._lock = threading.Lock()
        # If false then the operator does not send data until it receives
        # release data watermark. Otherwise, it sends as soon as it
        # receives it.
        self._release_data = False

    @staticmethod
    def connect(ground_vehicle_id_stream: erdos.ReadStream,
                release_sensor_stream: erdos.ReadStream):
        camera_stream = erdos.WriteStream()
        notify_reading_stream = erdos.WriteStream()
        return [camera_stream, notify_reading_stream]

    @erdos.profile_method()
    def release_data(self, timestamp):
        if timestamp.is_top:
            # The operator can always send data ASAP.
            self._release_data = True
        else:
            self._logger.debug("@{}: {} releasing sensor data".format(
                timestamp, self.config.name))
            watermark_msg = erdos.WatermarkMessage(timestamp)
            self._camera_stream.send_pickled(timestamp,
                                             self._pickled_messages[timestamp])
            # Note: The operator is set not to automatically propagate
            # watermark messages received on input streams. Thus, we can
            # issue watermarks only after the simulator callback is invoked.
            self._camera_stream.send(watermark_msg)
            with self._pickle_lock:
                del self._pickled_messages[timestamp]

    def run(self):
        # Read the vehicle id from the vehicle id stream
        vehicle_id_msg = self._vehicle_id_stream.read()
        vehicle_id = vehicle_id_msg.data
        self._logger.debug(
            "The CameraDriverOperator received the vehicle id: {}".format(
                vehicle_id))

        # Connect to the world. We connect here instead of in the constructor
        # to ensure we're connected to the latest world.
        _, world = get_world(self._flags.simulator_host,
                             self._flags.simulator_port,
                             self._flags.simulator_timeout)
        set_simulation_mode(world, self._flags)

        self._vehicle = get_vehicle_handle(world, vehicle_id)

        # Install the camera.
        camera_blueprint = world.get_blueprint_library().find(
            self._camera_setup.camera_type)
        camera_blueprint.set_attribute('image_size_x',
                                       str(self._camera_setup.width))
        camera_blueprint.set_attribute('image_size_y',
                                       str(self._camera_setup.height))
        camera_blueprint.set_attribute('fov', str(self._camera_setup.fov))
        if self._flags.simulator_camera_frequency == -1:
            camera_blueprint.set_attribute('sensor_tick', '0.0')
        else:
            camera_blueprint.set_attribute(
                'sensor_tick',
                str(1.0 / self._flags.simulator_camera_frequency))

        transform = self._camera_setup.get_transform().as_simulator_transform()

        self._logger.debug("Spawning a camera: {}".format(self._camera_setup))
        self._camera = world.spawn_actor(camera_blueprint,
                                         transform,
                                         attach_to=self._vehicle)

        # Register the callback on the camera.
        self._camera.listen(self.process_images)

    def process_images(self, simulator_image):
        """Invoked when an image is received from the simulator."""
        game_time = int(simulator_image.timestamp * 1000)
        timestamp = erdos.Timestamp(coordinates=[game_time])
        watermark_msg = erdos.WatermarkMessage(timestamp)
        with erdos.profile(self.config.name + '.process_images',
                           self,
                           event_data={'timestamp': str(timestamp)}):
            # Ensure that the code executes serially
            with self._lock:
                msg = None
                if self._camera_setup.camera_type == 'sensor.camera.rgb':
                    msg = FrameMessage(
                        timestamp,
                        CameraFrame.from_simulator_frame(
                            simulator_image, self._camera_setup))
                elif self._camera_setup.camera_type == 'sensor.camera.depth':
                    # Include the transform relative to the vehicle.
                    # simulator_image.transform returns the world transform,
                    # but we do not use it directly.
                    msg = DepthFrameMessage(
                        timestamp,
                        DepthFrame.from_simulator_frame(
                            simulator_image,
                            self._camera_setup,
                            save_original_frame=self._flags.
                            visualize_depth_camera))
                elif (self._camera_setup.camera_type ==
                      'sensor.camera.semantic_segmentation'):
                    msg = SegmentedFrameMessage(
                        timestamp,
                        SegmentedFrame.from_simulator_image(
                            simulator_image, self._camera_setup))

                if self._release_data:
                    self._camera_stream.send(msg)
                    self._camera_stream.send(watermark_msg)
                else:
                    # Pickle the data, and release it upon release msg receipt.
                    pickled_msg = pickle.dumps(
                        msg, protocol=pickle.HIGHEST_PROTOCOL)
                    with self._pickle_lock:
                        self._pickled_messages[msg.timestamp] = pickled_msg
                    self._notify_reading_stream.send(watermark_msg)
