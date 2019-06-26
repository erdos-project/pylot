from collections import deque
import math
import numpy as np
import time
from pid_controller.pid import PID

from erdos.op import Op
from erdos.utils import frequency, setup_csv_logging, setup_logging, time_epoch_ms

from pylot.control.messages import ControlMessage
import pylot.control.utils
from pylot.simulation.utils import get_3d_world_position_with_depth_map
import pylot.utils


class ERDOSAgentOperator(Op):
    def __init__(self,
                 name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(ERDOSAgentOperator, self).__init__(name)
        self._flags = flags
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._pid = PID(p=self._flags.pid_p,
                        i=self._flags.pid_i,
                        d=self._flags.pid_d)
        self._can_bus_msgs = deque()
        self._waypoint_msgs = deque()
        self._depth_msgs = deque()
        self._traffic_lights = deque()
        self._obstacles = deque()

    @staticmethod
    def setup_streams(input_streams, depth_camera_name):
        input_streams.filter_name(depth_camera_name).add_callback(
            ERDOSAgentOperator.on_depth_camera_update)

        # XXX(ionel): We get the exact position from the simulator.
        input_streams.filter(
            pylot.utils.is_can_bus_stream).add_callback(
                ERDOSAgentOperator.on_can_bus_update)
        input_streams.filter(
            pylot.utils.is_waypoints_stream).add_callback(
                ERDOSAgentOperator.on_waypoints_update)
        input_streams.filter(
            pylot.utils.is_traffic_lights_stream).add_callback(
                ERDOSAgentOperator.on_traffic_lights_update)
        input_streams.filter(
            pylot.utils.is_segmented_camera_stream).add_callback(
                ERDOSAgentOperator.on_segmented_frame)
        input_streams.filter(
            pylot.utils.is_obstacles_stream).add_callback(
                ERDOSAgentOperator.on_obstacles_update)
        input_streams.filter(
            pylot.utils.is_detected_lane_stream).add_callback(
                ERDOSAgentOperator.on_detected_lane_update)

        input_streams.add_completion_callback(
            ERDOSAgentOperator.on_notification)

        # Set no watermark on the output stream so that we do not
        # close the watermark loop with the carla operator.
        return [pylot.utils.create_control_stream()]

    def on_notification(self, msg):
        # Get can bus information.
        can_bus_msg = self._can_bus_msgs.popleft()
        vehicle_transform = can_bus_msg.data.transform
        vehicle_speed = can_bus_msg.data.forward_speed

        # Get waypoints.
        waypoint_msg = self._waypoint_msgs.popleft()
        wp_angle = waypoint_msg.wp_angle
        wp_vector = waypoint_msg.wp_vector
        wp_angle_speed = waypoint_msg.wp_angle_speed

        depth_msg = self._depth_msgs.popleft()

        # Transform traffic light output.
        tl_det_msg = self._traffic_lights.popleft()
        traffic_lights = self.__transform_tl_output(
            tl_det_msg, depth_msg, vehicle_transform)

        # Transform detector output.
        det_msg = self._obstacles.popleft()
        (pedestrians, vehicles) = self.__transform_detector_output(
            det_msg, depth_msg, vehicle_transform)

        self._logger.info('Current location {}'.format(vehicle_transform))
        self._logger.info('Pedestrians {}'.format(pedestrians))
        self._logger.info('Vehicles {}'.format(vehicles))

        self._logger.info("Timestamps {} {} {} {}".format(
            can_bus_msg.timestamp,
            tl_det_msg.timestamp,
            depth_msg.timestamp,
            det_msg.timestamp))

        speed_factor, state = self.__stop_for_agents(
            vehicle_transform,  wp_angle, wp_vector, vehicles,
            pedestrians, traffic_lights)

        control_msg = self.get_control_message(
            wp_angle, wp_angle_speed, speed_factor,
            vehicle_speed, msg.timestamp)
        self.get_output_stream('control_stream').send(control_msg)

    def on_waypoints_update(self, msg):
        self._waypoint_msgs.append(msg)

    def on_can_bus_update(self, msg):
        self._can_bus_msgs.append(msg)

    def on_depth_camera_update(self, msg):
        self._depth_msgs.append(msg)

    def on_segmented_frame(self, msg):
        self._logger.info("Received segmented frame update at {}".format(msg.timestamp))
        # TODO(ionel): Implement!

    def on_traffic_lights_update(self, msg):
        self._logger.info("Received traffic light update at {}".format(msg.timestamp))
        self._traffic_lights.append(msg)

    def on_obstacles_update(self, msg):
        self._logger.info("Received obstacle update at {}".format(msg.timestamp))
        self._obstacles.append(msg)

    def on_detected_lane_update(self, msg):
        # TODO(ionel): Implement!
        pass

    def execute(self):
        self.spin()

    def __transform_tl_output(self, tl_msg, depth_msg, vehicle_transform):
        traffic_lights = []
        for tl in tl_msg.detected_objects:
            x = (tl.corners[0] + tl.corners[1]) / 2
            y = (tl.corners[2] + tl.corners[3]) / 2
            pos = get_3d_world_position_with_depth_map(
                x, y, depth_msg.frame, depth_msg.width, depth_msg.height,
                depth_msg.fov, depth_msg.transform * vehicle_transform)
            state = 0
            if tl.label is not 'Green':
                state = 1
            traffic_lights.append((pos, state))
        return traffic_lights

    def __transform_detector_output(
            self, det_msg, depth_msg, vehicle_transform):
        vehicles = []
        pedestrians = []
        for detected_obj in det_msg.detected_objects:
            x = (detected_obj.corners[0] + detected_obj.corners[1]) / 2
            y = (detected_obj.corners[2] + detected_obj.corners[3]) / 2
            if detected_obj.label == 'person':
                pos = get_3d_world_position_with_depth_map(
                    x, y, depth_msg.frame, depth_msg.width, depth_msg.height,
                    depth_msg.fov, depth_msg.transform * vehicle_transform)
                pedestrians.append(pos)
            elif (detected_obj.label == 'car' or
                  detected_obj.label == 'bicycle' or
                  detected_obj.label == 'motorcycle' or
                  detected_obj.label == 'bus' or
                  detected_obj.label == 'truck'):
                pos = get_3d_world_position_with_depth_map(
                    x, y, depth_msg.frame, depth_msg.width, depth_msg.height,
                    depth_msg.fov, depth_msg.transform * vehicle_transform)
                vehicles.append(pos)
        return (pedestrians, vehicles)

    def __stop_for_agents(self,
                          vehicle_transform,
                          wp_angle,
                          wp_vector,
                          vehicles,
                          pedestrians,
                          traffic_lights):
        speed_factor = 1
        speed_factor_tl = 1
        speed_factor_p = 1
        speed_factor_v = 1

        for obs_vehicle_pos in vehicles:
            if pylot.control.utils.is_vehicle_on_same_lane(
                    vehicle_transform, obs_vehicle_pos):
                new_speed_factor_v = pylot.control.utils.stop_vehicle(
                    vehicle_transform, obs_vehicle_pos, wp_vector,
                    speed_factor_v, self._flags)
                speed_factor_v = min(speed_factor_v, new_speed_factor_v)

        for obs_ped_pos in pedestrians:
            if pylot.control.utils.is_pedestrian_hitable(
                    vehicle_transform, obs_ped_pos):
                new_speed_factor_p = pylot.control.utils.stop_pedestrian(
                    vehicle_transform,
                    obs_ped_pos,
                    wp_vector,
                    speed_factor_p,
                    self._flags)
                speed_factor_p = min(speed_factor_p, new_speed_factor_p)

        for tl in traffic_lights:
            if (pylot.control.utils.is_traffic_light_active(
                    vehicle_transform, tl[0]) and
                pylot.control.utils.is_traffic_light_visible(
                    vehicle_transform, tl[0], self._flags)):
                tl_state = tl[1]
                new_speed_factor_tl = pylot.control.utils.stop_traffic_light(
                    vehicle_transform,
                    tl[0],
                    tl_state,
                    wp_vector,
                    wp_angle,
                    speed_factor_tl,
                    self._flags)
                speed_factor_tl = min(speed_factor_tl, new_speed_factor_tl)

        speed_factor = min(speed_factor_tl, speed_factor_p, speed_factor_v)
        state = {
            'stop_pedestrian': speed_factor_p,
            'stop_vehicle': speed_factor_v,
            'stop_traffic_lights': speed_factor_tl
        }
        self._logger.info('Aggent speed factors {}'.format(state))
        return speed_factor, state

    def get_control_message(self, wp_angle, wp_angle_speed, speed_factor,
                            current_speed, timestamp):
        current_speed = max(current_speed, 0)
        steer = self._flags.steer_gain * wp_angle
        if steer > 0:
            steer = min(steer, 1)
        else:
            steer = max(steer, -1)

        # Don't go to fast around corners
        if math.fabs(wp_angle_speed) < 0.1:
            target_speed_adjusted = self._flags.target_speed * speed_factor
        elif math.fabs(wp_angle_speed) < 0.5:
            target_speed_adjusted = 20 * speed_factor
        else:
            target_speed_adjusted = 15 * speed_factor

        self._pid.target = target_speed_adjusted
        pid_gain = self._pid(feedback=current_speed)
        throttle = min(
            max(self._flags.default_throttle - 1.3 * pid_gain, 0),
            self._flags.throttle_max)

        if pid_gain > 0.5:
            brake = min(0.35 * pid_gain * self._flags.brake_strength, 1)
        else:
            brake = 0

        return ControlMessage(steer, throttle, brake, False, False, timestamp)
