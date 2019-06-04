import math
import time

from erdos.data_stream import DataStream
from erdos.op import Op
from erdos.timestamp import Timestamp
from erdos.utils import frequency, setup_csv_logging, setup_logging, time_epoch_ms

from control.messages import ControlMessage
import control.utils as agent_utils
import control.ground_utils
from pid_controller.pid import PID
from simulation.planner.map import CarlaMap
import pylot_utils


class GroundAgentOperator(Op):
    def __init__(self,
                 name,
                 city_name,
                 flags,
                 log_file_name=None,
                 csv_file_name=None):
        super(GroundAgentOperator, self).__init__(name)
        self._logger = setup_logging(self.name, log_file_name)
        self._csv_logger = setup_csv_logging(self.name + '-csv', csv_file_name)
        self._map = CarlaMap(city_name)
        self._flags = flags
        self._pid = PID(p=flags.pid_p, i=flags.pid_i, d=flags.pid_d)
        self._vehicle_transform = None
        self._vehicle_speed = None
        self._pedestrians = None
        self._vehicles = None
        self._traffic_lights = None
        self._traffic_signs = None
        self._wp_angle = None
        self._wp_vector = None
        self._wp_angle_speed = None

    @staticmethod
    def setup_streams(input_streams):
        input_streams.filter(pylot_utils.is_can_bus_stream).add_callback(
            GroundAgentOperator.on_can_bus_update)
        input_streams.filter(pylot_utils.is_ground_pedestrians_stream).add_callback(
            GroundAgentOperator.on_pedestrians_update)
        input_streams.filter(pylot_utils.is_ground_vehicles_stream).add_callback(
            GroundAgentOperator.on_vehicles_update)
        input_streams.filter(pylot_utils.is_ground_traffic_lights_stream).add_callback(
            GroundAgentOperator.on_traffic_lights_update)
        input_streams.filter(pylot_utils.is_ground_traffic_signs_stream).add_callback(
            GroundAgentOperator.on_traffic_signs_update)
        input_streams.filter(pylot_utils.is_waypoints_stream).add_callback(
            GroundAgentOperator.on_waypoints_update)
        # Set no watermark on the output stream so that we do not
        # close the watermark loop with the carla operator.
        return [pylot_utils.create_control_stream()]

    def on_waypoints_update(self, msg):
        self._wp_angle = msg.wp_angle
        self._wp_vector = msg.wp_vector
        self._wp_angle_speed = msg.wp_angle_speed

    def on_can_bus_update(self, msg):
        self._logger.info("Received vehicle pos %s", msg)
        self._vehicle_transform = msg.data.transform
        self._vehicle_speed = msg.data.forward_speed

    def on_pedestrians_update(self, msg):
        self._pedestrians = msg.pedestrians

    def on_vehicles_update(self, msg):
        self._vehicles = msg.vehicles

    def on_traffic_lights_update(self, msg):
        self._traffic_lights = msg.traffic_lights

    def on_traffic_signs_update(self, msg):
        self._traffic_signs = msg.speed_signs

    # TODO(ionel): Set the frequency programmatically.
    @frequency(10)
    def run_step(self):
        if (self._vehicle_transform is None or self._vehicle_speed is None
                or self._pedestrians is None or self._vehicles is None
                or self._traffic_lights is None or self._wp_angle is None):
            return
        speed_factor, state = self.stop_for_agents(
            self._wp_angle, self._wp_vector, self._vehicles, self._pedestrians,
            self._traffic_lights)
        control_msg = self.get_control_message(
            self._wp_angle, self._wp_angle_speed, speed_factor,
            self._vehicle_speed, Timestamp(coordinates=[0]))
        self.get_output_stream('control_stream').send(control_msg)

    def stop_for_agents(self, wp_angle, wp_vector, vehicles, pedestrians,
                        traffic_lights):
        speed_factor = 1
        speed_factor_tl = 1
        speed_factor_p = 1
        speed_factor_v = 1

        if self._flags.stop_for_vehicles:
            for vehicle in vehicles:
                if control.ground_utils.is_vehicle_on_same_lane(
                        self._vehicle_transform, vehicle.location, self._map):
                    new_speed_factor_v = agent_utils.stop_vehicle(
                        self._vehicle_transform,
                        vehicle.location,
                        wp_vector,
                        speed_factor_v,
                        self._flags)
                    speed_factor_v = min(speed_factor_v, new_speed_factor_v)

        if self._flags.stop_for_pedestrians:
            for pedestrian in pedestrians:
                if control.ground_utils.is_pedestrian_hitable(
                        pedestrian.location, self._map):
                    new_speed_factor_p = agent_utils.stop_pedestrian(
                        self._vehicle_transform,
                        pedestrian.location,
                        wp_vector,
                        speed_factor_p,
                        self._flags)
                    speed_factor_p = min(speed_factor_p, new_speed_factor_p)

        if self._flags.stop_for_traffic_lights:
            for tl in traffic_lights:
                if (control.ground_utils.is_traffic_light_active(
                        self._vehicle_transform, tl.location, self._map) and
                    agent_utils.is_traffic_light_visible(
                        self._vehicle_transform, tl.location, self._flags)):
                    new_speed_factor_tl = agent_utils.stop_traffic_light(
                        self._vehicle_transform,
                        tl.location,
                        tl.state,
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

        return speed_factor, state

    def execute(self):
        self.run_step()
        self.spin()

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
