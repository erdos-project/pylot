Simulation
==========

Using the CARLA simulator
-------------------------

Important flags
~~~~~~~~~~~~~~~
- ``--carla_host``: Specifies the hostname where CARLA is running.
- ``--carla_port``: Specifies the port on which the CARLA server is listening.
- ``--carla_mode``:
- ``--carla_fps``: Specifies the frames per second the simulator must run at.
- ``--carla_town``: Specifies the CARLA town to use.
- ``--carla_weather``: Sets the CARLA weather.
- ``--carla_num_people``: Specifies the number of people agents to spawn. 
- ``--carla_num_vehicles``: Specifies the number of vehicle agents to spawn. 
- ``--carla_spawn_point_index``: Specifies the spawning location of the
  ego-vehicle.
- ``--carla_camera_frequency``: Specifies the frequency at which the cameras
  are publishing frames.
- ``--carla_gnss_frequency``: Specifies the frequency at which the GNSS sensor
  is publishing readings.
- ``--carla_imu_frequency``: Specifies the frequency at which the IMU sensor
  is publishing readings.
- ``--carla_lidar_frequency``: Specifies the frequency at which the LiDARs are
  publishing point clouds.
- ``--carla_localization_frequency``: Specifies the frequency at which
  pose messages are sent when using perfect localization.
- ``--carla_control_frequency``: Speicifes the frequenc at which control
  commands are applied.

Running scenarios
-----------------

- ``--carla_scenario_runner``:

See the `reference <pylot.simulation.html>`_ for more information.
