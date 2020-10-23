Simulation
==========

Using the CARLA simulator
-------------------------

Important flags
~~~~~~~~~~~~~~~
- ``--simulator_host``: Specifies the hostname where simulator is running.
- ``--simulator_port``: Specifies the port on which the simulator server is listening.
- ``--simulator_mode``:
- ``--simulator_fps``: Specifies the frames per second the simulator must run at.
- ``--simulator_town``: Specifies the simulator town to use.
- ``--simulator_weather``: Sets the weather in the simulator.
- ``--simulator_num_people``: Specifies the number of people agents to spawn. 
- ``--simulator_num_vehicles``: Specifies the number of vehicle agents to spawn. 
- ``--simulator_spawn_point_index``: Specifies the spawning location of the
  ego-vehicle.
- ``--simulator_camera_frequency``: Specifies the frequency at which the cameras
  are publishing frames.
- ``--simulator_gnss_frequency``: Specifies the frequency at which the GNSS sensor
  is publishing readings.
- ``--simulator_imu_frequency``: Specifies the frequency at which the IMU sensor
  is publishing readings.
- ``--simulator_lidar_frequency``: Specifies the frequency at which the LiDARs are
  publishing point clouds.
- ``--simulator_localization_frequency``: Specifies the frequency at which
  pose messages are sent when using perfect localization.
- ``--simulator_control_frequency``: Speicifes the frequenc at which control
  commands are applied.

Running scenarios
-----------------

- ``--scenario_runner``:

See the `reference <pylot.simulation.html>`_ for more information.
