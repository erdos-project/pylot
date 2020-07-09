Localization
============

The package provides two types of localizations:

1. **Extended Kalman filter**: Fuses GNSS and IMU data for decimeter level
   accuracy in simulation.
2. **NDT Matching**: Pylot does not implement NDT matching, but instead provides
   and operator that bridges between the Autoware's NDT matching implentation
   and the pipeline. This operator can be used on real-world vehicles on which
   Autoware's NDT matching is deployed.
   

Important flags
---------------

- ``--localization``: Enables the localization component of the pipeline. If
  this flag is ``False``, then the pipeline uses the perfect localization it
  receives from the simulator.

More information
----------------
See the `reference <pylot.localization.html>`_ for more information.
