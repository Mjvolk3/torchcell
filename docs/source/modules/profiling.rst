torchcell.profiling
===================

.. module:: torchcell.profiling

.. currentmodule:: torchcell.profiling

A timing decorator for profiling. :func:`~torchcell.profiling.time_method` records the wall time of each call of the decorated function in a module-level store when the environment variable ``TORCHCELL_DEBUG_TIMING`` is ``1`` (otherwise it calls the function directly), and :func:`~torchcell.profiling.get_timings`, :func:`~torchcell.profiling.get_timing_summary` and :func:`~torchcell.profiling.print_timing_summary` read it back.

.. contents:: Contents
    :local:

Functions
---------

.. autosummary::
   :nosignatures:
   :toctree: ../generated

   time_method
   print_timing_summary
   reset_timings
   get_timings
   get_timing_summary
