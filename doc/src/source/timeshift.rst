.. _timeshift:

Timeshift detection between replicates
===============================================
A novel covariance function detecting timehifts between
time series accounts for temporal mismatches between time series, (of
replicates and samples) which share similar patterns, shifted in
time. This allows for additional correction of confounding variation
in time, as treatment might slow down reaction time of cell-cycle
genes, leading to a bunch of falsely positive predicted non differential
expressed genes downstream. 

To enable timeshift detection add the flag ``-t`` to the run
script. Timeshifts for all replicates will be reported in
``results.csv``, where the order of replicates is the same order as in
the input files ``FILE FILE`` (see :ref:`usage`).
