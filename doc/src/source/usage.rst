.. _usage:

Parameter options
----------------------

Calling signature::

 gptwosample [-h] [-o DIR] [-t] [-c N] [-p] [-v] [--version] [--backend [PDF,...]]  FILE FILE

where::

  FILE                  treatment/control files to compare against each other
  -h, --help            show this help message and exit
  -o DIR, --out DIR     set output dir [default: ./twosample_out/]
  -t, --timeshift       account for timeshifts in data [default: False]
  -c N, --confounder N  account for N confounders in data [default: 0]
  -p, --plot            plot data into outdir/plots? [default: False]
  -v, --verbose         set verbosity level [default: 0]
  --version             show program's version number and exit
  --backend [PDF,...]   matplotlib backend - see matplotlib.use(backend)


.. _dataformat:

Data format
------------------------------------
The format of the two ``.csv`` files (``FILE FILE`` in usage) is as follows:

    ============ =============== ==== ===============
    *arbitrary*  x1              ...  xl
    ============ =============== ==== ===============
    Gene ID 1    y1 replicate 1  ...  yl replicate 1
    ...          ...             ...  ...
    Gene ID 1    y1 replicate k1 ...  yl replicate k1

    ...

    Gene ID n    y1 replicate 1  ...  yl replicate 1
    ...          ...             ...  ...
    Gene ID n    y1 replicate kn ...  yl replicate kn
    ============ =============== ==== ===============

See ``gptwosample/examples/ToyCondition{1,2].csv`` for example data files.
All values, which cannot be translated by :py:func:`float` will be 
treated as missing values in the model.
