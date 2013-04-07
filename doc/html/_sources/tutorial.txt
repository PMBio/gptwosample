Step by step tutorial & examples
----------------------------------------

Once the data has been prepared, ``gptwosample`` can be executed from
the unix command line. See the full usage information in :ref:`usage`. 

See format for input data ``.csv`` files in :ref:`dataformat`. 

Make sure you either install gptwosample (:ref:`install`) 
.. or ``cd``
   into the extracted gptwosample folder before running this tutorial. 

Try printing the full help of the script using::

 python gptwosample --help

If an error occurs, you probably ``cd`` one level too deep and you can
``cd ..`` up one level. 

In this tutorial we will build up a full usage call of ``gptwosample``.
First, we want to run gptwosample verbosly, thus the call so far looks like::

 gptwosample -v

To enable plotting we provide the switch ``-p`` to the script::

 gptwosample -v -p

We want to correct for timeshifts (more on :ref:`timeshift`), thus we
enable the timeshift switch ``-t``::

 gptwosample -v -p -t

Next we could additionally learn x confounding factors (see
:ref:`confounders` for details on confounding factors) and account
for them while two-sampling::

 gptwosample -v -p -t -c x

but we do not want to account for confounders in this tutorial.

The output of the script shall be in the subfolder ``./tutorial/``, so
we add the output flag ``-o ./tutorial/``:

 gptwosample -v -p -t -o ./tutorial/

The script shall be run on the two toy condition files ``ToyCondition{1,2}.csv``
given in ``examples/ToyCondition{1,2}.csv``. These files
are non optional as this package is only for comparing two timeseries
experiments to each other::

 gptwosample -v -p -t -o ./tutorial/ examples/ToyCondition1.csv examples/ToyCondition2.csv

Note that the optional parameters could be collected together to give
rise to a more compact call signature::

 gptwosample -vpto tutorial examples/ToyCondition1.csv
 examples/ToyCondition2.csv

After hitting return the script runs gptwosample on every gene given
in the ToyCondition files and plots each gene into
``tutorial/plots/``. One example plot will look like:

.. image:: ../images/timeshiftexample.pdf
        :height: 12cm

The results are saved in the ``results.csv``, which contains all
predicted Bayes Factors and learnt covariance function parameters for
all genes (:ref:`results`).

For more tutorials and example files on how to use this package see
``gptwosample/examples``.
