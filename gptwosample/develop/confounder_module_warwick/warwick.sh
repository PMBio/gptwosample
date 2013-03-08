#!/bin/bash
python warwick.py $1 data conf ../../examples/warwick_control.csv ../../examples/warwick_treatment.csv plot_confounder $* &
python warwick.py $1 data conf sam $* &
python warwick.py $1 data conf rep $* &
python warwick.py $1 data noconf $* &
python warwick.py $1 data ideal $* &
python warwick.py $1 data raw $* &
wait
python plot_rocs.py $1