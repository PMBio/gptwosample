#!/bin/bash
echo running scripts...
mkdir -p $1
python warwick.py $1 data conf ../../examples/warwick_control.csv ../../examples/warwick_treatment.csv plot_confounder $* > $1/conf_all.txt &
python warwick.py $1 data conf sam plot_confounder $* > $1/conf_sam.txt &
python warwick.py $1 data conf rep plot_confounder $* > $1/conf_rep.txt &
python warwick.py $1 data noconf $* > $1/noconf.txt &
python warwick.py $1 data ideal $* > $1/ideal.txt &
python warwick.py $1 data raw $* > $1/raw.txt &
wait
echo finished, plotting roc curves
python plot_rocs.py $1