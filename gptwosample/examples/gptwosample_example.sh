#!/bin/bash

# Make sure example files are present:
if [ \( ! -f ./ToyCondition1.csv \) -o \( ! -f ./ToyCondition2.csv \) ]
then
    echo generating toy example files
    python generateToyExampleFiles.py
fi

# Make sure gptwosample is reachable:
export PYTHONPATH=$PYTHONPATH:../../:

# run gptwosample:
# -v   : verbose 
# -p   : and plot
# -o D : write results in directory D
# on ToyConditions{1,2}.csv
python ../../gptwosample.py -v -p -o ./gptwosample_example/ ToyCondition1.csv ToyCondition2.csv $*