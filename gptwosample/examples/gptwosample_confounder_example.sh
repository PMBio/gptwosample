#!/bin/bash

# Make sure example files are present:
if [ \( ! -f ./ToyCondition1.csv \) -o \( ! -f ./ToyCondition2.csv \) ]
then
    echo generating toy example files
    python generateToyExampleFiles.py
fi

# Make sure gptwosample is reachable:
export PYTHONPATH=$PYTHONPATH:../../:

# run verbose (-v) gptwosample with 4 confounders (-c4) on toy example genes:
python ../../gptwosample.py -vc4 -o ./gptwosample_confounder_example/ ToyCondition1.csv ToyCondition2.csv