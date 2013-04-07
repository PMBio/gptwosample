#!/bin/bash

# Make sure example files are present:
if [ \( ! -f ./ToyCondition1.csv \) -o \( ! -f ./ToyCondition2.csv \) ]
then
    echo generating toy example files
    python generateToyExampleFiles.py
fi

# Make sure gptwosample is reachable:
export PYTHONPATH=$PYTHONPATH:../../:

# run verbose (-v) gptwosample on toy example genes:
python ../../gptwosample.py -vo ./gptwosample_example/ ToyCondition1.csv ToyCondition2.csv