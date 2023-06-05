#!/bin/bash

# this is a way to check whether the right environment is activated
count=$(conda info | grep "active environment : cityflows-model" | wc -l)
if [ $count -eq 1 ]; then
    python -m src.utils.manage_data_files $@
else
    echo 'It seems like cityflows-model conda environment is not activated. Please run: conda activate cityflows-model'
fi
