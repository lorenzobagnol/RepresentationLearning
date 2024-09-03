#!/bin/bash
python_file="./colors.py"
echo '
    Running '$python_file
python $python_file --model 'som' --log 'True'
# sleep 10000000