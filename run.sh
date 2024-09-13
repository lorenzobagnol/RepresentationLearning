#!/bin/bash
python_file="./colors.py"
echo '
    Running '$python_file
python $python_file --model 'som' --log
# sleep 10000000