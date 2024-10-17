#!/bin/bash
python_file="./mnist.py"
echo '
    Running '$python_file
nohup python -u $python_file --model stm --log  > log.out 2>&1 
