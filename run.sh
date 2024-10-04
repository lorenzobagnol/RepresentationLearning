#!/bin/bash
python_file="./mnist.py"
echo '
    Running '$python_file
python $python_file --model stm --log
