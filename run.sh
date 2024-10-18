#!/bin/bash
sleep 7200
python_file="./mnist.py"
nohup python -u $python_file --model stm --training LifeLong --log  > log.out 2>&1 &
