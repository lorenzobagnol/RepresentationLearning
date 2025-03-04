#!/bin/bash	

sleep 1000000

# Add explicit logging of the starting script
echo "Starting mnist.py execution at $(date)" >> log.out


# Execute the Python script directly and ensure logs are redirected to both a log file and stdout/stderr
python_file="/workspace/mnist.py"
nohup python -u $python_file --model stm --training LifeLong --log  > log.out 2>&1 