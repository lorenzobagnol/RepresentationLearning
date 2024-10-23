#!/bin/bash

# Add explicit logging of the starting script
echo "Starting mnist.py execution at $(date)" >> log.out

# Execute the Python script directly and ensure logs are redirected to both a log file and stdout/stderr
python_file="mnist.py"
python -u $python_file --model stm --training LifeLong --log 2>&1 | tee -a log.out