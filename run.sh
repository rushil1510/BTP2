#!/bin/bash

# Clear the log file before running the script
> log.txt

# Record the start time
start_time=$(date +%s)

# Execute the Python script and log output
python3 optimize_burner.py &> log.txt

# Record the end time
end_time=$(date +%s)

# Calculate execution time
execution_time=$((end_time - start_time))

# Log execution time
echo "Execution time of optimize_burner.py: ${execution_time} seconds"

# Check if the __pycache__ directory exists
if [ -d "__pycache__" ]; then
    # Remove the __pycache__ directory and log the action
    rm -rf __pycache__
    echo "__pycache__ directory deleted." >> log.txt
else
    echo "__pycache__ directory does not exist." >> log.txt
fi
