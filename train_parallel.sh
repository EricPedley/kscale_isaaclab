#!/bin/bash

# Function to run training for a range of configs
run_training_range() {
    local start=$1
    local end=$2
    local process_id=$3
    
    echo "Process $process_id: Starting training for configs $start to $end"
    
    for ((i=$start; i<=$end; i++)); do
        echo "Process $process_id: Training config $i"
        ./isaaclab.sh -p scripts/reinforcement_learning/skrl/train.py \
            --task=Isaac-Quadcopter-Direct-v0 \
            env.config_path=source/isaaclab_tasks/isaaclab_tasks/direct/quadcopter/parameters/config_${i}.json \
            --headless
        
        # Check if the command was successful
        if [ $? -ne 0 ]; then
            echo "Process $process_id: Error training config $i, continuing..."
        else
            echo "Process $process_id: Successfully completed config $i"
        fi
    done
    
    echo "Process $process_id: Finished training configs $start to $end"
}

# Calculate ranges for 4 processes (0-999 = 1000 configs)
# Each process handles 250 configs
process1_start=0
process1_end=249

process2_start=250
process2_end=499

process3_start=500
process3_end=749

process4_start=750
process4_end=999

echo "Starting 4 parallel training processes for configs 0-999"

# Start 4 background processes
run_training_range $process1_start $process1_end 1 &
pid1=$!

sleep 10

run_training_range $process2_start $process2_end 2 &
pid2=$!

sleep 10

run_training_range $process3_start $process3_end 3 &
pid3=$!

sleep 10

run_training_range $process4_start $process4_end 4 &
pid4=$!

sleep 10

# Wait for all processes to complete
echo "Waiting for all processes to complete..."
echo "Process IDs: $pid1, $pid2, $pid3, $pid4"

wait $pid1
echo "Process 1 completed"

wait $pid2
echo "Process 2 completed"

wait $pid3
echo "Process 3 completed"

wait $pid4
echo "Process 4 completed"

echo "All training processes completed!"