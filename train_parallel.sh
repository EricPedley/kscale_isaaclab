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

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install tmux first."
    exit 1
fi

# Session name for tmux
SESSION_NAME="quadcopter_training"

echo "Starting 4 parallel training processes in tmux windows for configs 0-999"

# Kill existing session if it exists
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create new tmux session with first window
tmux new-session -d -s $SESSION_NAME

# Get the current working directory
CURRENT_DIR=$(pwd)

# Rename the first window and start Process 1
tmux rename-window -t $SESSION_NAME:0 "Process-1"
echo "Starting Process 1 (configs $process1_start-$process1_end) in window 0"
tmux send-keys -t $SESSION_NAME:0 "cd $CURRENT_DIR" Enter
tmux send-keys -t $SESSION_NAME:0 "$(declare -f run_training_range); run_training_range $process1_start $process1_end 1" Enter

# Create second window for Process 2
tmux new-window -t $SESSION_NAME -n "Process-2"
echo "Starting Process 2 (configs $process2_start-$process2_end) in window 1"
tmux send-keys -t $SESSION_NAME:1 "cd $CURRENT_DIR" Enter
tmux send-keys -t $SESSION_NAME:1 "$(declare -f run_training_range); sleep 10 && run_training_range $process2_start $process2_end 2" Enter

# Create third window for Process 3
tmux new-window -t $SESSION_NAME -n "Process-3"
echo "Starting Process 3 (configs $process3_start-$process3_end) in window 2"
tmux send-keys -t $SESSION_NAME:2 "cd $CURRENT_DIR" Enter
tmux send-keys -t $SESSION_NAME:2 "$(declare -f run_training_range); sleep 20 && run_training_range $process3_start $process3_end 3" Enter

# Create fourth window for Process 4
tmux new-window -t $SESSION_NAME -n "Process-4"
echo "Starting Process 4 (configs $process4_start-$process4_end) in window 3"
tmux send-keys -t $SESSION_NAME:3 "cd $CURRENT_DIR" Enter
tmux send-keys -t $SESSION_NAME:3 "$(declare -f run_training_range); sleep 30 && run_training_range $process4_start $process4_end 4" Enter

# Go back to first window
tmux select-window -t $SESSION_NAME:0

echo ""
echo "All 4 training processes have been started in tmux session '$SESSION_NAME'"
echo "Each process runs in its own window:"
echo "  Window 0: Process-1 (configs $process1_start-$process1_end)"
echo "  Window 1: Process-2 (configs $process2_start-$process2_end)"
echo "  Window 2: Process-3 (configs $process3_start-$process3_end)"
echo "  Window 3: Process-4 (configs $process4_start-$process4_end)"
echo ""
echo "To attach to the session and monitor progress:"
echo "  tmux attach-session -t $SESSION_NAME"
echo ""
echo "To switch between windows while attached:"
echo "  Ctrl+b then 0, 1, 2, or 3 (for specific window)"
echo "  Ctrl+b then n (next window)"
echo "  Ctrl+b then p (previous window)"
echo "  Ctrl+b then w (window list)"
echo ""
echo "To detach from session (keep processes running):"
echo "  Ctrl+b then d"
echo ""
echo "To kill the session and stop all processes:"
echo "  tmux kill-session -t $SESSION_NAME"