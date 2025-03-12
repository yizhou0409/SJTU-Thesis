#!/bin/bash
#SBATCH --job-name=eval_qwen_math  # Job name
#SBATCH --output=eval_output.txt   # Output file
#SBATCH --error=eval_error.txt     # Error file
#SBATCH --time=48:00:00            # Time limit (48 hours)
#SBATCH --partition=rtx8000        # Partition/queue name (change if needed)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=16         # Number of CPU cores
#SBATCH --mem=200G                 # Memory allocation
#SBATCH --gres=gpu:rtx8000:1               # Request 1 GPU

# Load necessary modules (if required)
module load anaconda               # Load Anaconda module (modify if needed)

# Activate Anaconda environment
source activate Qwen_Math

# Run the Python script with argument
python eval.py --data_names=gsm8k --source_model_name=Qwen/Qwen2.5-Math-1.5B-Instruct
