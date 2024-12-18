#!/bin/zsh

#SBATCH -J clam_survival       # Job name
#SBATCH -o ./slurm/%x.o%j     # Output log file (%j expands to job ID)
#SBATCH -e ./slurm/%x.e%j       # Error log file
#SBATCH --gres=gpu:1                   # Number of GPUs (1 GPU)
#SBATCH --nodelist=gnode1
#SBATCH --nodes=1                      # Number of nodes
#SBATCH --mem=300G                      # Memory per node
#SBATCH --time 14-0:00                # Time limit hrs:min:sec

eval "$(conda shell.bash hook)"
conda activate clam_latest

ulimit -n 30000

export NCCL_P2P_LEVEL=NVL

# Navigate to your project directory
cd /home/brentoh1030/workspace/CLAM

# Run your Python script
python main.py --drop_out 0.5 --early_stopping --lr 2e-3 --reg 5e-4 --k 10 --exp_code task_3_survival_prediction_CLAM_50 --subtyping --task task_3_survival_prediction --model_type clam_sb --log_data --task_type regression --data_root_dir /home/brentoh1030/workspace/gbmdata/test/TCGA-GBM --results_dir /home/brentoh1030/workspace/CLAM/results --split_dir /home/brentoh1030/workspace/CLAM/splits/task_3_survival_prediction_100 --embed_dim 1324
