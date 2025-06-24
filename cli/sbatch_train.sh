#!/bin/bash
#
#  ============= Job Name & Account =============
#SBATCH --job-name=autism_train           # descriptive name
#SBATCH --account=def-khenni            # your Compute Canada allocation

#  ============= Resources Requested ===========
#SBATCH --gres=gpu:1                      # 1 GPU
#SBATCH --nodes=1                         # 1 node
#SBATCH --cpus-per-task=4                 # 4 CPU cores
#SBATCH --mem=32768M                      # 32 GB RAM (in multiples of 1024M)
#SBATCH --time=12:00:00                   # walltime hh:mm:ss

#  ============= Output & Notifications ========
#SBATCH --output=/scratch/linah03/Autism/logs/EOlogs/out/train_%j.out
#SBATCH --error=/scratch/linah03/Autism/logs/EOlogs/err/train_%j.err
#SBATCH --mail-user=elie24saab@gmail.com   # your email address
#SBATCH --mail-type=END,FAIL               # send on job end & failure

#  ============= Environment Setup ============
module load python/3.10 cuda cudnn         # adjust modules if needed
source ~/venvAutism/bin/activate           # activate your virtualenv

#  ============= Run Training ================
cd /scratch/linah03/Autism                 # project root containing cli/, data_pipelines/, etc.
export PYTHONPATH=$PWD                     # ensure imports work

srun python cli/train.py
