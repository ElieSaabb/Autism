#!/bin/bash
#SBATCH --job-name=autism_train           # descriptive name
#SBATCH --account=def-khenni            # your Compute Canada allocation
#SBATCH --gres=gpu:1                      # 1 GPU
#SBATCH --nodes=1                         # 1 node
#SBATCH --cpus-per-task=4                 # 4 CPU cores
#SBATCH --mem=65536M                      # 64 GB RAM (in multiples of 1024M)
#SBATCH --time=12:00:00                   # walltime hh:mm:ss
#SBATCH --output=/scratch/linah03/Autism/logs/EOlogs/out/train_%j.out
#SBATCH --error=/scratch/linah03/Autism/logs/EOlogs/err/train_%j.err
#SBATCH --mail-user=elie24saab@gmail.com   # your email address
#SBATCH --mail-type=END,FAIL               # send on job end & failure

echo loading required server modules
echo module load  cuda cudnn

module load  cuda cudnn

echo activating env
source /lustre04/scratch/linah03/WorkSpace_ELI/Autism/venvAutism/bin/activate
export PATH=/lustre04/scratch/linah03/WorkSpace_ELI/Autism/venvAutism/bin:$PATH

echo checking python
which python

echo running python job
cd /lustre04/scratch/linah03/WorkSpace_ELI/Autism

export PYTHONPATH=$PWD:$PYTHONPATH

srun python cli/train.py

