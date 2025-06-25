#!/bin/bash
#SBATCH --job-name=autism_eval           # a name for your job
#SBATCH --account=def-khenni           # your Compute Canada allocation
#SBATCH --gres=gpu:1                     # no GPU needed for evaluation
#SBATCH --nodes=1                        # 1 node
#SBATCH --cpus-per-task=4                # adjust as needed
#SBATCH --mem=8192M                      # 8 GB RAM
#SBATCH --time=01:00:00                  # walltime hh:mm:ss
#SBATCH --output=/lustre04/scratch/linah03/WorkSpace_ELI/Autism/logs/EOlogs/out/eval_%j.out
#SBATCH --error=/lustre04/scratch/linah03/WorkSpace_ELI/Autism/logs/EOlogs/err/eval_%j.err
#SBATCH --mail-user=elie24saab@gmail.com  # your email
#SBATCH --mail-type=END,FAIL              # notify on job end/fail

echo loading required server modules
echo module load  cuda cudnn

module load  cuda cudnn

echo activating env
source /lustre04/scratch/linah03/WorkSpace_ELI/Autism/venvAutism/bin/activate

echo Setting PYTHONPATH
export PATH=/lustre04/scratch/linah03/WorkSpace_ELI/Autism/venvAutism/bin:$PATH

echo checking python
which python

echo running python job
cd /lustre04/scratch/linah03/WorkSpace_ELI/Autism

echo "Starting evaluation…"  
srun python cli/eval.py

echo "Done."  
