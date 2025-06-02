#!/bin/bash
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00

module purge
module load cuda cudnn
#module load StdEnv/2023
#module load intel/2023.2.1
#module load cuda/11.8

#SBATCH --mail-user=elie24saab@gmail.com # Optional: get email updates
#SBATCH --mail-type=ALL                    # Optional: notifications on start, end, fail

# Activate your virtual environment
source /home/linah03/Projects/Autism/venvAutism/bin/activate
export PATH=/home/linah03/Projects/Autism/venvAutism/bin:$PATH

echo
python -c "import tensorflow as tf; print(tf.sysconfig.get_build_info())"
echo

echo 
nvidia-smi
echo

echo 
which python
echo 

#cd /home/linah03/project

# Load the Python module
module load python/3.10


# Run your training script
python cli/train_model.py
