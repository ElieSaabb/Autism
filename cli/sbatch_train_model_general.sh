#!/bin/bash
#SBATCH --gpus-per-node=v100l:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --time=120:00:00

module purge
module load cuda cudnn
#module load StdEnv/2023
#module load intel/2023.2.1
#module load cuda/11.8

source /home/linah03/Projects/ESD/ESD_env/bin/activate
export PATH=/home/linah03/Projects/ESD/ESD_env/bin:$PATH

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

#----------------------------------------------------------------------#

train=${1:-"ECG"}  # String either ECG or EEG training setting
model_types=${2:-"bilstm"}  # List of models, should provide a singelton list with one model type.
# patient_ids=${3:-"aaaaajsl"}  # List of Patient ID, should provide a singelton list with one patient.
_patient_ids_=$(cut -d',' -f1 /home/linah03/Projects/ESD/helpers/filtered_patient_seizure_stats.csv | tail -n +2)
patient_ids=($_patient_ids_)
outputdir=${3:-"THUSZ"}
custome_batch=${4:-"False"}
general_model=${5:-"False"}
#----------------------------------------------------------------------#

echo Patient Specific Model on patient ${patient_ids}

echo Training using model ${label}
echo Training with custom batch ${custom_batch}

echo Logging at ${logdir}


# ON COMPUTE CANADA
basedir=/home/linah03/Projects/ESD
# ON PERSONAL COMPUTER
#basedir=/home/hussein/WorSpace/LBW/ESD/ESD_maindev

script=${basedir}/train_model_patient_specific.py

#----------------------------------------------------------------------#

echo Running ...
echo '>>>'
echo python ${script} --train ${train} --model_types ${model_types} --patient_ids ${patient_ids[@]} --outputdir ${outputdir} --custome_batch ${custome_batch} --general_model ${general_model}
echo '>>>'
echo ''


python ${script} --train ${train} --model_types ${model_types} --patient_ids ${patient_ids[@]} --outputdir ${outputdir} --custome_batch ${custome_batch} --general_model ${general_model}

#----------------------------------------------------------------------#
