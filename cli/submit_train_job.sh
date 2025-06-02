#!/usr/bin/env bash
# ON COMPUTE CANADA

# Where your code actually lives:
RootPath=/home/linah03/Projects/Autism
scriptdir=${RootPath}/cli
sbatch_script=${scriptdir}/sbatch_train_job.sh

# Create your log dirs
daymonth=$(date +%d%m)
RootName=results_v${daymonth}
dataset=ABIDE
scheme=General
base_output=/scratch/linah03/Results_Autism/${RootName}/${dataset}/${scheme}
mkdir -p ${base_output}/EOlogs/{out,error}

# Go there and submit
cd "${scriptdir}" || exit 1
echo "Submitting training job..."
sbatch "${sbatch_script}"
