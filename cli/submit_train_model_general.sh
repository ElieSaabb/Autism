#!/bin/bash
. /home/linah03/Projects/ESD/cli/functions.sh


#----------------------------------------------------------------------#
# ON COMPUTE CANADA
RootPath=/home/linah03/Projects/ESD
# ON PERSONAL COMPUTER
#RootPath=/home/hussein/WorSpace/LBW/ESD/ESD_maindev

scriptdir=/home/linah03/project
scriptname=sbatch_train_model_general.sh

daymonth=$(date +%d%m)
RootName=results_v${daymonth}
dataset=THUSZ
scheme=general

outputdir_=/scratch/linah03/Results_ESD/${RootName}/${dataset}/${scheme}
LogRootPath=/scratch/linah03/Results_ESD/${RootName}/${dataset}/${scheme}

LogsErrorsBasePath=${LogRootPath}/EOlogs
_logdir=${LogsErrorsBasePath}/out
_errordir=${LogsErrorsBasePath}/error
mkdir -p ${_logdir}
mkdir -p ${_errordir}


#----------------------------------------------------------------------#
_general_model_='True'
_custome_batch_="True False"
_train_='ECG EEG'
_model_types_='bilstm wavenet_bilstm' # wavenet
# _patient_ids_=`cut -d',' -f1 /home/linah03/Projects/ESD/helpers/filtered_patient_seizure_stats.csv | tail -n +2`


#----------------------------------------------------------------------#

cd /home/linah03/project


for custome_batch in ${_custome_batch_}; do
 outputdir=${outputdir_}/custombatch_${custome_batch}
 logdir=${_logdir}/custombatch_${custome_batch}
 errordir=${_errordir}/custombatch_${custome_batch}
 mkdir -p ${logdir}
 mkdir -p ${errordir}
 for train in $_train_; do
   for model_types in $_model_types_; do
 
        label=tmps_general_${_general_model_}_${train}_${model_types}_${custome_batch}
        outfile=$(find_next_run ${logdir} ${label})
        f="$(basename -- $outfile)"
        mkdir -p ${errordir}/${label}
        errorfile=${errordir}/${label}/${f}

        echo sbatch --job-name=${label} --output=${outfile}.out --error=${errorfile}.out ${scriptdir}/${scriptname} ${train} ${model_types} ${outputdir} ${custome_batch} ${_general_model_}

       RES=$(sbatch --job-name=${label} --output=${outfile}.out --error=${errorfile}.out ${scriptdir}/${scriptname} ${train} ${model_types} ${outputdir} ${custome_batch} ${_general_model_})
        echo ${RES}
        [ -e ${outfile}.id ] && rm ${outfile}.id
        echo ${RES##* } > ${outfile}.id
    done
 done
done
#----------------------------------------------------------------------#
