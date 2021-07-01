#!/bin/bash
#SBATCH -q regular
#SBATCH -A m342
#SBATCH -t 3:59:00
#SBATCH -n 16
#SBATCH -o cori_output/train/datasets/test_dataset/chunks_W4096_S4096/resnet50/C/n2_g8_A1_b64_r0.001_phylum_Oadam/train.%j.log
#SBATCH -e cori_output/train/datasets/test_dataset/chunks_W4096_S4096/resnet50/C/n2_g8_A1_b64_r0.001_phylum_Oadam/train.%j.log
#SBATCH -J test_taxon
#SBATCH -C gpu
#SBATCH -c 10
#SBATCH --ntasks-per-node 8
#SBATCH --gpus-per-task 1


conda activate zenv

JOB="$SLURM_JOB_ID"
OMP_NUM_THREADS=1
NCCL_DEBUG="INFO"
OPTIONS=" -g 8 -n 2 -e 1 -k 5 -y -f -E n2_g8_A1_b64_r0.001_phylum_Oadam"
OUTDIR="cori_output/train/datasets/test_dataset/chunks_W4096_S4096/resnet50/C/n2_g8_A1_b64_r0.001_phylum_Oadam/train.$JOB"
CONF="cori_output/train/datasets/test_dataset/chunks_W4096_S4096/resnet50/C/n2_g8_A1_b64_r0.001_phylum_Oadam/train.$JOB.yml"
INPUT="/global/cscratch1/sd/azaidi/taxon_trials/ar122_r202.rep.h5"
LOG="$OUTDIR.log"
CMD="deep-taxon train --slurm $OPTIONS $CONF $INPUT $OUTDIR"

mkdir -p $OUTDIR
cp $0 $OUTDIR.sh
srun $CMD > $LOG 2>&1
