#!/bin/bash
# Job name:
#SBATCH --job-name=arin_gpu3_test
#
# Account:
#SBATCH --account=fc_basics
#
# Partition:
#SBATCH --partition=savio4_gpu
#
# Number of nodes:
#SBATCH --nodes=1
#
# Number of tasks (one for each GPU desired for use case) (example):
#SBATCH --ntasks=1
#
# Processors per task:
# Always at least twice the number of GPUs (savio2_gpu and GTX2080TI in savio3_gpu)
# Four times the number for TITAN and V100 in savio3_gpu
# Eight times the number for A40 in savio3_gpu
#SBATCH --cpus-per-task=2
#
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#
#SBATCH --output=saviogpu3test_%j.out
#SBATCH --error=saviogpu3test_%j.err 
# Wall clock limit:
#SBATCH --time=24:00:30
#
## Command(s) to run (example):
pwd
hostname 
date 
echo starting job...
echo celebA dataset and resnet10 with DP-SGD, width 32, weightdecay 1e-4, noise 1.0, max grad norm 1.0, delta 1e-5 with ERM
source ~/.bashrc
export PYTHONUNBUFFERED=1
cd /global/home/users/arinchang/overparam_spur_corr
 
python run_expt_dp.py -id dp_width32_erm -s confounder -d CelebA --noise 1.0 --max_per_sample_grad_norm 1.0 -t Blond_Hair -c Male --lr 0.01 --batch_size 128 --weight_decay 0.0001 --model resnet10vw --n_epochs 50 --train_from_scratch --resnet_width 32
# removed --gres=gpu:1
wait
date
 
echo "All done"