#!/usr/bin/env bash

#SBATCH --job-name=image-gen-pytorch
#SBATCH --partition=normal
#SBATCH --gres=gpu:2g.10gb:1
#SBATCH --time=4:00:00
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=uctxx@student.kit.edu
#SBATCH --output=run-main-long.out
#SBATCH --error=run-main-long.err

module purge
module load devel/cuda/11.8

pixi install --locked
pixi run main
