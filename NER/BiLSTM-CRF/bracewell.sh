#! /bin/bash

#SBATCH --time=120:00:00
#SBATCH --mem=16G
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1


module load cuda/9.2.88
module load python/3.7.2 cudnn/v7.1.4-cuda92
module load pytorch/1.0.0-py37-cuda92


for i in {1..5}
do
    python3 train.py --dataset=conll2003 --output_dir=conll2003 \
    --elmo_json=/data/dai031/Corpora/ELMo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json \
    --elmo_hdf5=/data/dai031/Corpora/ELMo/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5
done