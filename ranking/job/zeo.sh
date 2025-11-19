#!/usr/bin/bash
#PBS -l select=1:ncpus=1:ngpus=1
#PBS -l walltime=276:00:00
#PBS -q gpu
#PBS -l place=vscatter:shared
#PBS -j oe
#PBS -N 0.1CLLnorm_weight_fix
source ~/.bashrc

##if you are running on local, delete the upper part of this code

export PYTHONPATH="$PYTHONPATH:.../ranking/chemprop-v1-old-branches"
cd .../ranking/chemprop-v1-old-branches
conda activate chemprop-gpu ##change to your env
START_TIME=$SECOND

SAVE_NAME='...'

python train.py \
    --aggregation mean \
    --data_path .../input_train.csv\
    --separate_test_path .../input_test.csv \
    --separate_val_path .../input_val.csv \
    --features_path .../feat_train.csv \
    --separate_val_features_path .../feat_val.csv \
    --separate_test_features_path .../feat_test.csv \
    --save_dir save_models/$SAVE_NAME \
    --ensemble_size 3 \
    --dataset_type ranking \
    --loss_function listnet \
    --metric packet_luce_ranking \
    --smiles_columns 'osda' \
    --target_columns 'normalized_yield' \
    --epochs 200 \
    --init_lr 1e-5 \
    --max_lr 1e-3 \
    --final_lr 1e-4 \
    --seed 42 \
    --depth 2 \
    --dropout 0.05 \
    --hidden_size 128 \
    --ffn_hidden_size 300 \
    --ffn_num_layers 2 \
    --activation LeakyReLU \
    --batch_size 128 \
    --listnet_eps 1e-6 \
    --lambda_contrastive 0.1 \
    --save_preds \

python predict.py \
    --checkpoint_dir save_models/$SAVE_NAME \
    --test_path  .../input_test.csv \
    --features_path  .../feat_test.csv \
    --preds_path save_models/$SAVE_NAME/preds_path/test/test.csv \

python predict.py \
    --checkpoint_dir save_models/$SAVE_NAME \
    --test_path  .../input_train.csv \
    --features_path  .../feat_train.csv \
    --preds_path save_models/$SAVE_NAME/preds_path/train/train.csv \

python predict.py \
    --checkpoint_dir save_models/$SAVE_NAME \
    --test_path  .../input_val.csv \
    --features_path  .../feat_val.csv \
    --preds_path save_models/$SAVE_NAME/preds_path/val/val.csv \

TRAINED_TIME=$(($SECONDS - $START_TIME))
echo '=========================================================='
echo "Training time: $(($TRAINED_TIME/60)) min $(($TRAINED_TIME%60)) sec"
echo '=========================================================='
conda deactivate