#!/bin/bash

config_path=$1
train_output_path=$2
ckpt_name=$3

stage=0
stop_stage=0

# hifigan
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    echo "in hifigan syn_e2e"
    FLAGS_allocator_strategy=naive_best_fit \
    FLAGS_fraction_of_gpu_memory_to_use=0.01 \
    python3 ${BIN_DIR}/../synthesize_e2e.py \
        --am=fastspeech2_opencpop \
        --am_config=${config_path} \
        --am_ckpt=${train_output_path}/checkpoints/${ckpt_name} \
        --am_stat=dump/train/speech_stats.npy \
        --voc=hifigan_opencpop \
        --voc_config=hifigan_opencpop_ckpt/default.yaml \
        --voc_ckpt=hifigan_opencpop_ckpt/snapshot_iter_2500000.pdz \
        --voc_stat=hifigan_opencpop_ckpt/feats_stats.npy \
        --lang=sing-zh \
        --text=${BIN_DIR}/../sentences_sing.txt \
        --output_dir=${train_output_path}/test_e2e \
        --phones_dict=dump/phone_id_map.txt \
        --inference_dir=${train_output_path}/inference
fi
