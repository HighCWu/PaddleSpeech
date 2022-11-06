#!/bin/bash

stage=0
stop_stage=100

config_path=$1

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    # get durations from transcriptions files
    echo "Generate durations.txt from transcriptions files ..."
    python3 ${MAIN_ROOT}/utils/gen_duration_for_svs.py \
        --inputdir=~/datasets/ \
        --output=durations.txt \
        --config=${config_path}
fi

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    # extract features
    echo "Extract features ..."
    python3 ${BIN_DIR}/../preprocess.py \
        --rootdir=~/datasets/opencpop/ \
        --dataset=opencpop \
        --dumpdir=dump \
        --dur-file=durations.txt \
        --config=${config_path} \
        --cut-sil=True \
        --num-cpu=20
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    # get features' stats(mean and std)
    echo "Get features' stats ..."
    python3 ${MAIN_ROOT}/utils/compute_statistics.py \
        --metadata=dump/train/raw/metadata.jsonl \
        --field-name="feats"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    # normalize, dev and test should use train's stats
    echo "Normalize ..."
   
    python3 ${BIN_DIR}/../normalize.py \
        --metadata=dump/train/raw/metadata.jsonl \
        --dumpdir=dump/train/norm \
        --stats=dump/train/feats_stats.npy
    python3 ${BIN_DIR}/../normalize.py \
        --metadata=dump/dev/raw/metadata.jsonl \
        --dumpdir=dump/dev/norm \
        --stats=dump/train/feats_stats.npy
    
    python3 ${BIN_DIR}/../normalize.py \
        --metadata=dump/test/raw/metadata.jsonl \
        --dumpdir=dump/test/norm \
        --stats=dump/train/feats_stats.npy
fi
