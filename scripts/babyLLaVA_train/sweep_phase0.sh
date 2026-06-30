FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using free port: $FREE_PORT"

TRAIN_JSON=/path/to/language_corpus_train.json
VAL_JSON=/path/to/language_corpus_val.json
OUTPUT_DIR=./checkpoints_phase0


learning_rates=("2e-4" "2e-5")
batch_sizes=(4)
gradient_accumulate=(1)
epochs=(10 5)
vocab_sizes=(6000)


for vocab_size in "${vocab_sizes[@]}"; do
    for bs in "${batch_sizes[@]}"; do
        for gacc in "${gradient_accumulate[@]}"; do
            for lr in "${learning_rates[@]}"; do
                for ep in "${epochs[@]}"; do
                    accelerate launch --main_process_port=$FREE_PORT llava/train/train_babylm.py \
                        --train_file $TRAIN_JSON \
                        --val_file $VAL_JSON \
                        --cache_dir $OUTPUT_DIR/.cache \
                        --arch tinyllama \
                        --tokenizer bpe \
                        --ctx_len 8192 \
                        --lr $lr \
                        --gacc $gacc \
                        --bs $bs \
                        --epoch $ep \
                        --output_dir $OUTPUT_DIR \
                        --positional_embedding sinusoide \
                        --vocab_size $vocab_size
                done
            done
        done
    done
done