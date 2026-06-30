device_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
INCLUDE_STR="localhost:$CUDA_VISIBLE_DEVICES"
unset CUDA_VISIBLE_DEVICES

FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using free port: $FREE_PORT"

TRAIN_JSON=/path/to/pretrain/data/train_split/single_image.json
VAL_JSON=/path/to/pretrain/data/val_split/single_image.json
OUTPUT_DIR=./checkpoints_phase1


learning_rates=("3e-3")
batch_sizes=(32)
gradient_accumulate=(1)
epochs=(5)
vocab_sizes=(6000)

for vocab_size in "${vocab_sizes[@]}"; do
    for bs in "${batch_sizes[@]}"; do
        for gacc in "${gradient_accumulate[@]}"; do
            for lr in "${learning_rates[@]}"; do
                for ep in "${epochs[@]}"; do
                    deepspeed --include=$INCLUDE_STR --master_port=$FREE_PORT llava/train/train_mem.py \
                        --deepspeed ./scripts/zero2.json \
                        --run_name babyllava_vit_tinyllama_SAYCam_phase1_lr${lr}_bs$((bs*gacc*device_num))_epoch${ep}_vocab${vocab_size} \
                        --model_name_or_path /path/to/phase0/ckpt/folder \
                        --version baby_plain \
                        --data_path $TRAIN_JSON \
                        --val_data_path $VAL_JSON \
                        --image_folder /path/to/image/root/folder \
                        --vision_tower /path/to/vision/tower/checkpoint.pth \
                        --tune_mm_mlp_adapter True \
                        --mm_projector_type mlp2x_gelu \
                        --mm_vision_select_layer -2 \
                        --mm_use_im_start_end False \
                        --mm_use_im_patch_token False \
                        --image_aspect_ratio pad \
                        --group_by_modality_length True \
                        --bf16 True \
                        --output_dir $OUTPUT_DIR \
                        --num_train_epochs $ep \
                        --per_device_train_batch_size $bs \
                        --per_device_eval_batch_size $bs \
                        --gradient_accumulation_steps $gacc \
                        --evaluation_strategy "steps" \
                        --eval_steps 0.2 \
                        --save_strategy "steps" \
                        --save_steps 0.2 \
                        --save_total_limit 5 \
                        --learning_rate $lr \
                        --weight_decay 0. \
                        --warmup_ratio 0.03 \
                        --lr_scheduler_type "cosine" \
                        --logging_steps 1 \
                        --tf32 True \
                        --model_max_length 8192 \
                        --gradient_checkpointing True \
                        --dataloader_num_workers 4 \
                        --lazy_preprocess True \
                        --report_to wandb
                done
            done
        done
    done
done
