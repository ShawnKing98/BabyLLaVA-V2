FREE_PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
echo "Using free port: $FREE_PORT"

device_num=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
INCLUDE_STR="localhost:$CUDA_VISIBLE_DEVICES"
unset CUDA_VISIBLE_DEVICES


learning_rates=("5e-5")
mm_vision_lrs=("1e-5")
batch_sizes=(128)
gradient_accumulate=(1)
epochs=(10)


backbones=(
    "/path/to/phase2/ckpt/folder"
)

backbone_names=(
    "vit_tinyllama"
)

datasets=(
    "path/to/sft_data/object_counting/count_instructions_train.json"
    "path/to/sft_data/visual_delay_response/binary_dataset_train.json"
    "path/to/sft_data/visual_delay_response/open_dataset_train.json"
)

run_names=(
    "count_instruction_ft"
    "vdr_binary_instruction_ft"
    "vdr_open_instruction_ft"
)



for ((j=0; j<${#backbones[@]}; j++)); do
    backbone=${backbones[$j]}
    backbone_name=${backbone_names[$j]}

    for ((i=0; i<${#datasets[@]}; i++)); do
        dataset=${datasets[$i]}
        run_name=${run_names[$i]}

        for bs in "${batch_sizes[@]}"; do
            for gacc in "${gradient_accumulate[@]}"; do
                for lr in "${learning_rates[@]}"; do
                    for mm_vision_lr in "${mm_vision_lrs[@]}"; do
                        for ep in "${epochs[@]}"; do
                            deepspeed --include=$INCLUDE_STR --master_port=$FREE_PORT llava/train/train_mem.py \
                                --deepspeed ./scripts/zero2_phase2.json \
                                --run_name babyllava_${backbone_name}_${run_name}_lr${lr}_visionlr${mm_vision_lr}_bs$((bs*gacc*device_num))_epoch${ep} \
                                --model_name_or_path $backbone \
                                --version baby_v1 \
                                --data_path $dataset \
                                --image_folder /path/to/image/root/folder \
                                --vision_tower /path/to/vision/tower/checkpoint.pth \
                                --tune_vision_tower True \
                                --pretrain_mm_mlp_adapter /path/to/phase1/ckpt/folder/mm_projector.bin \
                                --mm_projector_type mlp2x_gelu \
                                --mm_vision_select_layer -2 \
                                --mm_use_im_start_end False \
                                --mm_use_im_patch_token False \
                                --image_aspect_ratio pad \
                                --group_by_modality_length True \
                                --bf16 True \
                                --output_dir ./checkpoints_instruction_ft \
                                --num_train_epochs $ep \
                                --per_device_train_batch_size $bs \
                                --per_device_eval_batch_size 64 \
                                --gradient_accumulation_steps $gacc \
                                --evaluation_strategy "no" \
                                --save_strategy "steps" \
                                --save_steps 1000 \
                                --save_total_limit 1 \
                                --learning_rate $lr \
                                --mm_vision_lr $mm_vision_lr \
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
    done
done