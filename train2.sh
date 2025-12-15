MODEL_ID="stable-diffusion-v1-5/stable-diffusion-v1-5"
LOGGING_DIR="logs"
OUTPUT_DIR="p2p_test"
accelerate launch --mixed_precision="fp16" train_instruct_p2p.py \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --resolution=256 \
    --train_batch_size=4 --gradient_accumulation_steps=4 --gradient_checkpointing \
    --max_train_steps=15000 \
    --checkpointing_steps=5000 --checkpoints_total_limit=2 \
    --learning_rate=5e-05 --max_grad_norm=1 --lr_warmup_steps=0 \
    --conditioning_dropout_prob=0.05 \
    --mixed_precision=fp16 \
    --seed=42 \
    --logging_dir=$LOGGING_DIR \
    --report_to="all" \
    --output_dir=$OUTPUT_DIR
