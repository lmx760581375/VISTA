model=llava-v1.5-7b
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained=checkpoints/${model} \
    --tasks mmbench \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${model}_mmbench \
    --output_path ./logs/${model}
