model=llava-v1.5-7b
python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained=checkpoints/${model} \
    --tasks ai2d,mmvet,mmmu,mmmu_pro \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix ${model}_four_mmb \
    --output_path ./logs/${model}
