img_root=playground/data/coco/val2014
save_root=outputs/pope/aokvqa
model_root=checkpoints/
model=llava-1.5-7b-665k-pyramid3x
save_attention_path=attention_visualization/${model}
model_name=${model}
pope_subset=adversarial
# pope_subset=popular
# pope_subset=random

echo "------------- Running for model: $model -------------"

question_file=playground/data/pope/seem/aokvqa/aokvqa_pope_seem_${pope_subset}.json
answer_file=${save_root}/${model_name}_pope_aokvqa_${pope_subset}.jsonl

if test -e ${answer_file}; then
    python llava/eval/eval_pope.py \
        --question-file ${question_file} \
        --result-file ${answer_file}
else
    python llava/eval/model_vqa_pope.cca.py \
        --model-path ${model_root}/${model} \
        --question-file ${question_file} \
        --image-folder ${img_root} \
        --answers-file ${answer_file} \
        --save-attention-path ${save_attention_path} \
        # --visualize \
        # --layer-wise-attention \
        # --iter-num 61 
    python llava/eval/eval_pope.py \
        --question-file ${question_file} \
        --result-file ${answer_file} 
fi
