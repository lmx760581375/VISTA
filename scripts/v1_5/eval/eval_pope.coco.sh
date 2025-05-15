img_root=playground/data/coco/val2014
save_root=outputs/pope/coco
model_root=checkpoints/
model=llava-v1.5-7b
save_attention_path=attention_visualization/${model}
# save_attention_path=visualization/${model}
# save_attention_path=attention_visualization/test
model_name=${model}
pope_subset=adversarial
# pope_subset=popular
# pope_subset=random

echo "------------- Running for model: $model -------------"

question_file=playground/data/pope/coco/coco_pope_${pope_subset}.json
answer_file=${save_root}/${model_name}_pope_coco_${pope_subset}.jsonl
# answer_file=${save_root}/${model_name}_pope_coco_${pope_subset}_test_layer.jsonl

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
        # --save-attention-path ${save_attention_path} \
        # --visualize \
        # --layer-wise-attention \
        # --iter-num 61 
    python llava/eval/eval_pope.py \
        --question-file ${question_file} \
        --result-file ${answer_file} 
fi

# rm -rf ${answer_file}
