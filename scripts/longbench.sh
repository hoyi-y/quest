cd /media/8T3/by_lv/learn_programs/my_vllm_pro/quest/evaluation/LongBench

# /media/8T3/by_lv/learn_programs/models/Qwen3-1.7B/

model="Qwen3-1.7B"
# export HF_ENDPOINT=https://hf-mirror.com

for task in "triviaqa"                  # "qasper" "narrativeqa" "hotpotqa" "multifieldqa_en" "gov_report" "multifieldqa_en"
do
    python -u pred.py \
        --model $model --task $task

    for budget in 512 #1024 2048 4096
    do
        python -u pred.py \
            --model $model --task $task \
            --quest --token_budget $budget --chunk_size 16
    done
done

python -u eval.py --model $model

