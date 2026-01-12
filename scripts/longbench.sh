cd /media/8T3/by_lv/learn_programs/my_vllm_pro/quest/evaluation/LongBench

# /media/8T3/by_lv/learn_programs/models/Qwen3-1.7B/

model="Qwen3-1.7B"
# export HF_ENDPOINT=https://hf-mirror.com
# 建议先定义好 model 变量，或者确保外部已经定义
# model="你的模型路径"

for task in "qasper" "narrativeqa" "hotpotqa" "gov_report" "multifieldqa_en" #"triviaqa" 
do
    echo "========================================================"
    echo ">>> 进入任务: $task"

    # ------------------------------------------------------
    # 1. 记录 Baseline (第一次运行) 的时间
    # ------------------------------------------------------
    start_base=$(date +%s)
    
    python -u pred.py \
        --model $model --task $task

    end_base=$(date +%s)
    duration_base=$((end_base - start_base))
    
    echo "✅ [Baseline] $task 完成 - 耗时: ${duration_base} 秒"


    # ------------------------------------------------------
    # 2. 循环 budget 进行 Quest 运行
    # ------------------------------------------------------
    for budget in 512 1024 2048 4096
    do
        # 3. 记录 Quest (第二次运行) 的时间
        start_quest=$(date +%s)

        python -u pred.py \
            --model $model --task $task \
            --quest --token_budget $budget --chunk_size 16

        end_quest=$(date +%s)
        duration_quest=$((end_quest - start_quest))

        echo "✅ [Quest] $task (Budget: $budget) 完成 - 耗时: ${duration_quest} 秒"
    done
done

python -u eval.py --model $model


