
cd /media/8T3/by_lv/learn_programs/my_vllm_pro/quest/evaluation/passkey
# cd evaluation/passkey

MODEL=Llama-3.1-8B-Instruct
MODELPATH=/media/8T3/by_lv/work_projects/models/llama/Llama-3.1-8B-Instruct/
OUTPUT_DIR=results/$MODEL

mkdir -p $OUTPUT_DIR

length=40000

for token_budget in 512 1024 2048 4096
do
    python passkey.py -m $MODELPATH \
        --iterations 100 --fixed-length $length \
        --quest --token_budget $token_budget --chunk_size 16 \
        --output-file $OUTPUT_DIR/$MODEL-quest-$token_budget.jsonl
        
done
