import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoConfig,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModelForCausalLM,
)


from tqdm import tqdm
import numpy as np
import random
import argparse
from evaluation.quest_attention import enable_quest_attention_eval
from evaluation.llama import enable_tuple_kv_cache_for_llama 
from evaluation.mistral import enable_tuple_kv_cache_for_mistral
from evaluation.qwen3 import enable_tuple_kv_cache_for_qwen3


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "llama2-7b-chat-4k",
            "longchat-v1.5-7b-32k",
            "xgen-7b-8k",
            "internlm-7b-8k",
            "chatglm2-6b",
            "chatglm2-6b-32k",
            "chatglm3-6b-32k",
            "vicuna-v1.5-7b-16k",
            "Mistral-7B-Instruct-v0.3",
            "Meta-Llama-3.1-8B-Instruct",
            "Qwen3-1.7B",
            "Qwen2.5-0.5B-Instruct",
        ],
    )
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")

    parser.add_argument("--task", type=str, help="task name", required=True)

    parser.add_argument("--token_budget", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--quest", action="store_true", help="Enable Quest Attention")

    return parser.parse_args(args)


# This is the customized building prompt for chat models
def build_chat(tokenizer, prompt, model_name):
    if "chatglm3" in model_name:
        prompt = tokenizer.build_chat_input(prompt)
    elif "chatglm" in model_name:
        prompt = tokenizer.build_prompt(prompt)
    elif "longchat" in model_name or "vicuna" in model_name:
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("vicuna")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    elif "llama2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    elif "xgen" in model_name:
        header = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions.\n\n"
        )
        prompt = header + f" ### Human: {prompt}\n###"
    elif "internlm" in model_name:
        prompt = f"<|User|>:{prompt}<eoh>\n<|Bot|>:"
    elif "qwen" in model_name:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


def is_qwen_base(model_name):
    name = model_name.lower()
    return "qwen" in name and "instruct" not in name and "chat" not in name


def extract_answer(pred, dataset):
    qa_datasets = {
        "qasper",
        "narrativeqa",
        "multifieldqa_en",
        "multifieldqa_zh",
        "hotpotqa",
        "2wikimqa",
        "triviaqa",
    }
    if dataset not in qa_datasets:
        return pred
    lines = [line.strip() for line in pred.splitlines() if line.strip()]
    if "Answer:" in pred:
        parts = pred.split("Answer:")
        for part in parts[1:]:
            for line in part.splitlines():
                line = line.strip()
                if line:
                    return line
    if lines:
        return lines[0]
    return pred.strip()


def get_pred_generate(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
):
    preds = []
    no_chat_datasets = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}

    for json_obj in tqdm(data):
        # 1) raw prompt
        prompt = prompt_format.format(**json_obj)

        # 2) truncate in the middle to fit max_length (token-level)
        tokenized = tokenizer(prompt, truncation=False, return_tensors="pt")
        tokenized_prompt = tokenized.input_ids[0]

        # chatglm3 special handling (keep your original behavior)
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]

        if len(tokenized_prompt) > max_length:
            half = max_length // 2
            prompt = (
                tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True)
                + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
            )

        # 3) build chat template for chat models (except some datasets)
        if dataset not in no_chat_datasets and not is_qwen_base(model_name):
            prompt = build_chat(tokenizer, prompt, model_name)

        # 4) tokenize final prompt
        # add_special_tokens: 对于你手拼的 qwen3 模板，通常用默认(True)也能跑
        # 但为了避免重复插入 BOS 等，建议 False/True 以模型表现为准。
        # 这里用默认行为（不强行改），更通用。
        inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")

        # 5) generate
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_gen,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                # 可选：如果你遇到提前停不下来/停太早，再加 eos_token_id
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )[0]

        # 6) decode only newly generated tokens
        gen_ids = out[inputs["input_ids"].shape[-1] :]
        pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
        pred = extract_answer(pred, dataset)
        pred = post_process(pred, model_name)

        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )

    return preds


def get_pred(
    model,
    tokenizer,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    device,
    model_name,
):
    import transformers
    print(transformers.__version__)
    preds = []
    for json_obj in tqdm(data):
        raw = prompt_format.format(**json_obj)

        # 1) 在 raw 上找切点
        q_pos = raw.rfind("Question:") if dataset in ["qasper","hotpotqa"] else -1
        # ... 其他 dataset 的切点同理
        if q_pos != -1:
            ctx_raw, q_raw = raw[:q_pos], raw[q_pos:]
        else:
            ctx_raw, q_raw = raw, ""

        # 2) 对 ctx_raw 做 build_chat（重要：不要先 build 再切）
        if dataset not in ["trec","triviaqa","samsum","lsht","lcc","repobench-p"] and not is_qwen_base(model_name):
            ctx_text = build_chat(tokenizer, ctx_raw, model_name)
        else:
            ctx_text = ctx_raw

        # ⚠️ q_raw 是否需要也套 chat 模板：取决于你“模拟”的语义
        # 最稳的研究做法是：不要把问题当成裸文本塞进 assistant 段里乱拼，
        # 你要么把 question 也变成同一模板体系下的“user 补充”，要么就别两段式。
        q_text = q_raw  # 先这样最小化改动

        ctx = tokenizer(ctx_text, return_tensors="pt", add_special_tokens=False).to("cuda")
        q = tokenizer(q_text, return_tensors="pt", add_special_tokens=False).to("cuda")

        # 3) prefill：吃上下文
        with torch.no_grad():
            out = model(
                input_ids=ctx.input_ids,
                attention_mask=torch.ones_like(ctx.input_ids),
                use_cache=True,
            )
            cache = out.past_key_values

            # 4) decode：逐 token 吃 question（带 position_ids 递增）
            pos = ctx.input_ids.shape[1]
            for tok in q.input_ids[0]:
                position_ids = torch.tensor([[pos]], device="cuda", dtype=torch.long)
                attention_mask = torch.ones((1, pos + 1), device="cuda", dtype=torch.long)

                out = model(
                    input_ids=tok.view(1, 1),
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
                pos += 1

            # 5) 开始生成（同样维护 position_ids/attention_mask）
            next_tok = out.logits[:, -1, :].argmax(dim=-1).view(1, 1)
            generated = [next_tok.item()]
            for _ in range(max_gen - 1):
                position_ids = torch.tensor([[pos]], device="cuda", dtype=torch.long)
                attention_mask = torch.ones((1, pos + 1), device="cuda", dtype=torch.long)

                out = model(
                    input_ids=next_tok,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=cache,
                    use_cache=True,
                )
                cache = out.past_key_values
                next_tok = out.logits[:, -1, :].argmax(dim=-1).view(1, 1)
                generated.append(next_tok.item())
                pos += 1
                if next_tok.item() == tokenizer.eos_token_id:
                    break

        pred = tokenizer.decode(generated, skip_special_tokens=True)
        pred = extract_answer(pred, dataset)
        # inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        # with torch.no_grad():
        #     out = model.generate(
        #         **inputs,
        #         max_new_tokens=max_gen,
        #         do_sample=False,
        #         num_beams=1,
        #         use_cache=True,
        #     )[0]
        # pred = tokenizer.decode(out[inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        # pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
        pred = post_process(pred, model_name)
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
    return preds


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name, device):
    if 'llama' in model_name.lower() or 'longchat' in model_name.lower():
        enable_tuple_kv_cache_for_llama()
    if 'mistral' in model_name.lower():
        enable_tuple_kv_cache_for_mistral()

        
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = model.eval()

    if args.quest:
        enable_quest_attention_eval(model, args)

    return model, tokenizer


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("config/model2path.json", "r"))
    model2maxlen = json.load(open("config/model2maxlen.json", "r"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = args.model
    # define your model
    model, tokenizer = load_model_and_tokenizer(
        model2path[model_name], model_name, device
    )
    print(model.config.model_type)
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = [args.task]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("pred"):
        os.makedirs("pred")
    if not os.path.exists("pred_e"):
        os.makedirs("pred_e")
    for dataset in datasets:
        # 1. 构造本地文件的绝对路径
        if args.e:
            # 比如 LongBench-E 的文件路径
            local_path = f"/media/8T3/by_lv/learn_programs/my_vllm_pro/quest/evaluation/LongBench/data/{dataset}_e.jsonl"
        else:
            # 标准 LongBench 的文件路径
            local_path = f"/media/8T3/by_lv/learn_programs/my_vllm_pro/quest/evaluation/LongBench/data/{dataset}.jsonl"
        
        if args.e:
            # data = load_dataset("/media/8T3/by_lv/learn_programs/my_vllm_pro/quest/evaluation/LongBench/data.zip", f"{dataset}_e", split="test")
            data = load_dataset("json", data_files={"test": local_path}, split="test")
            # data = load_dataset("THUDM/LongBench", f"{dataset}_e", split="test")
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
            if args.quest:
                out_path = f"pred_e/{model_name}/{dataset}-{args.token_budget}.jsonl"
            else:
                out_path = f"pred_e/{model_name}/{dataset}-full.jsonl"
        else:
            # data = load_dataset("/media/8T3/by_lv/learn_programs/my_vllm_pro/quest/evaluation/LongBench/data.zip", dataset, split="test")
            # data = load_dataset("THUDM/LongBench", dataset, split="test")
            data = load_dataset("json", data_files={"test": local_path}, split="test")
            if not os.path.exists(f"pred/{model_name}"):
                os.makedirs(f"pred/{model_name}")
            if args.quest:
                out_path = f"pred/{model_name}/{dataset}-{args.token_budget}.jsonl"
            else:
                out_path = f"pred/{model_name}/{dataset}-full.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(
            model,
            tokenizer,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            device,
            model_name,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")
