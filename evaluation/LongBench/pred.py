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
            "Qwen3-1.7B",       #新添加
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
    elif "qwen3" in model_name or "qwen" in model_name:
        prompt = (
            "<|im_start|>system\n"
            "You are a helpful assistant.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    return prompt


def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    return response


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
    preds = []
    for json_obj in data: #原代码为tqdm(data)
        prompt = prompt_format.format(**json_obj)
        # truncate to fit max_length (we suggest truncate in the middle, since the left and right side may contain crucial instructions)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if "chatglm3" in model_name:
            tokenized_prompt = tokenizer(
                prompt, truncation=False, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        # split the prompt and question (simulate decoding in the question stage)
        if dataset in ["qasper", "hotpotqa"]:
            q_pos = prompt.rfind("Question:")
        elif dataset in ["multifieldqa_en", "gov_report"]:
            q_pos = prompt.rfind("Now,")
        elif dataset in ["triviaqa"]:
            q_pos = prompt.rfind("Answer the question")
        elif dataset in ["narrativeqa"]:
            q_pos = prompt.rfind("Do not provide")
        else:
            q_pos = -1

        # max simulation length is 100
        q_pos = max(len(prompt) - 100, q_pos)

        # if q_pos != None:
        #     question = prompt[q_pos:]
        #     prompt = prompt[:q_pos]
        if q_pos is not None and q_pos >= 0:
            question = prompt[q_pos:]
            prompt = prompt[:q_pos]
        else:
            question = ""    

        if "chatglm3" in model_name:
            # input = prompt.to(device)
            input = prompt.to("cuda")
        else:
            # input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
            q_input = tokenizer(question, truncation=False, return_tensors="pt").to(
                "cuda"
            )
            q_input.input_ids = q_input.input_ids[:, 1:]

        context_length = input.input_ids.shape[-1] + q_input.input_ids.shape[-1]

        if (
            dataset == "samsum"
        ):  # prevent illegal output on samsum (model endlessly repeat "\nDialogue"), might be a prompting issue
            assert False
            output = model.generate(
                **input,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=False,
                temperature=1.0,
                min_length=context_length + 1,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.encode("\n", add_special_tokens=False)[-1],
                ],
            )[0]
        else:
            with torch.no_grad():
                output = model(
                    input_ids=input.input_ids,
                    past_key_values=None,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
                for input_id in q_input.input_ids[0]:
                    output = model(
                        input_ids=input_id.unsqueeze(0).unsqueeze(0),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = output.past_key_values

                pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content = [pred_token_idx.item()]
                for _ in range(max_gen - 1):
                    outputs = model(
                        input_ids=pred_token_idx,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    # print("max_gen变化………………………………")

                    past_key_values = outputs.past_key_values
                    pred_token_idx = (
                        outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    )
                    generated_content += [pred_token_idx.item()]
                    stop_token_ids = set()

                    # 1. 添加默认的 EOS (通常是 151643 <|endoftext|>)
                    if tokenizer.eos_token_id is not None:
                        stop_token_ids.add(tokenizer.eos_token_id)

                    # 2. 【必需】添加 ChatML 专用停止符 <|im_end|> (通常是 151645)
                    # 很多时候 tokenizer.eos_token_id 不包含这个，导致 Chat 模型停不下来
                    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                    if isinstance(im_end_id, int):
                        stop_token_ids.add(im_end_id)

                    # 3. 【必需】添加 Base 模型结束符 <|endoftext|> (作为双重保险)
                    endoftext_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
                    if isinstance(endoftext_id, int):
                        stop_token_ids.add(endoftext_id)

                    # 4. 【强烈建议】添加 "换行符" (针对 QA 评测)
                    # 作用：防止模型回答完一个问题后，换行继续编造 "Answer: ..."
                    # newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
                    # stop_token_ids.add(newline_id)
                    # print(pred_token_idx.item()) 
                    # print(tokenizer.decode([pred_token_idx.item()]))
                    # stop_token_ids = {tokenizer.eos_token_id, 151645, 151668} # Qwen3 特定的停止符
                    # if True:
                    if pred_token_idx.item() in stop_token_ids:
                        # print(f"检测到停止符: {pred_token_idx.item()}")
                        break

            # output = model.generate(
            #     **input,
            #     max_new_tokens=max_gen,
            #     num_beams=1,
            #     do_sample=False,
            #     temperature=1.0,
            # )[0]

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
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
    # if 'qwen' in model_name.lower():
    #     enable_tuple_kv_cache_for_qwen3()
        
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda"  
    )
    model = model.eval()

    if args.quest:
        print("enable quest-------")
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
        if args.e:
            data_file = f"/media/8T3/by_lv/learn_programs/my_vllm_pro/quest/evaluation/LongBench/data/{dataset}_e.jsonl"
        else:
            data_file = f"/media/8T3/by_lv/learn_programs/my_vllm_pro/quest/evaluation/LongBench/data/{dataset}.jsonl"
 
        # 直接读取本地 jsonl 文件
        with open(data_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        if args.e:
            # data = load_dataset("/media/8T3/by_lv/learn_programs/my_vllm_pro/quest/evaluation/LongBench/data", f"{dataset}_e", split="test")
            if not os.path.exists(f"pred_e/{model_name}"):
                os.makedirs(f"pred_e/{model_name}")
            out_path = f"pred_e/{model_name}/{dataset}.jsonl"
            if args.quest:
                out_path = f"pred_e/{model_name}/{dataset}-{args.token_budget}.jsonl"
            else:
                out_path = f"pred_e/{model_name}/{dataset}-full.jsonl"
        else:
            # data = load_dataset("/media/8T3/by_lv/learn_programs/my_vllm_pro/quest/evaluation/LongBench/data", dataset, split="test")
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
        if True:
            break       

