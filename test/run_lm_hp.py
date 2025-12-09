import argparse
from tqdm import tqdm
from typing import Dict
import gc
from openai import OpenAI, APIError, Timeout, APIConnectionError
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from utils import load_file, save_file_jsonl, metric_max_over_ground_truths, \
    exact_match_score, match, qa_f1_score, save_file_json, \
    num_tokens_from_string, check_string_exist, postprocess_output, \
    PROMPT_DICT, MODEL_PROMPT_KEY_MAPPING, fewshot_examples


def call_openai_api(openai_client: OpenAI, prompt: [Dict], model="gpt-3.5-turbo-0125", temperature=0.0, top_p=0.95,
                    max_tokens=50, chat_completions=True):
    # https://platform.openai.com/docs/guides/text-generation
    if chat_completions:
        # Chat completions API
        try:
            response = openai_client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant to answer questions."},
                    {"role": "user", "content": prompt}
                ],
            )
            result = response.choices[0].message.content
        except Exception as e:
            print(f"\nERROR: {e} =========")
            return "ERROR: API error outputs"
    else:
        # Completions API
        try:
            response = openai_client.completions.create(
                model=model,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                prompt=prompt,
            )
            result = response.choices[0].text
        except (APIError, Timeout, APIConnectionError):
            result = "ERROR: API error outputs"

    return result


def call_model(prompts, pipe, temperature=0.8, top_p=0.95, max_new_tokens=50):
    """使用HuggingFace Pipeline进行推理"""
    # Pipeline可以处理batch
    outputs = pipe(
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True if temperature > 0 else False,
        num_return_sequences=1,
        pad_token_id=pipe.tokenizer.eos_token_id,
        return_full_text=False  # 只返回生成的文本，不包含输入
    )

    # 提取生成的文本
    if isinstance(prompts, list):
        preds = [output[0]['generated_text'] for output in outputs]
    else:
        preds = [outputs[0]['generated_text']]

    return preds


def get_prompt_name(item, args):
    """ Infer prompt key from the model name and retrieval mode """

    prompt_key = MODEL_PROMPT_KEY_MAPPING[args.model_name]

    if "do_retrieval" in item:
        if item["do_retrieval"] == 1:
            prompt_name = f"{prompt_key}_retrieval"
        elif item["do_retrieval"] == 0:
            prompt_name = f"{prompt_key}_no_retrieval"
    elif args.retrieval_mode == "adaptive_retrieval":
        if args.prompt_method == "TAARE":
            prompt_name = f"{prompt_key}_adaptive_retrieval_TAARE"
        elif args.prompt_method == "vanilla":
            prompt_name = f"{prompt_key}_adaptive_retrieval"

    return prompt_name


def calculate_tokens(item):
    q_token_num = num_tokens_from_string(item["question"])
    item["q_token_num"] = q_token_num
    context_token_num = num_tokens_from_string(item["evidence"])
    item["context_token_num"] = context_token_num


def format_context(item, args):
    """如果always_retrieval，则给item添加证据，否则不加证据。同时计算token数量"""
    if "do_retrieval" in item:
        if item["do_retrieval"] == 1:
            retrieval_result = item["context"][:args.doc_top_n]

            if isinstance(retrieval_result[0], str):
                evidences = [f"[{i + 1}] {context}" for i, context in enumerate(retrieval_result)]
            else:
                # map集合
                evidences = [
                    f"[{i + 1}] {context['title'].strip() if 'title' in context else ''}\n{context['text'].strip() if 'text' in context else ''}"
                    for i, context in enumerate(retrieval_result)]
            item["evidence"] = "\n".join(evidences)

            calculate_tokens(item)

        elif item["do_retrieval"] == 0:
            item["evidence"] = ""
            calculate_tokens(item)

    elif args.retrieval_mode == "adaptive_retrieval":
        item["evidence"] = ""
        calculate_tokens(item)


def run_batch_inference(args, input_data, pipe=None, isOpenAI=None,
                        openai_client=None, chat_completions=None):
    # print(f"推理前：{input_data}")
    for idx in tqdm(range(len(input_data))):

        item = input_data[idx]
        item["today"] = datetime.today().strftime('%Y-%m-%d')
        item["fewshot_examples"] = fewshot_examples
        # 根据do_retrieval参数是否存在，判断此次是获取判断是否检索的模版名称，还是获取问题模版名称
        prompt_name = get_prompt_name(item, args)
        # 将证据放到item["evidence"]中
        format_context(item, args)
        # 替换文本中的占位符
        formatted_prompt = PROMPT_DICT[prompt_name].format_map(item)

        # print(f"============= prompt =================")
        # print(f"{formatted_prompt}\n")

        if isOpenAI:
            text = call_openai_api(
                openai_client=openai_client,
                prompt=formatted_prompt,
                model=args.model_name,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                chat_completions=chat_completions
            )
            text = postprocess_output(text, formatted_prompt)
        else:
            predictions = call_model([formatted_prompt], pipe=pipe,
                                     temperature=args.temperature,
                                     top_p=args.top_p,
                                     max_new_tokens=args.max_tokens
                                     )

            text = predictions[0]
            text = postprocess_output(text, formatted_prompt)
        # 用do_retrieval参数区分此次推理结果是判断是否需要检索，还是最终结果
        if "do_retrieval" not in item:
            # 模型决定是否检索
            item["do_retrieve_pred"] = text
            # 是否检索
            item["do_retrieval"] = check_string_exist(text)
        else:
            # 总是检索或者总是不检索
            item["model_prediction"] = text
    # print(f"推理后：{input_data}")
    return input_data


def load_model(args):
    """使用HuggingFace Pipeline加载模型"""
    print(f"Loading model: {args.model_name}")

    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # 加载tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True
    )

    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa"  # 显式指定使用 PyTorch 原生加速关注力
    )

    model = torch.compile(model)

    # 创建pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
    )

    print(f"Model loaded successfully on {device}")

    return pipe


def load_openai(args):
    with open(args.openai_config_path) as f:
        openai_api_key = f.read()

    openai_client = OpenAI(api_key=openai_api_key)

    if "gpt-4" in args.model_name or "gpt-3.5" in args.model_name:
        chat_completions = True
    else:
        chat_completions = False

    return openai_client, chat_completions


def main(args):
    isOpenAI = True if args.model_name in \
                       ["text-davinci-003", "gpt-3.5-turbo", "gpt-3.5-turbo-0125", "gpt-4-0125-preview"] else False

    ########### load model ###########
    openai_client, chat_completions, pipe = None, None, None
    if isOpenAI:
        openai_client, chat_completions = load_openai(args)
    else:
        pipe = load_model(args)

    ########### load dataset ###########
    input_data = load_file(args.input_data_path)
    print(f"# total input_data: {len(input_data)}")
    print(f"{input_data[0]}")

    if args.data_source != "retrievalqa":
        input_data = [item for item in input_data if item["data_source"] == args.data_source]
    if args.limit_input > 0:
        input_data = input_data[:args.limit_input]
    print(f"\nselected data #: {len(input_data)}, data source: {args.data_source}")

    ########### prepare retrieval context ###########
    if args.retrieval_mode == "always_retrieval":
        for item in input_data:
            item["do_retrieval"] = 1
    elif args.retrieval_mode == "no_retrieval":
        for item in input_data:
            item["do_retrieval"] = 0
    elif args.retrieval_mode == "adaptive_retrieval":
        # prompt model to decide whether to retrieve
        input_data = run_batch_inference(
            pipe=pipe,
            input_data=input_data,
            isOpenAI=isOpenAI,
            openai_client=openai_client,
            chat_completions=chat_completions,
            args=args
        )

        # reload model before inference
        if not isOpenAI:
            # 清理内存并重新加载模型
            print("Cleaning up memory and reloading model...")
            del pipe
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            print("Successfully cleaned up memory!")

            pipe = load_model(args)

    count = sum([item["do_retrieval"] for item in input_data])
    print(f"\n\n ========================== total retrieval: {count} ========================== \n")

    ########### Run prediction ###########
    input_data = run_batch_inference(
        pipe=pipe,
        input_data=input_data,
        isOpenAI=isOpenAI,
        openai_client=openai_client,
        chat_completions=chat_completions,
        args=args
    )

    ########### Calculate metrics ###########
    em_total, f1_total, acc_total, match_total = 0, 0, 0, 0
    for item in input_data:
        pred = item["model_prediction"]
        gts = item["ground_truth"]
        # EM Score (Exact Match)标准化后预测值与真实列表中的一个值完全匹配 要求答案精确
        em_score = 1.0 if metric_max_over_ground_truths(exact_match_score, pred, gts) else 0.0
        # 预测值包含第一个真实值，原始包含
        accuracy_score = 1.0 if gts[0] in pred else 0.0
        # 标准化后预测值包含真实值的任意一种
        # match accuracy（匹配准确率） 评估 QA 性能，该指标衡量标准答案是否包含在模型预测中，而不是严格的精确匹配。
        match_score = match(pred, gts)  # loose match 宽松匹配
        # 词重叠度 QA任务标准指标
        f1_score = metric_max_over_ground_truths(qa_f1_score, pred, gts)

        item["em_score"] = em_score
        item["accuracy_score"] = accuracy_score
        item["match_score"] = match_score
        item["f1_score"] = f1_score

        em_total += em_score
        f1_total += f1_score
        acc_total += accuracy_score
        match_total += match_score

    total_q_tokens = sum([item["q_token_num"] for item in input_data])
    total_context_tokens = sum([item["context_token_num"] for item in input_data])
    estimate_q_cost = total_q_tokens / 1000 * 0.0005
    estimate_context_cost = total_context_tokens / 1000 * 0.0005
    estimate_no_retrieval_cost = estimate_q_cost
    estimate_always_retrieval_cost = estimate_q_cost + estimate_context_cost

    saved_cost_rate = 1 - estimate_q_cost / (estimate_q_cost + estimate_context_cost)
    total_retrieval = sum([item["do_retrieval"] for item in input_data])

    print(
        f"\n ======= estimate no retrieval (q) API cost: {estimate_no_retrieval_cost}, total tokens #: {total_q_tokens} ================")
    print(
        f" ======= estimate always retrieval (q+context) API cost: {estimate_always_retrieval_cost}, total tokens #: {total_context_tokens + total_q_tokens} ================")
    print(f" ======= total retrieval: [{total_retrieval}/{len(input_data)}] ================\n")

    total_score = {
        # 数据源名称
        "data_source": args.data_source,
        # 数据数量
        "total_data_count": len(input_data),
        # 检索次数
        "retrieval_frequency": total_retrieval,
        # retrieval accuracy检索准确率（需不需要检索），由于作者用的数据集中的1271条数据都需要检索，所以此值约高越好
        "retrieval_rate": round(total_retrieval / len(input_data) * 100, 1),
        # match accuracy（匹配准确率），有多少条QA匹配上了
        "match_score": round(match_total / len(input_data) * 100, 1),
        "f1_score": round(f1_total / len(input_data) * 100, 1),
        "em_score": round(em_total / len(input_data) * 100, 1),
        "accuracy_score": round(acc_total / len(input_data) * 100, 1),
        "match_total": match_total,
        "f1_total": f1_total,
        "em_total": em_total,
        "accuracy_total": acc_total,
        "total_q_tokens": total_q_tokens,
        "total_context_tokens": total_context_tokens,
        "total_no_retrieval_tokens": total_q_tokens,
        "total_always_retrieval_tokens": total_context_tokens,
        "estimate_no_retrieval_cost": estimate_no_retrieval_cost,
        "estimate_always_retrieval_cost": estimate_always_retrieval_cost,
        "saved_cost_rate": saved_cost_rate,
        'args': vars(args)
    }

    print()
    print(total_score)

    # remove 'evidence' before saving results
    for item in input_data:
        if "evidence" in item:
            del item["evidence"]
        if "today" in item:
            del item["today"]
        if "fewshot_examples" in item:
            del item["fewshot_examples"]

    save_file_json(total_score, args.output_score_path)
    save_file_jsonl(input_data, args.output_prediction_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--openai_config_path', type=str, default='../openai_config.txt',
                        help='OpenAI Config file path')
    parser.add_argument('--data_source', type=str, default="retrievalqa")
    parser.add_argument('--retrieval_mode', type=str, default="no_retrieval")
    parser.add_argument('--input_data_path', type=str, default="../data/retrievalqa.jsonl", help='Input data path')
    parser.add_argument('--output_score_path', type=str, default="./output_score_path_full.jsonl",
                        help='Output json file path')
    parser.add_argument('--output_prediction_path', type=str, default="./output_prediction_path_full.jsonl",
                        help='Output jsonl file path')
    parser.add_argument('--model_name', type=str, default='TinyLlama/TinyLlama-1.1B-Chat-v1.0', help='Model name')
    parser.add_argument('--max_tokens', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--doc_top_n', type=int, default=5)
    parser.add_argument('--limit_input', type=int, default=200)
    parser.add_argument('--prompt_method', type=str, default="vanilla")
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument("--world_size", type=int, default=1,
                        help="world size to use multiple GPUs (not used in HF Pipeline version)")

    args = parser.parse_args()

    main(args)

# 200
# 100%|██████████| 200/200 [04:15<00:00,  1.28s/it]
# 加model = torch.compile(model)
# 100%|██████████| 200/200 [03:58<00:00,  1.19s/it]
# 加model = torch.compile(model)，attn_implementation="sdpa"
# 100%|██████████| 200/200 [03:36<00:00,  1.08s/it]