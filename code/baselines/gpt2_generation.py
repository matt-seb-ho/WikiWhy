import argparse
from time import perf_counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from wikiwhy_gpt2 import (
    AnswerModel, 
    ExplainerModel,
    WikiWhyQA, 
    WikiWhyExplain,
    build_qa_input,
    build_exp_input
)   

# ### Custom generation method
# HuggingFace's generate() does not take token_type_ids as input.
# 
# Adapted from Thomas Wolf's repo: 
# https://github.com/huggingface/transfer-learning-conv-ai/blob/master/interact.py

@dataclass
class GenerationArgs:
    max_length = 222
    max_gen_length = 171
    min_length = 0
    temperature = 1.0
    top_k = 50
    top_p = 1.0
    sample = False
    device = "cpu"


def top_filtering(logits, top_k=0, top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ 
    Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering

    Args:
        logits: logits distribution shape (vocabulary size)
        top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
        top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
            whose total probability mass is greater than or equal to the threshold top_p.
            In practice, we select the highest probability tokens whose cumulative probability mass exceeds
            the threshold top_p.
        threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits

def generate(
    inputs,
    tokenizer, 
    model,
    reformat_example,
    task,
    special_tokens,
    args=GenerationArgs(), 
    current_output=[]
):
    # handle current output as a string
    if isinstance(current_output, str):
        current_output = tokenizer.encode(current_output)

    model.to(args.device)
    special_tokens_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    encoded_inputs = {k: tokenizer.encode(v) for k, v in inputs.items()}

    for i in range(args.max_exp_length):
        # instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)
        instance = reformat_example(encoded_inputs, current_output, tokenizer, special_tokens, add_eos=False)
        if task == "qa":
            instance = build_qa_input(
                encoded_inputs["question"],
                current_output, 
                tokenizer=tokenizer, 
                special_tokens=special_tokens,
                add_eos=False
            )

        else:
            instance = build_exp_input(
                encoded_inputs["cause"],
                encoded_inputs["effect"],
                current_output, 
                tokenizer=tokenizer, 
                special_tokens=special_tokens,
                add_eos=False
            )
            # print(instance)

        input_ids = torch.tensor(
            instance["input_ids"], 
            device=args.device
        ).unsqueeze(0)
        token_type_ids = torch.tensor(
            instance["type_ids"],
            device=args.device
        ).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)["logits"]
        logits = logits[0, -1, :] / args.temperature
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if not args.sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return tokenizer.decode(current_output)


def df_generate(task, df, tokenizer, model, args, new_column_name="generated"):
    if task == "qa":
        input_fields = ["question"] 
        format_input = build_qa_input
        special_tokens = AnswerModel.SPECIAL_TOKENS
    else: 
        input_fields = ["cause", "effect"]
        format_input = build_exp_input
        special_tokens = ExplainerModel.SPECIAL_TOKENS
    

    df[new_column_name] = df.apply(
        lambda r: generate(
            inputs=r[input_fields].to_dict(),
            tokenizer=tokenizer,
            model=model,
            reformat_example=format_input,
            task=task,
            special_tokens=special_tokens,
            args=args,
            current_output=[]
        ),
        axis=1
    )


def gen2(
    inputs,
    model,
    build_input,
    special_tokens_ids,
    args=GenerationArgs(), 
):
    current_output = []
    for i in range(args.max_gen_length):
        instance = build_input(inputs, current_output)
        input_ids = torch.tensor(
            instance["input_ids"], 
            device=args.device
        ).unsqueeze(0)
        token_type_ids = torch.tensor(
            instance["type_ids"],
            device=args.device
        ).unsqueeze(0)

        logits = model(input_ids, token_type_ids=token_type_ids)["logits"]
        logits = logits[0, -1, :] / args.temperature 
        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if not args.sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def generate_column(task, df, tokenizer, model, args, new_label="generated"):
    if task == "qa":
        input_fields = ["question"] 
        special_tokens = AnswerModel.SPECIAL_TOKENS
        def build_input(inputs, outputs):
            return build_qa_input(
                inputs[0],
                outputs, 
                tokenizer=tokenizer, 
                special_tokens=special_tokens,
                add_eos=False
            )
    else:
        input_fields = ["cause", "effect"]
        special_tokens = ExplainerModel.SPECIAL_TOKENS
        def build_input(inputs, outputs):
            return build_exp_input(
                inputs[0],
                inputs[1],
                outputs, 
                tokenizer=tokenizer, 
                special_tokens=special_tokens,
                add_eos=False
            )

    special_token_ids = tokenizer.convert_tokens_to_ids(special_tokens)
    df[new_label] = df.apply(
        lambda r: tokenizer.decode(gen2(
            [tokenizer.encode(r[field]) for field in input_fields],
            model,
            build_input,
            special_token_ids,
            args=args
        )),
        axis=1
    )


if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument("--input", required=True)
    psr.add_argument("--output", required=True)
    psr.add_argument("--checkpoint", required=True)
    psr.add_argument("--task", choices=["qa", "exp"], required=True)
    psr.add_argument("--temp", type=float, default=1.0)
    psr.add_argument("--max_length", type=int, default=222)
    psr.add_argument("--max_gen_length", type=int, default=171)
    psr.add_argument("--min_length", type=int, default=0)
    psr.add_argument("--temperature", type=float, default=1.0)
    psr.add_argument("--top_k", type=int, default=50)
    psr.add_argument("--top_p", type=float, default=1.0)
    psr.add_argument("--sample", type=bool, default=False)
    psr.add_argument("--device", default="cpu")
    
    args = psr.parse_args()
    
    input_df = pd.read_csv(args.input, index_col="id")
    print(f"\nDataset Summary\n - total_size: {len(input_df)}")
    print(" - columns:", input_df.columns.to_list())
    
    # get test split
    test_set = input_df
    if "split" in input_df.columns.tolist():
        test_set = input_df.loc[input_df["split"] == "test"]
    test_set = test_set.copy()
    print("test set size:", len(test_set))
    
    explain = args.task == "exp"
    model = (
        ExplainerModel if explain 
        else AnswerModel
    ).load_from_checkpoint(args.checkpoint)
    
    # print(model.hyperparams)
    # print(model.tokenizer)
    # print(model.model)
    # print(vars(args))

    # setup device
    # args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.model.to(args.device)

    # generate completions
    start = perf_counter()
    generate_column(args.task, test_set, model.tokenizer, model.model, args)
    print(f"Finished in {perf_counter() - start}s")

    # save file
    test_set.to_csv(args.output)
