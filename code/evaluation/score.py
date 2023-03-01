from collections import defaultdict
from time import perf_counter
import json
import random
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import evaluate # huggingface evaluate library

# -------------------------------
# constants

# default roberta large is overly permissive
BERTSCORE_MODEL = "microsoft/deberta-xlarge-mnli"

# F-score columns
PRECISE_COLUMN = "precise"
COVERED_COLUMN = "covered"
TRUE_POSITIVES_COLUMN = "true_positive"
POSITIVE_PREDICTIONS_COLUMN = "predicted_positive"
RELEVANT_COLUMN = "relevant"

# match thresholding
BERT_THRESHOLD = 0.60
BLEURT_THRESHOLD = 0.15
MATCH_THRESHOLD = BERT_THRESHOLD

# -------------------------------
# load BLEURT 

# ent_bank uses bleurt-large-512
# bleurt = evaluate.load("bleurt", "bleurt-large-512")


# -------------------------------
# load bertscore 

bert = evaluate.load("bertscore")


def plain_bertscore(df, pcol, rcol):
    return bert.compute(
        predictions=df[pcol].tolist(),
        references=df[rcol].tolist(),
        device="cuda:2",
        model_type=BERTSCORE_MODEL
    )

# -------------------------------
# helper routines

def split_into_entries(exp):
    """
    split by sentence, removing periods and empty strings
    
    """
    if isinstance(exp, str):
        res = [item.strip() for item in exp.strip('. ').split('.')]
        return list(filter(lambda s: s, res))
    else:
        raise TypeError(f"{type(exp)} is not supported")


def expand_sequences(lst):
    perms = []
    def add(idx, stem):
        if idx == len(lst):
            perms.append(stem)
        else:
            if lst[idx]:
                for opt in lst[idx]:
                    cpy = stem.copy()
                    cpy.append(opt)
                    add(idx + 1, cpy)
            else:
                add(idx + 1, stem)
    add(0, [])
    return perms


def lcs(s1, s2):
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
    for i, e1 in enumerate(s1):
        for j, e2 in enumerate(s2):
            if e1 == e2:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[-1][-1]


def pretty_print(d):
    print(json.dumps(d, indent=2))


# -------------------------------
# unordered score

def unordered_row_score(
    row, 
    pred_column, 
    ref_column, 
    score_fn, 
    match_threshold,
    score_cache=None, 
    update_cache=False,
    score_args={},
    res_key="scores"
):
    """
    Initial implementation allows multiple pred sentences to match
    the same reference sentence. Depends on BLEURT_THRESHOLD
    
    Args
        row (pd.Series): df row 
        pred (str): prediction column name 
        ref (str): reference (gold) column name 
        score_cache (defaultdict(list))
        update_cache (bool)
        
    Returns
        (pd.Series): [add description]
        
    Side Effects
        if update_cache=True and score_cache is present, 
        updates cache with newly computed BLEURT scores.
    """
    # prep prediction and reference as list of sentences
    preds = split_into_entries(row[pred_column])
    refs = split_into_entries(row[ref_column])
        
    output = {
        PRECISE_COLUMN: 0,
        POSITIVE_PREDICTIONS_COLUMN: len(preds),
        COVERED_COLUMN: set(),
        RELEVANT_COLUMN: len(refs)
    }
    
    # for each predicted idea, align against gold idea
    for idx, pred in enumerate(preds):
        if score_cache and not update_cache:
            # read-only mode
            scores = score_cache[row.name][idx]
        else:
            res = score_fn.compute(
                predictions=([pred] * len(refs)), 
                references=refs,
                **score_args
            )
            """
            print(
                f"predictions: {[pred] * len(refs)},\t"
                f"references: {refs}\t"
                f"score: {res[res_key]}"
            )
            """
            scores = res[res_key]
            if update_cache:
                score_cache[row.name].append(scores)
                
        if max(scores) > match_threshold:
            output['precise'] += 1
        
        for idx, score in enumerate(scores):
            if score > match_threshold:
                output['covered'].add(idx)
    
    output['covered'] = len(output['covered'])
    return pd.Series(output)


def unordered_score(
    df, pred_column, ref_column, 
    score_fn, match_threshold, res_key="scores",
    score_cache=None, update_cache=False, cache_file=None,
    score_args={}
):
    
    if score_cache is None:
        score_cache = defaultdict(list)
    
    if not update_cache and cache_file is not None:
        with open(cache_file) as f:
            from_file = json.load(f)
            print(f"read score cache from {cache_file}")
        for id, scores in from_file.items():
            score_cache[int(id)] = scores

    def helper(r):
        return unordered_row_score(
            r, 
            pred_column, 
            ref_column, 
            score_fn, 
            match_threshold,
            res_key=res_key,
            score_cache=score_cache, 
            update_cache=update_cache,
            score_args=score_args
        )
    res = df.apply(helper, axis=1)
    
    if update_cache and cache_file is not None:
        with open(cache_file, 'w') as f:
            json.dump(score_cache, f, indent=2)
            
    return res, score_cache

# -------------------------------
# ordered score

def lcs_row_score(row, raw_scores, threshold):
    if isinstance(raw_scores, str):
        with open(raw_scores) as f:
            raw_scores = json.load(f)
        
    # for each predicted idea, align against gold idea
    matches = []
    row_scores = raw_scores[row.name]
    num_ref = len(row_scores[0])
    
    for pred_scores in row_scores:
        matches.append([
            idx for idx, score in enumerate(pred_scores) 
            if score > threshold 
        ])
        
    ref_idxs = list(range(num_ref))
    longest_correct_length = max(list(map(
        lambda sequence: lcs(sequence, ref_idxs),
        expand_sequences(matches)
    )))
    
    return pd.Series({
        TRUE_POSITIVES_COLUMN: longest_correct_length,
        POSITIVE_PREDICTIONS_COLUMN: len(row_scores),
        RELEVANT_COLUMN: num_ref 
    })


def lcs_score(df, raw_scores, threshold):
    return df.apply(lambda r: lcs_row_score(r, raw_scores, threshold), axis=1)


# -------------------------------
# compute F1 for ordered/unordered

def f1_score(eval_df):
    eval_sum = eval_df.sum()
    precision = (
        eval_sum.get(PRECISE_COLUMN, eval_sum.get(TRUE_POSITIVES_COLUMN, 0)) 
        / eval_sum[POSITIVE_PREDICTIONS_COLUMN]
    )
    recall = (
        eval_sum.get(COVERED_COLUMN, eval_sum.get(TRUE_POSITIVES_COLUMN, 0)) 
        / eval_sum[RELEVANT_COLUMN]
    )
    return {
        'precision': precision,
        'recall': recall,
        'f1': 2 * precision * recall / (precision + recall)
    }



# -------------------------------
# main routine

if __name__ == "__main__":
    psr = argparse.ArgumentParser()
    psr.add_argument("reference", help="csv path")
    psr.add_argument("prediction", help="csv path")
    psr.add_argument("-de", "--device", help="e.g. 'cuda:0'")
    psr.add_argument("-sc", "--score_cache", help="json path")
    psr.add_argument("-wc", "--write_cache", action='store_true', help="boolean flag")
    psr.add_argument("-th", "--threshold", type=float, default=MATCH_THRESHOLD, help="match iff s > threshold")
    psr.add_argument("-pc", "--pred_col", default="explanation", help="column name of prediction")
    psr.add_argument("-su", "--summary", help="json path")
    psr.add_argument("-dt", "--details", help="csv path")
    psr.add_argument("-va", "--vanilla", action="store_true", help="boolean flag for running vanilla bertscore")
    
    args = psr.parse_args()

    start = perf_counter()
    df = pd.read_csv(args.reference, index_col="id")
    pred_df = pd.read_csv(args.prediction, index_col="id")
    df["prediction"] = pred_df[args.pred_col]
    
    # filter out NaN
    df = df[~df["prediction"].isna()]

    unordered_eval, score_cache = unordered_score(
        df, 
        "prediction", 
        "explanation", 
        bert, # bleurt
        args.threshold, 
        res_key="f1", # "scores"
        update_cache=args.write_cache, 
        cache_file=args.score_cache,
        score_args={
            "device": args.device,
            "model_type": BERTSCORE_MODEL
        }
    )
    ordered_eval = lcs_score(df, score_cache, args.threshold)

    
    res = {
        "info": {
            "similarity_metric": "bertscore",
            "model_type": BERTSCORE_MODEL,
            "match_threshold": args.threshold
        },
        "unordered": f1_score(unordered_eval),
        "ordered": f1_score(ordered_eval)
    }

    if args.vanilla:
        vanilla = plain_bertscore(df, "prediction", "explanation")
        res["vanilla_bertscore_mean"] = np.mean(vanilla['f1']),
    
    pretty_print(res)
    if args.summary is not None:
        with open(args.summary, 'w') as f:
            json.dump(res, f, indent=2)

    if args.details is not None:
        # consolidate to unordered_eval and save to file
        unordered_eval[TRUE_POSITIVES_COLUMN] = ordered_eval[TRUE_POSITIVES_COLUMN]
        unordered_eval.to_csv(args.details)

    print(f"\n\ncompleted scoring in {perf_counter() - start}s")
