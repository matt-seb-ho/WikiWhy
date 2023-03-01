import time
import random
import json
import csv
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from icecream import ic

import torch
from torch.utils.data import (
    Dataset, DataLoader, TensorDataset, 
    RandomSampler, random_split
)

import evaluate
from datasets import load_metric

# import classes + consts + helper methods from train script
from tune_gpt import (
    CausalQA, ExpGen, tokenizer, unpack_exp,
    build_model_input, build_model_input_from_row,
    SPECIAL_TOKENS, ATTR_TO_SPECIAL_TOKEN
)

# imports for sentence mover distance
from typing import List, Union, Iterable
from itertools import zip_longest
from moverscore_v2 import word_mover_score
from sentence_mover.sentence_mover import SentenceMoverSim

# ----------- Load Metrics -------------

# huggingface datasets module::
bertscore = load_metric("bertscore")
# bleu = load_metric('bleu')
# sbleu = load_metric('sacrebleu')

# huggingface evaluate module:
bleu = evaluate.load("bleu")
sbleu = evaluate.load("sacrebleu")
bleurt = evaluate.load("bleurt", module_type="metric")
rouge = evaluate.load("rouge")
# bertscore = evaluate.load('bertscore')

# ---------- Consolidating to DataFrame ----------
# convert file to dict mapping hit_id => answer
def build_id_map(fname, delimiter=','):
    names = ["id", "hypothesis"] if delimiter == '\t' else None
    temp = pd.read_csv(fname, delimiter=delimiter, names=names) 
    hyp_col = (temp.columns if names is None else names)[1]

    res = {}
    for _, r in temp.iterrows():
        res[r['id']] = r[hyp_col]
    return res

def add_id_map(df, id_map, col_name, rm_empty=True):
    def add_helper(r):
        eid = r['id']
        return id_map[eid] if eid in id_map else ''
    
    df[col_name] = df.apply(add_helper, axis=1)
    view = df.loc[df[col_name] != ''] if rm_empty else df
    return view.copy()

# ---------- DataFrame Metrics Wrappers ----------

# wrapper functions (operate on dataframe columns)
def get_bleu_score(df, pred_col, ref_col, bleu=bleu):
    preds = df[pred_col].to_list()
    refs = df[ref_col].to_list()
    refs = [[ref] for ref in refs]
    return bleu.compute(predictions=preds, references=refs)

def get_rouge_score(df, pred_col, ref_col):
    return rouge.compute(
        predictions=df[pred_col].to_list(),
        references=df[ref_col].to_list()
    )

def get_bert_score(df, pred_col, ref_col, device):
    preds = df[pred_col].to_list()
    refs = df[ref_col].to_list()
    # bertsc = bertscore.compute(predictions=preds, references=refs, lang='en', device=device)
    bertsc = bertscore.compute(
        predictions=preds, 
        references=refs, 
        model_type="distilbert-base-uncased",
        lang='en',
        device=device
    )
    bertsc["mean_precision"] = np.mean(bertsc["precision"])
    return bertsc

def get_bleurt_score(df, pred_col, ref_col):
    preds = df[pred_col].to_list()
    refs = df[ref_col].to_list()
    
    bleurtsc = bleurt.compute(predictions=preds, references=refs)
    return bleurtsc

def break_on_sentence(txt):
    broken = txt.split(". ")
    return [sentence + '.' for sentence in broken]

def wm_score(hypothesis: str, references: List[str], trace=0):
    idf_dict_hyp = defaultdict(lambda: 1.)
    idf_dict_ref = defaultdict(lambda: 1.)
    hypothesis = [hypothesis] * len(references)
    
    scores = word_mover_score(
        references, hypothesis, idf_dict_ref, idf_dict_hyp, 
        stop_words=[], n_gram=1, remove_subwords=False
    )    
    sentence_score = np.mean(scores)
    
    if trace > 0:
        print(hypothesis, references, sentence_score)
            
    return sentence_score

def get_wm_score(df, pred_col, ref_col):
    scores = []
    for i, r in tqdm(df.iterrows(), total=len(df)):
        scores.append(wm_score(r[pred_col], [r[ref_col]]))
    return scores

def get_sm_score(df, pred_col, ref_col, sms, batch_size):
    # sms takes lists of lists as inputs 
    # (each examples has list of hyp sentences and a list of ref sentences)
    preds, refs = [], []
    for i, r in tqdm(df.iterrows(), total=len(df)):
        preds.append(break_on_sentence(r[pred_col]))
        refs.append(break_on_sentence(r[ref_col]))
    return sms.batch_compute(refs, preds, batch_size)

# ---------- All in 1 -------------------
def get_scores(df, hyp_col, ref_col, dev_num=0, sms=None, sms_bs=32):

    device = torch.device(f"cuda:{dev_num}")
    scores = {
        "bleu": get_bleu_score(df, hyp_col, ref_col),
        "rouge": get_rouge_score(df, hyp_col, ref_col),
        "bert": get_bert_score(df, hyp_col, ref_col, device),
        "wm": get_wm_score(df, hyp_col, ref_col),
    }

    if sms is not None:
        scores["sm"] = get_sm_score(df, hyp_col, ref_col, sms, sms_bs)
    
    return scores

# ---------- Displaying Scores ----------

def pretty_print(dict_in):
    print(json.dumps(dict_in, indent=2))

def display_scores(score, label=None):
    if label is not None:
        print(f"//////// {label} /////////")
    pretty_print(score["bleu"])
    print("bert f1 avg:", np.mean(score["bert"]["f1"]))
    print("wm avg:", np.mean(score["wm"]), end='\n\n')

def write_score_summary(scores, fname):
    sc = scores.copy()
    sc["bert_f1_avg"] = np.mean(sc.pop("bert")["f1"])
    sc["wm_avg"] = np.mean(sc.pop("wm"))
    sc["sm_avg"] = np.mean(sc.pop("sm"))

    with open(fname, 'w') as f:
        json.dump(sc, f, indent=2)
        print(f"Wrote to {fname}")

# --------- Misc ---------------
def add_string_exp_column(df, colname="exp_str"):
    df[colname] = df.apply(
        lambda r: unpack_exp(r['explanation']), axis=1
    )
