"""
fine-tune gpt-2
- WikiWhy dataset subclasses
- pytorch lightning training module

"""
import json
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm, trange

import torch
import torch.nn.functional as F
from torch.utils.data import (
    Dataset, DataLoader, 
    RandomSampler, random_split
)

from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    AdamW, get_linear_schedule_with_warmup
)

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# -------------------------------------
# helper functions for data preparation
# -------------------------------------

# prepared tokenized input for model
def build_qa_input(question, answer, tokenizer, special_tokens, max_len=None, add_eos=True):
    """ build training inputs from raw text  """
    # tokenize text as needed and get special token ids
    if isinstance(question, str):
        question, answer = map(tokenizer.encode, (question, answer))
    bos, eos, qst, ans, pad = tokenizer.convert_tokens_to_ids(special_tokens)

    # build inputs
    instance = {}    
    iids = (
        [bos, qst] + question + [ans] + answer + ([eos] if add_eos else [])
    )
    tids = (
        [qst] * (2 + len(question)) + [ans] * (1 + len(answer) + int(add_eos)) 
    )
    
    # padding + attention mask
    if max_len is not None:
        pad_len = max_len - len(iids)
        instance['attention_mask'] = [1] * len(iids) + [0] * pad_len
        iids += [pad] * pad_len
        tids += [pad] * pad_len
    
    instance['input_ids'] = iids
    instance['type_ids'] = tids

    # only compute loss for explanation portion
    instance['labels'] = [iid if tid == ans else -100 for iid, tid in zip(iids, tids)]

    return instance

# prepared tokenized input for model
def build_exp_input(cause, effect, explanation, tokenizer, special_tokens, max_len=None, add_eos=True):
    """ build training inputs from raw text  """
    # tokenize text as needed and get special token ids
    if isinstance(cause, str):
        cause, effect, explanation = map(tokenizer.encode, (cause, effect, explanation))
    bos, eos, cse, eff, exp, pad = tokenizer.convert_tokens_to_ids(special_tokens)

    # build inputs
    instance = {}    
    iids = (
        [bos, cse] + cause + [eff] + effect 
        + [exp] + explanation + ([eos] if add_eos else [])
    )
    tids = (
        [cse] * (2 + len(cause)) + [eff] * (1 + len(effect)) 
        + [exp] * (1 + len(explanation) + int(add_eos))
    )
    
    # padding + attention mask
    if max_len is not None:
        pad_len = max_len - len(iids)
        instance['attention_mask'] = [1] * len(iids) + [0] * pad_len
        iids += [pad] * pad_len
        tids += [pad] * pad_len
    
    instance['input_ids'] = iids
    instance['type_ids'] = tids

    # only compute loss for explanation portion
    instance['labels'] = [iid if tid == exp else -100 for iid, tid in zip(iids, tids)]

    return instance

def unpack_exp(json_str):
    exp_steps = json.loads(json_str)
    explanation = ''
    for step in exp_steps:
        explanation += step.strip(' .,') + '. '
    explanation = explanation[:-1] # remove trailing space
    return explanation

# ----------------
# Dataset Subclass
# ----------------
class WikiWhy(Dataset):
    def __init__(self, df, tokenizer, special_tokens, max_len):
        self.examples = []
        for i, row in df.iterrows():
            instance = self.build_input_from_row(
                row, tokenizer, special_tokens, max_len
            )
            tensorized = {key: torch.tensor(val) for key, val in instance.items()}
            self.examples.append(tensorized)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]
    
    @staticmethod
    def build_input_from_row(row, tokenizer, special_tokens, max_len):
        raise NotImplementedError()

class WikiWhyQA(WikiWhy):
    @staticmethod
    def build_input_from_row(r, tokenizer, special_tokens, max_len=None):
        question, answer = r[['question', 'cause']]
        return build_qa_input(
            question, 
            answer, 
            tokenizer, 
            special_tokens, 
            max_len=max_len
        )

class WikiWhyExplain(WikiWhy):
    @staticmethod
    def build_input_from_row(r, tokenizer, special_tokens, max_len=None):
        cause, effect, explanation = r[['cause', 'effect', 'explanation']]
        return build_exp_input(
            cause, 
            effect, 
            explanation, 
            tokenizer, 
            special_tokens, 
            max_len=max_len
        )

# setup heper routine
def find_max_len(data, tokenizer, special_tokens, task):
    build_input = (
        WikiWhyQA if task == "qa" 
        else WikiWhyExplain
    ).build_input_from_row

    def len_helper(row):
        return len(build_input(row, tokenizer, special_tokens)['input_ids'])
                
    input_lengths = data.apply(len_helper, axis=1)
    return input_lengths.max()

# ----------------
# Lightning Module
# ----------------

class WikiWhyGpt2(pl.LightningModule):
    # class constants
    SPECIAL_TOKENS = []
    ATTR_TO_SPECIAL_TOKEN = {}

    def __init__(self, hyperparams={}):
        super().__init__()

        self.tokenizer = GPT2Tokenizer.from_pretrained(
            'gpt2', 
            **self.ATTR_TO_SPECIAL_TOKEN
        )
        self.model = GPT2LMHeadModel.from_pretrained('gpt2')
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.hyperparams = hyperparams

        self.save_hyperparameters("hyperparams")

    def forward(self, **kwargs):
        return self.model(**kwargs)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.hyperparams)
        return optimizer
    
    def process_batch(self, batch):
        input_ids = batch['input_ids'].to(self.device)
        type_ids = batch['type_ids'].to(self.device)
        masks = batch['attention_mask'].to(self.device)
        labels = batch['labels'].to(self.device)
        return self.model(
            input_ids, 
            attention_mask=masks, 
            token_type_ids=type_ids,
            labels=labels
        )
        
    def training_step(self, batch, batch_idx):
        outputs = self.process_batch(batch)
        loss = outputs[0]
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self.process_batch(batch)
        loss = outputs[0]
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        outputs = self.process_batch(batch)
        loss = outputs[0]
        self.log('test_loss', loss)

class AnswerModel(WikiWhyGpt2):
    # class constants
    SPECIAL_TOKENS = ["<bos>", "<eos>", "<question>", "<answer>", "<pad>"]
    ATTR_TO_SPECIAL_TOKEN = {
        'bos_token': '<bos>', 
        'eos_token': '<eos>', 
        'pad_token': '<pad>',
        'additional_special_tokens': ['<question>', '<answer>']
    }

class ExplainerModel(WikiWhyGpt2):
    SPECIAL_TOKENS = ["<bos>", "<eos>", "<cause>", "<effect>", "<explanation>", "<pad>"]
    ATTR_TO_SPECIAL_TOKEN = {
        'bos_token': '<bos>', 
        'eos_token': '<eos>', 
        'pad_token': '<pad>',
        'additional_special_tokens': ['<cause>', '<effect>', '<explanation>']
    }

def find_exp(number):
    base10 = np.log10(abs(number))
    return abs(np.floor(base10))

if __name__ == "__main__":

    psr = argparse.ArgumentParser()
    psr.add_argument("--task", choices=["qa", "exp"], required=True)
    psr.add_argument("--lr")
    psr.add_argument("--epochs", type=int, default=8)
    psr.add_argument("--batch_size", type=int, default=32)
    psr.add_argument("--checkpoint", default=".")
    psr.add_argument("--devices", type=int, nargs="+", required=True)
    args = psr.parse_args()

    assert args.task in ["qa", "exp"]

    lr_part = "" if args.lr is None else f"_{find_exp(args.lr)}lre"
    args.checkpoint = (
        f"{args.checkpoint}"
        f"{'/' if args.checkpoint[-1] != '/' else ''}"
        f"gpt2_{args.task}{lr_part}_{args.epochs}e.ckpt"
    )

    # ----------------
    # initalize model
    # ----------------

    hyperparams = (
        {} if args.lr is None else 
        {"learning_rate": args.lr}
    )
    
    model = (AnswerModel if args.task == "qa" else ExplainerModel)(hyperparams)

    # ----------------
    # read data from file
    # ----------------

    # load data from file
    data_file_path = "../data/dataset.csv"
    df = pd.read_csv(data_file_path, delimiter=',', index_col="id")

    print(f"\nDataset Summary\n - total_size: {len(df)}")
    print(" - columns:", df.columns.to_list())

    # max len
    # MAX_EXAMPLE_LEN = find_max_len(
    #     df, model.tokenizer, model.SPECIAL_TOKENS, args.task
    # )
    # precomputed (from previous runs)
    MAX_EXAMPLE_LEN = 90 if task == "qa" else 222

    # report hp
    print("Using hyperparameters...")
    print(" - batch size:", args.batch_size)
    print(" - learning rate:", args.lr)
    print(" - max epochs:", args.epochs)
    print(" - max example length:", MAX_EXAMPLE_LEN)
    print(" - checkpoint file:", args.checkpoint)


    # --------------------------
    # prep dataloaders
    # --------------------------


    # prep train, test, validation splits

    # # original split code (really turned the brain off for this one)
    # testdev = df[df["exp_checked"]]
    # train = df[~df["exp_checked"]]
    # test = testdev.sample(frac=0.5, random_state=0)
    # dev = testdev[~testdev.index.isin(test.index)]

    train = df.loc[df["split"] == "train"]
    test = df.loc[df["split"] == "test"]
    dev = df.loc[df["split"] == "dev"]
    print("Splits")
    print(f"- train: {len(train)}\n - test: {len(test)}\n - dev: {len(dev)}")

    # initialize dataset objects
    dataset = WikiWhyQA if args.task == "qa" else WikiWhyExplain
    train_dataset = dataset(
        train, model.tokenizer, model.SPECIAL_TOKENS, MAX_EXAMPLE_LEN
    )
    val_dataset = dataset(
        dev, model.tokenizer, model.SPECIAL_TOKENS, MAX_EXAMPLE_LEN
    )

    # dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )

    # ----------------
    # train model
    # ----------------

    # use whichever gpus are available
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=args.devices, 
        strategy='dp', 
        max_epochs=args.epochs
    )
    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(args.checkpoint)
    print(f"Saved checkpoint to {args.checkpoint}")
