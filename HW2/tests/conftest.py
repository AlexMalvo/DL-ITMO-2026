import ast
import collections
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import nbformat
import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from transformers import DataCollatorForTokenClassification, Trainer, TrainingArguments

ROOT = Path(__file__).resolve().parents[1]


def definitions(nodes):
    result = []
    for node in nodes:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            result.append(node)
        elif isinstance(node, ast.If):
            result.extend(definitions(node.body))
    return result


@pytest.fixture
def ns(tmp_path):
    torch.set_num_threads(2)
    notebook = nbformat.read(ROOT / 'hw2_nlp.ipynb', 4)
    scope = dict(torch=torch, np=np, collections=collections, copy=copy, json=json,
                 hashlib=hashlib, math=math, random=random, time=time, Path=Path,
                 Dataset=Dataset, DatasetDict=DatasetDict,
                 classification_report=classification_report, f1_score=f1_score,
                 precision_score=precision_score, recall_score=recall_score,
                 TrainingArguments=TrainingArguments, Trainer=Trainer,
                 DataCollatorForTokenClassification=DataCollatorForTokenClassification,
                 EPOCHS=1, LR=2e-4, BATCH_SIZE=8, SEED=0,
                 device=torch.device('cpu'), WORK_DIR=tmp_path,
                 tokenizer=None, MAX_LENGTH=32,
                 LABEL_NAMES=['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC'])
    for index in [14, 22, 27, 29, 34, 35, 40, 45, 50, 71, 75, 79]:
        tree = ast.parse(notebook.cells[index].source)
        module = ast.Module(body=definitions(tree.body), type_ignores=[])
        exec(compile(module, f'notebook_cell_{index}', 'exec'), scope)
    return scope
