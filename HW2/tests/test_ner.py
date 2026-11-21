import ast
import io
import json
from pathlib import Path
import random
import re
import tokenize

import nbformat
import numpy as np
import pytest
import torch
from datasets import Dataset, DatasetDict
from seqeval.metrics.sequence_labeling import get_entities
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (BertConfig, BertForTokenClassification, DebertaV2Config,
                          DebertaV2ForTokenClassification, DistilBertConfig,
                          DistilBertForTokenClassification, PreTrainedTokenizerFast,
                          RobertaConfig, RobertaForTokenClassification)

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_contract():
    notebook = nbformat.read(ROOT / 'hw2_nlp.ipynb', 4)
    manifest = json.loads((ROOT / 'tests/notebook_contract.json').read_text(encoding='utf-8'))
    nbformat.validate(notebook)
    for cell, expected in zip(notebook.cells, manifest['original_cells']):
        assert cell.id == expected['id']
        assert all(cell.metadata.get(key) == value for key, value in expected['metadata'].items())
    for cell in notebook.cells:
        if cell.id in manifest['assertions']:
            tree = ast.parse(cell.source)
            actual = [ast.dump(node) for node in ast.walk(tree) if isinstance(node, ast.Assert)]
            for required in manifest['assertions'][cell.id]:
                assert required in actual
    functions = {}
    for cell in notebook.cells:
        if cell.cell_type == 'code' and not any(line.lstrip().startswith('%') for line in cell.source.splitlines()):
            functions.update({node.name: ast.dump(node.args) for node in ast.walk(ast.parse(cell.source))
                              if isinstance(node, ast.FunctionDef)})
    for name, signature in manifest.get('function_signatures', {}).items():
        assert functions[name] == signature


def test_comment_style():
    notebook = nbformat.read(ROOT / 'hw2_nlp.ipynb', 4)
    sources = [cell.source for cell in notebook.cells if cell.cell_type == 'code']
    sources.extend(path.read_text(encoding='utf-8') for path in ROOT.glob('*.py'))
    sources.extend(path.read_text(encoding='utf-8') for path in (ROOT / 'tests').glob('*.py'))
    for source in sources:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.COMMENT:
                text = token.string[1:].strip()
                assert not text or re.fullmatch('[а-яё ]+', text)


def test_original_bio_assertions(ns):
    notebook = nbformat.read(ROOT / 'hw2_nlp.ipynb', 4)
    exec(notebook.cells[28].source, ns)


def test_bio_matches_seqeval(ns):
    rng = random.Random(45)
    alphabet = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-art-film', 'I-art-film']
    for _ in range(500):
        tags = [rng.choice(alphabet) for _ in range(rng.randrange(30))]
        expected = [(start, end + 1, kind) for kind, start, end in get_entities(tags)]
        assert ns['bio_to_spans'](tags) == expected
        canonical = ns['spans_to_bio'](expected, len(tags))
        assert ns['bio_to_spans'](canonical) == expected


@pytest.mark.parametrize('spans,n', [([(0, 2, 'PER'), (1, 3, 'LOC')], 4),
                                    ([(-1, 2, 'PER')], 3), ([(0, 4, 'PER')], 3),
                                    ([(1, 1, 'PER')], 3), ([(0, 1, '')], 3)])
def test_invalid_spans_are_rejected(ns, spans, n):
    with pytest.raises(ValueError):
        ns['spans_to_bio'](spans, n)


def test_alignment_special_tokens_and_repeated_ids(ns):
    labels = [1, 0, 5]
    ids = [None, 0, 0, None, 0, 1, 2, 2, None]
    assert ns['align_labels'](labels, ids) == [-100, 1, -100, -100, -100, 0, 5, -100, -100]
    assert ns['align_labels'](labels, ids, {1: 2, 5: 6}) == [-100, 1, 2, -100, 2, 0, 5, 6, -100]
    assert ns['align_labels']([], [None, None]) == [-100, -100]
    with pytest.raises(ValueError):
        ns['align_labels']([0], [1])


def test_majority_unseen_words_and_ties(ns):
    tagger = ns['fit_majority_tagger']([['Анна'], ['Анна'], ['Анна'], ['Москва']],
                                      [['B-PER'], ['O'], ['B-PER'], ['B-LOC']])
    assert tagger == {'Анна': 'B-PER', 'Москва': 'B-LOC'}
    assert ns['predict_majority'](tagger, ['Анна', 'анна', 'неизвестно']) == ['B-PER', 'O', 'O']
    tie = ns['fit_majority_tagger']([['имя'], ['имя']], [['B-ORG'], ['B-PER']])
    assert tie['имя'] == 'B-ORG'
    with pytest.raises(ValueError):
        ns['fit_majority_tagger']([['a', 'b']], [['O']])


def test_token_f1_ignores_background(ns):
    assert ns['token_f1']([['O'] * 99 + ['B-PER']], [['O'] * 100]) == 0
    assert ns['token_f1']([['B-PER', 'I-PER', 'O', 'B-LOC']], [['B-PER', 'O', 'O', 'B-LOC']]) == 0.8
    assert ns['token_f1']([['B-PER']], [['B-LOC']]) == 0
    assert ns['token_f1']([], []) == 0
    with pytest.raises(ValueError):
        ns['token_f1']([['O']], [[]])


def test_error_priority_and_nonexclusive_overlap(ns):
    gold = [(0, 2, 'PER'), (4, 5, 'LOC'), (7, 9, 'ORG'), (11, 12, 'PER')]
    predicted = [(0, 2, 'PER'), (4, 6, 'LOC'), (7, 9, 'LOC'), (13, 14, 'ORG')]
    assert [row[0] for row in ns['categorize_errors'](gold, predicted)] == ['correct','boundary','type','missed','spurious']
    assert [row[0] for row in ns['categorize_errors']([(0, 3, 'LOC')], [(1, 2, 'PER'), (2, 3, 'LOC')])] == ['boundary']
    assert [row[0] for row in ns['categorize_errors']([(0, 2, 'LOC'), (2, 4, 'LOC')], [(0, 4, 'LOC')])] == ['boundary','boundary']
    assert [row[0] for row in ns['categorize_errors']([(0, 2, 'PER')], [(2, 3, 'PER')])] == ['missed','spurious']
    assert ns['categorize_errors']([(0, 2, 'PER')], [(0, 3, 'PER'), (0, 2, 'PER')])[0][0] == 'correct'


@pytest.mark.parametrize('kind', ['bert','roberta','distilbert','deberta'])
def test_parameter_counts(ns, kind):
    common = dict(vocab_size=31, hidden_size=24, num_hidden_layers=2, num_attention_heads=4,
                  intermediate_size=48, max_position_embeddings=32, num_labels=5)
    if kind == 'bert':
        model = BertForTokenClassification(BertConfig(**common))
    elif kind == 'roberta':
        model = RobertaForTokenClassification(RobertaConfig(**common))
    elif kind == 'distilbert':
        model = DistilBertForTokenClassification(DistilBertConfig(vocab_size=31, dim=24, hidden_dim=48,
                                                                 n_layers=2, n_heads=4, num_labels=5))
    else:
        model = DebertaV2ForTokenClassification(DebertaV2Config(**common, relative_attention=True,
                                                               position_buckets=8, norm_rel_ebd='layer_norm'))
    blocks = ns['params_by_block'](model)
    assert sum(blocks[key] for key in ['embeddings','encoder','head','other']) == sum(p.numel() for p in model.parameters())
    assert blocks['head'] == 24 * 5 + 5
    assert len(blocks['layers']) == 2 and sum(blocks['layers']) == blocks['encoder']
    assert blocks['embeddings'] >= 31 * 24
    assert blocks['other'] > 0 if kind == 'deberta' else blocks['other'] == 0


def test_roundtrip_json_normalizes_orphan_tags(ns, tmp_path):
    source = Dataset.from_list([dict(tokens=['Анна','в','Москве'], ner_tags=[2,0,5])])
    path = tmp_path / 'spans.jsonl'
    assert ns['write_span_jsonl'](source, path, ns['LABEL_NAMES']) == 1
    restored = ns['read_span_jsonl'](path, ns['LABEL_NAMES'])
    assert restored[0]['tokens'] == source[0]['tokens']
    assert restored[0]['ner_tags'] == [1,0,5]


def test_split_deduplication_prevents_leakage(ns):
    ds = DatasetDict(
        train=Dataset.from_list([dict(tokens=['Анна'],ner_tags=[1]),dict(tokens=['Анна'],ner_tags=[1])]),
        validation=Dataset.from_list([dict(tokens=['Анна'],ner_tags=[3]),dict(tokens=['Москва'],ner_tags=[5])]),
        test=Dataset.from_list([dict(tokens=['Москва'],ner_tags=[5]),dict(tokens=['река'],ner_tags=[0])]))
    cleaned, audit = ns['deduplicate_splits'](ds)
    assert [len(cleaned[key]) for key in ['train','validation','test']] == [1,1,1]
    assert audit['validation']['conflicting_duplicates'] == 1
    assert cleaned['test'][0]['tokens'] == ['река']


def test_corruptions_keep_span_coordinates(ns):
    words = [['Анна','приехала','в','Петербург'],['Москва']]
    for kind in ['lower','upper','typos']:
        result = ns['corrupt_tokens'](words,kind,seed=4)
        assert list(map(len,result)) == list(map(len,words))
        assert result == ns['corrupt_tokens'](words,kind,seed=4)
    assert words[0][0] == 'Анна'
    assert ns['corrupt_tokens'](words,'lower')[0][0] == 'анна'


def test_confidence_filter_only_removes_whole_spans(ns):
    tags = [['B-PER','I-PER','O','B-LOC']]
    probs = [[0.9,0.3,0.99,0.7]]
    mean = ns['filter_by_confidence'](tags,probs,0.5,'mean')
    minimum = ns['filter_by_confidence'](tags,probs,0.5,'min')
    assert mean == tags and minimum == [['O','O','O','B-LOC']]
    previous = set(ns['bio_to_spans'](tags[0]))
    for threshold in np.linspace(0,1,21):
        current = set(ns['bio_to_spans'](ns['filter_by_confidence'](tags,probs,float(threshold))[0]))
        assert current <= previous
        previous = current


def tiny_tokenizer():
    backend = Tokenizer(WordLevel({'[PAD]':0,'[UNK]':1,'[CLS]':2,'[SEP]':3,'Анна':4,'Москва':5}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]', pad_token='[PAD]',
                                  cls_token='[CLS]', sep_token='[SEP]')


@pytest.mark.parametrize('freeze', [False,True])
def test_training_loop_and_freezing(ns, freeze):
    torch.manual_seed(0)
    model = BertForTokenClassification(BertConfig(vocab_size=6,hidden_size=16,num_hidden_layers=1,
                                                   num_attention_heads=2,intermediate_size=32,num_labels=3,
                                                   hidden_dropout_prob=0,attention_probs_dropout_prob=0))
    ds = Dataset.from_list([dict(input_ids=[2,4,5,3],attention_mask=[1,1,1,1],labels=[-100,1,2,-100]) for _ in range(16)])
    initial = {name:p.detach().clone() for name,p in model.named_parameters()}
    history,seconds = ns['train_ner'](model,ds,ds,epochs=2,batch_size=4,freeze_encoder=freeze,tok=tiny_tokenizer())
    assert seconds > 0 and sum('eval_loss' in row for row in history) >= 2
    assert all(np.isfinite(row['eval_loss']) for row in history if 'eval_loss' in row)
    changed = {name for name,p in model.named_parameters() if not torch.equal(p,initial[name])}
    assert any(name.startswith('classifier.') for name in changed)
    assert not any(name.startswith('bert.') for name in changed) if freeze else any(name.startswith('bert.') for name in changed)
    assert not model.training and all(p.grad is None for p in model.parameters())


def test_onnx_dynamic_shapes_and_quantization(ns, tmp_path):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    torch.manual_seed(0)
    config = BertConfig(vocab_size=10,hidden_size=16,num_hidden_layers=1,num_attention_heads=2,
                        intermediate_size=32,num_labels=3,hidden_dropout_prob=0,attention_probs_dropout_prob=0)
    model = BertForTokenClassification(config).eval()
    sample = dict(input_ids=torch.tensor([[2,4,5,3]]),attention_mask=torch.ones(1,4,dtype=torch.long))
    path = ns['export_onnx'](model,tmp_path/'model.onnx',sample)
    adapter = ns['ORTTokenClassifier'](path,config)
    batch = dict(input_ids=torch.tensor([[2,4,3],[2,5,0]]),attention_mask=torch.tensor([[1,1,1],[1,1,0]]))
    with torch.no_grad():
        expected = model(**batch).logits
    assert torch.allclose(adapter(**batch).logits,expected,atol=1e-5,rtol=1e-4)
    quantized = tmp_path/'int8.onnx'
    quantize_dynamic(str(path),str(quantized),weight_type=QuantType.QInt8,op_types_to_quantize=['MatMul','Gemm'])
    result = ns['ORTTokenClassifier'](quantized,config)(**batch).logits
    assert result.shape == expected.shape and torch.isfinite(result).all()
