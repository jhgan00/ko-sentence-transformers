import csv
import random
from sentence_transformers.readers import InputExample


def load_kor_sts_samples(filename):
    samples = []
    with open(filename, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
            samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
    return samples


def load_kor_nli_samples(filename):

    data = {}

    def add_to_samples(sent1, sent2, label):
        if sent1 not in data:
            data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
        data[sent1][label].add(sent2)

    with open(filename, 'r', encoding='utf-8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            sent1 = row['sentence1'].strip()
            sent2 = row['sentence2'].strip()
            add_to_samples(sent1, sent2, row['gold_label'])
            add_to_samples(sent2, sent1, row['gold_label'])  # Also add the opposite
    samples = []
    for sent1, others in data.items():
        if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
            samples.append(InputExample(
                texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
            samples.append(InputExample(
                texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))
    return samples
