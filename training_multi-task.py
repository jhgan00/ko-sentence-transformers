"""
The system trains KoBERT on the KorNLI dataset with MultipleNegativesRankingLoss.\
Entailments are positive pairs and the contradiction on KorNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset
Usage:
python training_nli_v2.py
"""
import csv
import glob
import logging
import math
import os
import random
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import DataLoader

from ko_sentence_transformers.models import KoBertTransformer
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample
from sentence_transformers import models, losses, datasets
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

seed = 777
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)

model_name = "monologg/kobert"
train_nli_batch_size = 64
train_sts_batch_size = 8
max_seq_length = 75
num_epochs = 6
model_save_path = 'output/training_nli_v2_' + model_name.replace("/", "-") + '-' + datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S")

# Here we define our SentenceTransformer model
word_embedding_model = KoBertTransformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

nli_dataset_path = 'KorNLUDatasets/KorNLI'
sts_dataset_path = 'KorNLUDatasets/KorSTS'

# Read the AllNLI.tsv.gz file and create the training dataset
logging.info("Read KorNLI train dataset")


train_sts_samples = []
train_file = os.path.join(sts_dataset_path, "sts-train.tsv")
with open(train_file, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        train_sts_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
train_sts_dataloader = DataLoader(train_sts_samples, shuffle=True, batch_size=train_sts_batch_size)
train_sts_loss = losses.CosineSimilarityLoss(model=model)

def add_to_samples(sent1, sent2, label):
    if sent1 not in train_data:
        train_data[sent1] = {'contradiction': set(), 'entailment': set(), 'neutral': set()}
    train_data[sent1][label].add(sent2)


train_data = {}
train_files = glob.glob(os.path.join(nli_dataset_path, "*train.ko.tsv"))
for train_file in train_files:
    with open(train_file, 'rt', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            sent1 = row['sentence1'].strip()
            sent2 = row['sentence2'].strip()

            add_to_samples(sent1, sent2, row['gold_label'])
            add_to_samples(sent2, sent1, row['gold_label'])  # Also add the opposite

train_nli_samples = []
for sent1, others in train_data.items():
    if len(others['entailment']) > 0 and len(others['contradiction']) > 0:
        train_nli_samples.append(InputExample(
            texts=[sent1, random.choice(list(others['entailment'])), random.choice(list(others['contradiction']))]))
        train_nli_samples.append(InputExample(
            texts=[random.choice(list(others['entailment'])), sent1, random.choice(list(others['contradiction']))]))

logging.info("Train samples: {}".format(len(train_nli_samples)))
train_nli_dataloader = datasets.NoDuplicatesDataLoader(train_nli_samples, batch_size=train_nli_batch_size)
train_nli_loss = losses.MultipleNegativesRankingLoss(model)

# Read STSbenchmark dataset and use it as development set
logging.info("Read KorSTS benchmark dev dataset")
dev_samples = []
dev_file = os.path.join(sts_dataset_path, "sts-dev.tsv")
with open(dev_file, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_sts_batch_size,
                                                                 name='sts-dev')

# Configure the training
warmup_steps = math.ceil(len(train_sts_dataloader) * num_epochs * 0.1)
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_nli_dataloader, train_nli_loss), (train_sts_dataloader, train_sts_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=int((len(train_nli_dataloader) + len(train_sts_dataloader)) * 0.1),
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=False  # Set to True, if your GPU supports FP16 operations
          )

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

test_samples = []
test_file = os.path.join(sts_dataset_path, "sts-test.tsv")
with open(test_file, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0
        test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))

model = SentenceTransformer(model_save_path)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_sts_batch_size,
                                                                  name='sts-test')
test_evaluator(model, output_path=model_save_path)
