"""
This examples trains KoBERT for the KorSTS benchmark from scratch.
It generates sentence embeddings that can be compared using cosine-similarity to measure the similarity.
Usage:
python training_sts.py
"""
import sys
import csv
import logging
import math
import os
from datetime import datetime

from sentence_transformers import SentenceTransformer, LoggingHandler, models, losses
import kobert
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

sts_dataset_path = 'KorNLUDatasets/KorSTS'
model_name = "monologg/kobert"

# Read the dataset
train_batch_size = 16
num_epochs = 1
model_save_path = 'output/training_stsbenchmark_' + model_name.replace("/", "-") + '-' + datetime.now().strftime(
    "%Y-%m-%d_%H-%M-%S")

word_embedding_model = kobert.KoBertTransformer(model_name)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Convert the dataset to a DataLoader ready for training
logging.info("Read KorSTS benchmark train dataset")

train_samples = []
train_file = os.path.join(sts_dataset_path, "sts-train.tsv")
with open(train_file, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        train_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

# Read STSbenchmark dataset and use it as development set
logging.info("Read KorSTS benchmark dev dataset")
dev_samples = []
dev_file = os.path.join(sts_dataset_path, "sts-dev.tsv")
with open(dev_file, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        dev_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size,
                                                                 name='sts-dev')

# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)

# Read STSbenchmark dataset and use it as development set
logging.info("Read KorSTS benchmark test dataset")
test_samples = []
test_file = os.path.join(sts_dataset_path, "sts-test.tsv")
with open(test_file, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)
