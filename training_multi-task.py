"""
This is an example how to train SentenceTransformers in a multi-task setup.
The system trains transfomer models on the KorNLI and on the KorSTS dataset.
"""
import argparse
import glob
import logging
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, models, losses, datasets
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from data_util import load_kor_sts_samples, load_kor_nli_samples

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str)
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--nli_batch_size", type=int, default=64)
parser.add_argument("--sts_batch_size", type=int, default=8)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--output_prefix", type=str, default="kor_multi_")
parser.add_argument("--seed", type=int, default=777)
args = parser.parse_args()

# Fix random seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Configure logger
logging.basicConfig(
    format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO, handlers=[LoggingHandler()]
)

# Read the dataset
model_save_path = os.path.join(
    args.output_dir,
    args.output_prefix + args.model_name_or_path.replace("/", "-") + '-' + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
)

# Define SentenceTransformer model
word_embedding_model = models.Transformer(args.model_name_or_path, max_seq_length=args.max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Read the dataset
nli_dataset_path = 'KorNLUDatasets/KorNLI'
sts_dataset_path = 'KorNLUDatasets/KorSTS'
logging.info("Read KorNLI train/KorSTS dev dataset")

# Read NLI training dataset
nli_train_files = glob.glob(os.path.join(nli_dataset_path, "*train.ko.tsv"))
dev_file = os.path.join(sts_dataset_path, "sts-dev.tsv")
nli_train_samples = []
for nli_train_file in nli_train_files:
    nli_train_samples += load_kor_nli_samples(nli_train_file)
nli_train_dataloader = datasets.NoDuplicatesDataLoader(nli_train_samples, batch_size=args.nli_batch_size)
nli_train_loss = losses.MultipleNegativesRankingLoss(model)

# Read STS training dataset
sts_dataset_path = 'KorNLUDatasets/KorSTS'
sts_train_file = os.path.join(sts_dataset_path, "sts-train.tsv")
sts_train_samples = load_kor_sts_samples(sts_train_file)
sts_train_dataloader = DataLoader(sts_train_samples, shuffle=True, batch_size=args.sts_batch_size)
sts_train_loss = losses.CosineSimilarityLoss(model=model)

# Read STS dev dataset
dev_samples = load_kor_sts_samples(dev_file)
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
    dev_samples, batch_size=args.sts_batch_size, name='sts-dev'
)

# In multi-task training setting,
print("length of nli data loader:", len(nli_train_dataloader))
print("length of sts data loader:", len(sts_train_dataloader))
steps_per_epoch = min(len(nli_train_dataloader), len(sts_train_dataloader))

# Configure the training.
warmup_steps = math.ceil(steps_per_epoch * args.num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
train_objectives = [(nli_train_dataloader, nli_train_loss), (sts_train_dataloader, sts_train_loss)]
model.fit(train_objectives=train_objectives,
          evaluator=dev_evaluator,
          epochs=args.num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)

# Load the stored model and evaluate its performance on STS benchmark dataset
model = SentenceTransformer(model_save_path)
logging.info("Read KorSTS benchmark test dataset")
test_file = os.path.join(sts_dataset_path, "sts-test.tsv")
test_samples = load_kor_sts_samples(test_file)
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)
