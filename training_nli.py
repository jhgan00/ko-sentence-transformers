"""
The system trains transformers on the KorNLI dataset with MultipleNegativesRankingLoss.
Entailments are positive pairs and the contradiction on KorNLI dataset is added as a hard negative.
At every 10% training steps, the model is evaluated on the STS benchmark dataset
Usage:
python training_nli.py --model_name_or_path klue/bert-base
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
from sentence_transformers import SentenceTransformer, LoggingHandler, models, losses, datasets
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

from data_util import load_kor_sts_samples, load_kor_nli_samples

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str)
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_epochs", type=int, default=1)
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--output_prefix", type=str, default="kor_nli_")
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
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Read the dataset
nli_dataset_path = 'KorNLUDatasets/KorNLI'
sts_dataset_path = 'KorNLUDatasets/KorSTS'
logging.info("Read KorNLI train/KorSTS dev dataset")
train_files = glob.glob(os.path.join(nli_dataset_path, "*train.ko.tsv"))
dev_file = os.path.join(sts_dataset_path, "sts-dev.tsv")
train_samples = []
for train_file in train_files:
    train_samples += load_kor_nli_samples(train_file)
train_dataloader = datasets.NoDuplicatesDataLoader(train_samples, batch_size=args.batch_size)
dev_samples = load_kor_sts_samples(dev_file)
dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=args.batch_size, name='sts-dev')
train_loss = losses.MultipleNegativesRankingLoss(model)

# Configure the training.
warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs * 0.1)  # 10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
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
