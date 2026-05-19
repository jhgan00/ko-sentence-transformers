"""
This examples trains KoBERT for the STS benchmark from scratch.
It generates sentence embeddings that can be compared using cosine-similarity to measure the similarity.
Usage:
python training_sts.py --model_name_or_path klue/bert-base
"""
import argparse
import logging
import math
import os
import random
from datetime import datetime

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, LoggingHandler, models, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator
from torch.utils.data import DataLoader

from data_util import load_kor_sts_samples

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str)
parser.add_argument("--max_seq_length", type=int, default=128)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_epochs", type=int, default=5)
parser.add_argument("--output_dir", type=str, default="output")
parser.add_argument("--output_prefix", type=str, default="kor_sts_")
parser.add_argument("--seed", type=int, default=777)
parser.add_argument("--matryoshka_dims", type=str, default=None,
                    help="Comma-separated nested dims for Matryoshka Representation Learning "
                         "(e.g. '768,512,256,128,64,32'). If omitted, MRL is disabled.")
parser.add_argument("--matryoshka_weights", type=str, default=None,
                    help="Comma-separated per-dim loss weights matching --matryoshka_dims. "
                         "Defaults to uniform 1.0 when omitted.")
args = parser.parse_args()

matryoshka_dims = None
matryoshka_weights = None
if args.matryoshka_dims:
    matryoshka_dims = [int(d) for d in args.matryoshka_dims.split(",")]
    if args.matryoshka_weights:
        matryoshka_weights = [float(w) for w in args.matryoshka_weights.split(",")]
        if len(matryoshka_weights) != len(matryoshka_dims):
            raise ValueError("--matryoshka_weights length must match --matryoshka_dims length")

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
logging.info("Read KorSTS train/dev dataset")
sts_dataset_path = 'KorNLUDatasets/KorSTS'
train_file, dev_file = os.path.join(sts_dataset_path, "sts-train.tsv"), os.path.join(sts_dataset_path, "sts-dev.tsv")
train_samples, dev_samples = load_kor_sts_samples(train_file), load_kor_sts_samples(dev_file)
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=args.batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)

if matryoshka_dims:
    train_loss = losses.MatryoshkaLoss(
        model, train_loss,
        matryoshka_dims=matryoshka_dims,
        matryoshka_weights=matryoshka_weights,
    )
    dev_evaluator = SequentialEvaluator([
        EmbeddingSimilarityEvaluator.from_input_examples(
            dev_samples, batch_size=args.batch_size,
            truncate_dim=d, name=f'sts-dev-{d}',
        ) for d in matryoshka_dims
    ], main_score_function=lambda scores: scores[0])
else:
    dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(
        dev_samples, batch_size=args.batch_size, name='sts-dev'
    )

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
if matryoshka_dims:
    test_evaluator = SequentialEvaluator([
        EmbeddingSimilarityEvaluator.from_input_examples(
            test_samples, truncate_dim=d, name=f'sts-test-{d}',
        ) for d in matryoshka_dims
    ], main_score_function=lambda scores: scores[0])
else:
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')
test_evaluator(model, output_path=model_save_path)
