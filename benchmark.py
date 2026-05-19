import argparse
import os
import csv
import logging
from sentence_transformers import LoggingHandler, SentenceTransformer, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator, SequentialEvaluator

sts_dataset_path = 'KorNLUDatasets/KorSTS'

logging.basicConfig(
    format='%(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    handlers=[LoggingHandler()]
)

parser = argparse.ArgumentParser()
parser.add_argument("--matryoshka_model", type=str, default=None,
                    help="Model name or local path to evaluate at multiple truncated dims. "
                         "Skips the default 6-model sweep when provided.")
parser.add_argument("--matryoshka_dims", type=str, default="768,512,256,128,64,32",
                    help="Comma-separated dims for the Matryoshka sweep.")
args = parser.parse_args()

# Read STSbenchmark dataset and use it as development set
test_samples = []
test_file = os.path.join(sts_dataset_path, "sts-test.tsv")
with open(test_file, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1
        test_samples.append(InputExample(texts=[row['sentence1'], row['sentence2']], label=score))
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='sts-test')

if args.matryoshka_model:
    dims = [int(d) for d in args.matryoshka_dims.split(",")]
    mrl_evaluator = SequentialEvaluator([
        EmbeddingSimilarityEvaluator.from_input_examples(
            test_samples, truncate_dim=d, name=f'sts-test-{d}',
        ) for d in dims
    ], main_score_function=lambda scores: scores[0])
    model = SentenceTransformer(args.matryoshka_model)
    mrl_evaluator(model)
    raise SystemExit(0)

model = SentenceTransformer("jhgan/ko-sroberta-multitask")
test_evaluator(model)

model = SentenceTransformer("jhgan/ko-sroberta-nli")
test_evaluator(model)

model = SentenceTransformer("jhgan/ko-sroberta-sts")
test_evaluator(model)

model = SentenceTransformer("jhgan/ko-sbert-multitask")
test_evaluator(model)

model = SentenceTransformer("jhgan/ko-sbert-nli")
test_evaluator(model)

model = SentenceTransformer("jhgan/ko-sbert-sts")
test_evaluator(model)
#
# model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
# test_evaluator(model)
#
# model = SentenceTransformer("distiluse-base-multilingual-cased-v2")
# test_evaluator(model)
#
# model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
# test_evaluator(model)
#
# model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
# test_evaluator(model)
