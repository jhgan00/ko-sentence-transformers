import os
import torch
import numpy as np
import onnxruntime as ort
from pathlib import Path
from transformers.convert_graph_to_onnx import convert
from transformers import PreTrainedTokenizerFast
from sentence_transformers import SentenceTransformer, util


def mean_pooling(embeddings: np.ndarray, attention_mask: np.ndarray):

    input_mask_expanded = attention_mask[..., np.newaxis].astype(np.float32)
    sum_embeddings = np.sum(embeddings * input_mask_expanded, axis=1)
    sum_mask = input_mask_expanded.sum(axis=1)
    sum_mask = np.clip(sum_mask, a_min=1e-9, a_max=None)

    return sum_embeddings / sum_mask


if __name__ == "__main__":

    # export to onnx
    output_dir = "models"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=False)
 
    output_fpath = os.path.join(output_dir, "ko-sroberta-multitask.onnx")
    convert(framework="pt", model="jhgan/ko-sroberta-multitask", output=Path(output_fpath), opset=12)

    # test run
    tokenizer = PreTrainedTokenizerFast.from_pretrained("jhgan/ko-sroberta-multitask")
    session = ort.InferenceSession(output_fpath)

    # input preparation
    corpus = [
        '한 남자가 음식을 먹는다.',
        '한 남자가 빵 한 조각을 먹는다.',
        '그 여자가 아이를 돌본다.',
        '한 남자가 말을 탄다.',
        '한 여자가 바이올린을 연주한다.',
        '두 남자가 수레를 숲 속으로 밀었다.',
        '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.',
        '원숭이 한 마리가 드럼을 연주한다.',
        '치타 한 마리가 먹이 뒤에서 달리고 있다.'
    ]

    queries = [
        '한 남자가 파스타를 먹는다.',
        '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.',
        '치타가 들판을 가로 질러 먹이를 쫓는다.'
    ]

    
    tokenized_corpus = tokenizer(corpus, return_tensors="pt", padding=True, truncation=True)
    corpus_onnx = {k: v.cpu().detach().numpy() for k, v in tokenized_corpus.items()}

    tokenized_queries = tokenizer(queries, return_tensors="pt", padding=True, truncation=True)
    queries_onnx = {k: v.cpu().detach().numpy() for k, v in tokenized_queries.items()}

    # run inference
    corpus_embeddings, pooled_corpus = session.run(None, corpus_onnx)
    corpus_embeddings = mean_pooling(corpus_embeddings, corpus_onnx["attention_mask"])

    query_embeddings, pooled_queries = session.run(None, queries_onnx)
    query_embeddings = mean_pooling(query_embeddings, queries_onnx["attention_mask"])

    cos_scores = util.pytorch_cos_sim(query_embeddings, corpus_embeddings).cpu()

    top_k = 5
    for query, cosine in zip(queries, cos_scores):
        top_results = np.argpartition(-cosine, range(top_k))[:top_k]
        print("Query:", query)
        for idx in top_results:
            print(corpus[idx].strip(), "(Score: %.4f)" % (cosine[idx]))
        print("\n")

