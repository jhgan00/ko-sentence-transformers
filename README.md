# ko-sentence-transformers

이 프로젝트는 한국어 사전학습 모델을 한국어 문장 임베딩에 활용하기 위해 만들어졌습니다.
KorNLU 데이터셋으로 한국어 사전학습 모델을 파인튜닝하고, 파인튜닝된 모델을 `sentence-transformers`에서 손쉽게 다운로드받아 활용할 수 있도록 합니다.

## KorSTS Benchmarks

- 카카오브레인의 KorNLU 데이터셋을 활용하여 모델을 학습시킨 후 다국어 모델의 성능과 비교한 결과입니다.
- 사전학습 모델은 [`klue`](https://huggingface.co/klue )의 `bert-base`, `roberta-base`를 활용하였습니다
- `ko-*-nli`, `ko-*-sts` 모델은 각각 KorNLI, KorSTS 데이터셋을 활용하여 학습되었으며, `ko-*-multitask` 모델은 두 데이터셋을 모두 활용하여 멀티태스크로 학습되었습니다.
- 학습 및 성능 평가 과정은 `training_*.py`, `benchmark.py` 에서 확인할 수 있습니다.
- 학습된 모델은 허깅페이스 모델 허브에 공개되어있습니다. 

|model|cosine_pearson|cosine_spearman|euclidean_pearson|euclidean_spearman|manhattan_pearson|manhattan_spearman|dot_pearson|dot_spearman|
|:-------------------------|-----------------:|------------------:|--------------------:|---------------------:|--------------------:|---------------------:|--------------:|---------------:|
|[ko-sroberta-multitask](https://huggingface.co/jhgan/ko-sroberta-multitask)|**84.77**|**85.6**|**83.71**|**84.40**|**83.70**|**84.38**|82.42|82.33|
|[ko-sbert-multitask](https://huggingface.co/jhgan/ko-sbert-multitask)|84.13|84.71|82.42|82.66|82.41|82.69|80.05|79.69|
|[ko-sroberta-base-nli](https://huggingface.co/jhgan/ko-sroberta-nli)|82.83|83.85|82.87|83.29|82.88|83.28|80.34|79.69|
|[ko-sbert-nli](https://huggingface.co/jhgan/ko-sbert-multitask)|82.24|83.16|82.19|82.31|82.18|82.3|79.3|78.78|
|[ko-sroberta-sts](https://huggingface.co/jhgan/ko-sroberta-sts)|81.84|81.82|81.15|81.25|81.14|81.25|79.09|78.54|
|[ko-sbert-sts](https://huggingface.co/jhgan/ko-sbert-sts)|81.55|81.23|79.94|79.79|79.9|79.75|76.02|75.31|
paraphrase-multilingual-mpnet-base-v2|80.69|82.00|80.33|80.39|80.48|80.61|70.30|68.48
distiluse-base-multilingual-cased-v1|75.50|74.83|73.05|73.15|73.67|73.86|74.79|73.95
distiluse-base-multilingual-cased-v2|75.62|74.83|73.03|72.87|73.68|73.62|63.80|62.35
paraphrase-multilingual-MiniLM-L12-v2|73.87|74.44|72.55|71.95|72.45|71.85|55.86|55.26

## Examples


 - 예시 출처: https://github.com/BM-K/KoSentenceBERT-SKT

허깅페이스 허브에 업로드된 sentence-BERT 모델을 가져와 `sentence-transformers` 에서 활용할 수 있습니다.
아래는 임베딩 벡터를 통해 가장 유사한 문장을 찾는 예시입니다. 
더 많은 예시는 [sentence-transformers 문서](https://www.sbert.net/index.html)를 참고해주세요. 

```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

embedder = SentenceTransformer("jhgan/ko-sroberta-multitask")

# Corpus with example sentences
corpus = ['한 남자가 음식을 먹는다.',
 '한 남자가 빵 한 조각을 먹는다.',
 '그 여자가 아이를 돌본다.',
 '한 남자가 말을 탄다.',
 '한 여자가 바이올린을 연주한다.',
 '두 남자가 수레를 숲 속으로 밀었다.',
 '한 남자가 담으로 싸인 땅에서 백마를 타고 있다.',
 '원숭이 한 마리가 드럼을 연주한다.',
 '치타 한 마리가 먹이 뒤에서 달리고 있다.']

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Query sentences:
queries = ['한 남자가 파스타를 먹는다.',
  '고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.',
  '치타가 들판을 가로 질러 먹이를 쫓는다.']

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
top_k = 5
for query in queries:
 query_embedding = embedder.encode(query, convert_to_tensor=True)
 cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
 cos_scores = cos_scores.cpu()

 #We use np.argpartition, to only partially sort the top_k results
 top_results = np.argpartition(-cos_scores, range(top_k))[0:top_k]

 print("\n\n======================\n\n")
 print("Query:", query)
 print("\nTop 5 most similar sentences in corpus:")

 for idx in top_results[0:top_k]:
  print(corpus[idx].strip(), "(Score: %.4f)" % (cos_scores[idx]))
```

```
======================


Query: 한 남자가 파스타를 먹는다.

Top 5 most similar sentences in corpus:
한 남자가 음식을 먹는다. (Score: 0.5892)
한 남자가 빵 한 조각을 먹는다. (Score: 0.4919)
한 남자가 말을 탄다. (Score: 0.1077)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.0283)
한 여자가 바이올린을 연주한다. (Score: -0.0296)


======================


Query: 고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.

Top 5 most similar sentences in corpus:
원숭이 한 마리가 드럼을 연주한다. (Score: 0.7099)
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.2859)
한 여자가 바이올린을 연주한다. (Score: 0.2431)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.1852)
한 남자가 말을 탄다. (Score: 0.1474)


======================


Query: 치타가 들판을 가로 질러 먹이를 쫓는다.

Top 5 most similar sentences in corpus:
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.8251)
원숭이 한 마리가 드럼을 연주한다. (Score: 0.2119)
한 남자가 말을 탄다. (Score: 0.1850)
두 남자가 수레를 숲 속으로 밀었다. (Score: 0.1596)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.1276)
```

## Training

직접 모델을 파인튜닝하려면 [`KorNLUDatasets`](https://github.com/kakaobrain/KorNLUDatasets) 저장소를 클론한 후 `training_*.py` 스크립트를 실행시키면 됩니다.
`train.sh` 파일에서 학습 예시를 확인할 수 있습니다.

```bash
git clone https://github.com/jhgan00/ko-sentence-transformers.git
cd ko-sentence-transformers
pip install -r requirements.txt
git clone https://github.com/kakaobrain/KorNLUDatasets.git
python training_multi-task.py --model_name_or_path klue/roberta-base
```

## ONNX 변환

`requirements.txt` 설치 후 `onnx` 디렉토리에서 `export_onnx.py` 스크립트를 실행합니다.
변환된 `onnx` 모델은 `onnx/models/ko-sroberta-multitask.onnx` 경로에 저장됩니다.

```
git clone https://github.com/jhgan00/ko-sentence-transformers.git
cd ko-sentence-transformers
pip install -r requirements.txt
cd onnx
python export_onnx.py
```

## Updates

### Dec 27, 2021

- 사전학습 bert 모델을 KLUE 모델로 변경
- KLUE roberta-base 모델 추가

### May 7, 2023

- `onnx` 변환 스크립트 & 자바 예시 추가

## References

- Ham, J., Choe, Y. J., Park, K., Choi, I., & Soh, H. (2020). Kornli and korsts: New benchmark datasets for korean natural language understanding. arXiv
preprint arXiv:2004.03289
- Reimers, Nils and Iryna Gurevych. “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.” ArXiv abs/1908.10084 (2019)
- Reimers, Nils and Iryna Gurevych. “Making Monolingual Sentence Embeddings Multilingual Using Knowledge Distillation.” EMNLP (2020)
- [Ko-Sentence-BERT-SKTBERT](https://github.com/BM-K/KoSentenceBERT-SKT)
- [KoBERT](https://github.com/SKTBrain/KoBERT)
