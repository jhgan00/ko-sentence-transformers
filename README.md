# ko-sentence-transformers

이 프로젝트는 KoBERT 모델을 `sentence-transformers` 에서 보다 쉽게 사용하기 위해 만들어졌습니다.
[`Ko-Sentence-BERT-SKTBERT`](https://github.com/BM-K/KoSentenceBERT-SKT) 프로젝트에서는 KoBERT 모델을 `sentence-transformers` 에서 활용할 수 있도록 하였습니다. 
하지만 설치 과정에 약간의 번거로움이 있었고, 라이브러리 코드를 직접 수정하기 때문에 허깅페이스 허브를 활용하기 어려웠습니다.
`ko-sentence-transformers` 는 간단한 설치만으로 한국어 사전학습 모델을 문장 임베딩에 활용할 수 있도록 합니다. 

## Installation

`pip install` 을 통해 설치할 수 있습니다.

```bash
pip install ko-sentence-transformers
```

## Examples

사전학습된 KoBERT 모델을 가져와 `sentence-transformers`  API 에서 활용할 수 있습니다.
`training_nli_v2.py`, `training_sts.py` 파일에서 모델 파인튜닝 예시를 확인할 수 있습니다.

```python
from sentence_transformers import SentenceTransformer, models
from ko_sentence_transformers.models import KoBertTransformer
word_embedding_model = KoBertTransformer("monologg/kobert", max_seq_length=75)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
```

허깅페이스 허브에 업로드된 모델 역시 간단히 불러와 활용할 수 있습니다. 

```python
from sentence_transformers import SentenceTransformer, util
import numpy as np

embedder = SentenceTransformer("jhgan/ko-sbert-sts")

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
한 남자가 음식을 먹는다. (Score: 0.7417)
한 남자가 빵 한 조각을 먹는다. (Score: 0.6684)
한 남자가 말을 탄다. (Score: 0.1089)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.0717)
두 남자가 수레를 숲 속으로 밀었다. (Score: 0.0244)


======================


Query: 고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.

Top 5 most similar sentences in corpus:
원숭이 한 마리가 드럼을 연주한다. (Score: 0.7057)
한 여자가 바이올린을 연주한다. (Score: 0.3154)
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.2171)
두 남자가 수레를 숲 속으로 밀었다. (Score: 0.1294)
그 여자가 아이를 돌본다. (Score: 0.0979)


======================


Query: 치타가 들판을 가로 질러 먹이를 쫓는다.

Top 5 most similar sentences in corpus:
치타 한 마리가 먹이 뒤에서 달리고 있다. (Score: 0.7986)
두 남자가 수레를 숲 속으로 밀었다. (Score: 0.3255)
한 남자가 담으로 싸인 땅에서 백마를 타고 있다. (Score: 0.2688)
한 남자가 말을 탄다. (Score: 0.1530)
원숭이 한 마리가 드럼을 연주한다. (Score: 0.0913)
```

## KorSTS Benchmarks

카카오브레인의 KorNLU 데이터셋을 활용하여 `sentence-BERT` 모델을 학습시킨 후 다국어 모델의 성능과 비교한 결과입니다.
`ko-sbert-nli` 모델은 KorNLI 데이터셋을 활용하여 학습되었고, `ko-sbert-sts` 모델은 KorSTS 데이터셋을 활용하여 학습되었습니다. 
학습 및 성능 과정은 `training_nli_v2.py`, `training_sts.py`, `benchmark.py` 에서 확인할 수 있습니다.
학습된 모델은 허깅페이스 모델 허브에 공개되어있습니다.

- [ko-sbert-sts](https://huggingface.co/jhgan/ko-sbert-sts)
- [ko-sbert-nli](https://huggingface.co/jhgan/ko-sbert-nli) 

모델|Cosine Pearson|Cosine Spearman|Manhattan Pearson|Manhattan Spearman|Euclidean Pearson|Euclidean Spearman|Dot Pearson|Dot Spearman|
|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|:----:|
ko-sbert-nli|**82.03**|**82.36**|80.08|79.91|80.06|79.85|**75.76**|**74.72**
ko-sbert-sts|80.79|79.91|78.08|77.35|78.03|77.31|**75.96**|**75.20**
paraphrase-multilingual-mpnet-base-v2|80.69|82.00|**80.33**|**80.39**|**80.48**|**80.61**|70.30|68.48
distiluse-base-multilingual-cased-v1|75.50|74.83|73.05|73.15|73.67|73.86|74.79|73.95
distiluse-base-multilingual-cased-v2|75.62|74.83|73.03|72.87|73.68|73.62|63.80|62.35
paraphrase-multilingual-MiniLM-L12-v2|73.87|74.44|72.55|71.95|72.45|71.85|55.86|55.26


## References

- Ham, J., Choe, Y. J., Park, K., Choi, I., & Soh, H. (2020). Kornli and korsts: New benchmark datasets for korean natural language understanding. arXiv
preprint arXiv:2004.03289
- Reimers, Nils and Iryna Gurevych. “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.” ArXiv abs/1908.10084 (2019)
- [Ko-Sentence-BERT-SKTBERT](https://github.com/BM-K/KoSentenceBERT-SKT)
- [KoBERT](https://github.com/SKTBrain/KoBERT)