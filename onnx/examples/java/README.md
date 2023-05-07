# Java Onnxruntime Example

- `export_onnx.py` 스크립트를 실행해 모델을 `onnx` 포맷으로 변환해주세요
- 변환된 모델을 `src/main/java/resources` 경로에 위치시켜주세요
- `run.sh` 스크립트를 실행해주세요

```bash
git clone https://github.com/jhgan00/ko-sentence-transformers.git
cd ko-sentence-transformers
pip install -r requirements.txt
cd onnx
python export_onnx.py
mkdir -p examples/java/src/main/resources/ 
cp ./models/ko-sroberta-multitask.onnx examples/java/src/main/resources/
cd examples/java
bash run.sh
```