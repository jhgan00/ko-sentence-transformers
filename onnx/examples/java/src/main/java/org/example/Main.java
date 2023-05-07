package org.example;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.huggingface.tokenizers.jni.TokenizersLibrary;
import ai.onnxruntime.OrtException;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;


public class Main {

    public static void main(String[] args) throws IOException, OrtException {

        Map<String, String> tokenizerConfig = new HashMap<>();
        tokenizerConfig.putIfAbsent("tokenizer", "jhgan/ko-sroberta-multitask");
        HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.builder(tokenizerConfig).optPadToMaxLength().build();
        String modelPath = Main.class.getClassLoader().getResource("ko-sroberta-multitask.onnx").getFile();
        long maxLength = TokenizersLibrary.LIB.getMaxLength(tokenizer.getHandle());
        SentenceTransformer model = new SentenceTransformer(modelPath, maxLength);

        String[] corpus = new String[]{
            "한 남자가 음식을 먹는다.",
            "한 남자가 빵 한 조각을 먹는다.",
            "그 여자가 아이를 돌본다.",
            "한 남자가 말을 탄다.",
            "한 여자가 바이올린을 연주한다.",
            "두 남자가 수레를 숲 속으로 밀었다.",
            "한 남자가 담으로 싸인 땅에서 백마를 타고 있다.",
            "원숭이 한 마리가 드럼을 연주한다.",
            "치타 한 마리가 먹이 뒤에서 달리고 있다."
        };

        String[] queries = new String[]{
            "한 남자가 파스타를 먹는다.",
            "고릴라 의상을 입은 누군가가 드럼을 연주하고 있다.",
            "치타가 들판을 가로 질러 먹이를 쫓는다."
        };

        Encoding[] corpusEncodings = tokenizer.batchEncode(corpus);
        Encoding[] queryEncodings = tokenizer.batchEncode(queries);

        double[][] corpusEmbeddings = model.run(corpusEncodings);
        double[][] queryEmbeddings = model.run(queryEncodings);

        double[][] cosineSimilarity = MatrixUtils.pairwiseCosineSimilarity(queryEmbeddings, corpusEmbeddings);

        for (int i = 0; i < queries.length; i++) {
            int[] argsort = MatrixUtils.argsort(cosineSimilarity[i], false);
            System.out.println("쿼리: " + queries[i]);
            for (int j = 0; j < 5; j++) {
                System.out.println("\t" + corpus[argsort[j]] + "(" + cosineSimilarity[i][argsort[j]] + ")");
            }
        }

    }

}