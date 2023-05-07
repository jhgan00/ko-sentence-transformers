package org.example;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.util.HashMap;
import java.util.Map;

public class SentenceTransformer {


    private final OrtEnvironment env;
    private final OrtSession session;
    private int maxLength;

    public SentenceTransformer(String modelPath, long maxLength) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        var sessionOptions = new OrtSession.SessionOptions();
        sessionOptions.addCPU(false);
        sessionOptions.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        this.session = this.env.createSession(modelPath, sessionOptions);
        this.maxLength = (int) maxLength;
    }

    private Map<String, OnnxTensor> preprocess(Encoding[] encodings) throws OrtException{

        Map<String, OnnxTensor> container = new HashMap<>();

        int batchSize = encodings.length;

        long[][] ids = new long[batchSize][maxLength];
        long[][] attentionMask = new long[batchSize][maxLength];
        long[][] typeIds = new long[batchSize][maxLength];

        for (int i = 0; i < batchSize; i++) {
            ids[i] = encodings[i].getIds();
            attentionMask[i] = encodings[i].getAttentionMask();
            typeIds[i] = encodings[i].getTypeIds();
        }

        OnnxTensor idsTensor = OnnxTensor.createTensor(env, ids);
        OnnxTensor maskTensor = OnnxTensor.createTensor(env, attentionMask);
        OnnxTensor typeTensor = OnnxTensor.createTensor(env, typeIds);

        container.put("input_ids", idsTensor);
        container.put("attention_mask", maskTensor);
        container.put("token_type_ids", typeTensor);

        return container;

    }

    public double[][] run(Encoding[] encodings) throws OrtException{
        Map<String, OnnxTensor> inputContainer = preprocess(encodings);
        OrtSession.Result result = session.run(inputContainer);
        float[][] floatArray = (float[][]) result.get(1).getValue();
        int rows = floatArray.length;
        int cols = floatArray[0].length;
        double[][] doubleArray = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                doubleArray[i][j] = floatArray[i][j];
            }
        }
        return doubleArray;
    }

}
