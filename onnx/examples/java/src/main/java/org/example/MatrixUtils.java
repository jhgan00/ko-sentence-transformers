package org.example;

import java.util.Arrays;
import java.util.Comparator;
import java.util.stream.IntStream;

public class MatrixUtils {

    public static double[][] normalize(double[][] matrix) {
        double[][] result = new double[matrix.length][matrix[0].length];
        for (int i = 0; i < matrix.length; i++) {
            double sumOfSquares = Math.sqrt(Arrays.stream(matrix[i]).map((x) -> Math.pow(x, 2.)).sum());
            for (int j = 0; j < matrix[i].length; j++) {
                result[i][j] = matrix[i][j] / sumOfSquares;
            }
        }
        return result;
    }

    public static double[][] matMul(double[][] A, double[][] B) {
        int rowsA = A.length;
        int colsA = A[0].length;
        int colsB = B[0].length;

        double[][] result = new double[rowsA][colsB];

        for (int i = 0; i < rowsA; i++) {
            for (int j = 0; j < colsB; j++) {
                for (int k = 0; k < colsA; k++) {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return result;
    }

    public static double[][] transpose(double[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        double[][] transposeMatrix = new double[cols][rows];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                transposeMatrix[j][i] = matrix[i][j];
            }
        }
        return transposeMatrix;
    }


    public static double[][] pairwiseCosineSimilarity(double[][] A, double[][] B) {
        double[][] normA = normalize(A);
        double[][] normB = normalize(B);
        return matMul(normA, transpose(normB));
    }

    public static int[] argmax(double[][] matrix) {
        int rows = matrix.length;
        int[] argMaxIndices = new int[rows];

        for (int i = 0; i < rows; i++) {
            double[] row = matrix[i];
            int maxIndex = 0;
            double maxValue = row[0];
            for (int j = 1; j < row.length; j++) {
                if (row[j] > maxValue) {
                    maxIndex = j;
                    maxValue = row[j];
                }
            }
            argMaxIndices[i] = maxIndex;
        }
        return argMaxIndices;
    }

    public static int[] argsort(double[] array, boolean ascending) {
        Integer[] indices = IntStream.range(0, array.length)
                .boxed()
                .toArray(Integer[]::new);
        if (ascending) {
            Arrays.sort(indices, Comparator.comparingDouble(i -> array[i]));
        }
        else {
            Arrays.sort(indices, Comparator.comparingDouble(i -> -array[i]));
        }
        return Arrays.stream(indices).mapToInt(Integer::intValue).toArray();
    }
}
