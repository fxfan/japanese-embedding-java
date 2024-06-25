package org.example;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Map;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

public class Main {
    public static void main(String[] args) throws Exception {

        Path modelPath = Paths.get(Main.class.getResource("/model.onnx").getPath());
        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession sess = env.createSession(modelPath.toString());

        System.out.println("### Input Info");
        System.out.println(sess.getInputInfo().entrySet().stream()
            .map(e -> e.getKey() + ": " + e.getValue())
            .collect(Collectors.joining("\n")));

        String modelName = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2";
        String text = "ジョバンニは、口笛を吹いているようなさびしい口付きで、檜のまっ黒にならんだ町の坂を下りて来たのでした";
        HuggingFaceTokenizer tokenizer = HuggingFaceTokenizer.newInstance(modelName);
        Encoding encoding = tokenizer.encode(text);
        long[] inputIds = encoding.getIds();
        long[] attentionMask = encoding.getAttentionMask();

        Map<String, OnnxTensor> input = Map.of(
            "input_ids", OnnxTensor.createTensor(env, new long[][]{ inputIds }),
            "attention_mask", OnnxTensor.createTensor(env, new long[][]{ attentionMask })
        );

        OrtSession.Result result = sess.run(input);

        String resultString = StreamSupport.stream(result.spliterator(), false)
            .map(e -> e.getKey() + ": " + e.getValue())
            .collect(Collectors.joining("\n"));
        System.out.println("### Result");
        System.out.println(resultString);

        String embedding = result.get("sentence_embedding")
            .map(val -> {
                if (val instanceof OnnxTensor tensor) {
                    float[] array = new float[tensor.getFloatBuffer().remaining()];
                    tensor.getFloatBuffer().get(array);
                    return Arrays.toString(array);
                } else {
                    return "Unexpected type for sentence_embedding";
                }
            })
            .orElse("sentence_embedding not found");
        System.out.println("### Sentence Embedding");
        System.out.println(embedding);

        result.close();
        sess.close();
        env.close();
    }

}
