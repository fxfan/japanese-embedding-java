# Japanese Embedding Java

This repository demonstrates how to export a HuggingFace Japanese embedding model to ONNX format and use it in a Java application.

## Setup

### Export the ONNX model

1. Install optimum and related libraries:

    ```shell
    venv $ python -m pip install optimum
    venv $ pip install onnx
    venv $ pip install onnx onnxruntime
    
    # The version of numpy used by optimum-cli is not compatible with version 2.x.
    # Downgrade numpy if necessary.
    venv $ pip install numpy==1.26.4
     ```

2. Export the HuggingFace model to ONNX format:

    ```shell
    venv $ optimum-cli export onnx --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 ./onnx-model/
    cp onnx-model/model.onnx src/main/resources/
    ```

### Run the Java application
1. Ensure the `model.onnx` file is placed in `src/main/resources/`.
2. Build and run the Java application using Maven or Gradle.