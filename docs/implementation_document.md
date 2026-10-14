# Implementation Document

## Project structure

The repository is organized into documentation, data, model artifacts, source code, and tests:

- `docs/`: project documentation and weekly progress reports
- `dataset/`: source images and extracted landmark data
- `model_results/`: evaluation results of training
- `models/`: the MediaPipe hand-landmark model and serialized MLP parameters
- `src/`: application, image-processing, web-interface, and model-training code
- `tests/`: automated tests for the model, helper functions, image pipeline, and training
- `dataset.zip`: archived copy of the image dataset
- `pyproject.toml` and `poetry.lock`: project metadata and dependency definitions

The main program consists of separate pipelines for sign-language letter recognition, dataset generation, and model training. All three pipelines use the Multi-Layer Perceptron (MLP) implemented in `src/model/architecture.py`. In addition, `src/web.*` provides a separate web-based interface for running the letter-recognition pipeline across supported operating systems.

### Letter recognition

The letter recognition pipeline is selected through `src/main.py`. It supports recognition from individual images, directories of images, and a live camera stream. The selected pipeline in `src/image_pipeline.py` uses MediaPipe's Hand Landmarker to detect one hand and its 21 landmarks. The landmarks are then centered on the wrist and scaled using the distance to the middle-finger MCP joint, after which they are passed to the MLP along with the handedness indicator. The model output is mapped to a letter, while predictions with a confidence below 0.4 are reported as unknown (`UNK`). Detected landmarks are drawn on the input image or video frame.

### Dataset generation

Dataset generation uses the same landmark-detection and normalization process as letter recognition. When the user selects this option in `src/main.py`, images from a file or directory are processed in `src/image_pipeline.py` and their features are written to a CSV file given by the user. The label is determined from the first character of each image filename, while the CSV stores the label, handedness, and normalized landmark coordinates; the wrist coordinates are omitted because normalization places the wrist at the origin. `src/flip_images.py` can be used to create horizontally flipped copies of images, allowing both hand orientations to be included in the dataset.

### Model training

The model training pipeline is implemented in `src/model/train.py`. It loads the generated CSV file, separates the samples into training and validation sets using an 80/20 split, and trains the MLP with backpropagation and the Adam optimizer. Cross-entropy loss is used for classification, and training and validation loss and accuracy are evaluated during training. The results are saved as a plot, and the trained model parameters are serialized to a JSON file for later use by the recognition pipelines.

### Web interface

The web interface provides a browser-based alternative to the command-line pipeline. The required functions for recognition are copied from `src/image_pipeline.py` to `src/web.py` and modified to work with a simple HTTP server. Single image files can be opened in the interface or the browser's camera access can be enabled for recognition from stream. `src/web.html` provides the browser interface, which displays the predicted letter and its confidence. Because the interface uses browser APIs, it can be accessed from supported browsers on different operating systems.


## Possible flaws and improvements

The program can't recognize letters that require motion (J, Z, Ä, Ö, Å). This would be the next improvement, along with the program recognizing words instead of just letters.

The program also struggles to recognize letters T and N, often recognizing them as S, and letters K, R, and U are recognized as V. This is due to the model having trouble learning, not an architectural design flaw in the model or training algorithm.

## Performance of methods

Performance of model tested against PyTorch equivalents with n=10^8. Data propagations is slightly faster in own model than PyTorch's, but cross-entropy computation was ~9x slower (0.66s -> 5.37s) and softmax almost 2x (0.68s -> 1.20s). In practice, however, this doesn't make a significant difference, as both functions take inputs of size n=24.

## LLM usage

*GitHub Copilot* was used to generate the web interface (`src/web.html` and `src/web.py`). The *HTML* file was not modified after generation, while the *Python* script was modified to include proper documentation and work similär to original image pipeline.

*Duck.ai* (GPT-6 Luna) used to correct isolated syntax errors and help identify the root causes of other errors. Although it often struggled identifying the root causes, its suggestions helped me discover them myself.

## Final sources

- [MediaPipe hand landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker)
- [Sign Language MNIST dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- [MLP geeksforgeeks (GFG)](https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/)
- [Adam GFG](https://www.geeksforgeeks.org/deep-learning/adam-optimizer/)
- [Softmax and loss GFG](https://www.geeksforgeeks.org/machine-learning/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss/)
- [Tests for neural networks](https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/)
- [Weight initialization techniques](https://medium.com/@piyushkashyap045/understanding-weight-initialization-techniques-in-neural-networks-582e80a1e839)