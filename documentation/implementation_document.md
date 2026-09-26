## Project structure

```
sign-language-recognizer/
├── documentation/
|   ├── coverage_report.txt
|   ├── implementation_document.md
|   ├── pylint_report.txt
|   ├── specification_document.md
|   ├── test_document.md
|   └── weekly_progress/
|       ├── week_1.md
|       ...
|       └── week_6.md
├── models/
|   └── hand_landmarker.task
├── src/
|   ├── image_pipeline.py
|   ├── main.py
|   └── model/
|       ├── architecture.py
|       ├── helpers.py
|       ├── test_model.py
|       └── train.py
├── .gitignore
├── dataset.zip
├── LICENSE
├── README.md
└── requirements.txt
```

- `dataset.zip`: images used in project
- `implementation_document.md`: project structure, implementation details, and final sources
- `specification_document.md`: project overview and initial sources
- `test_document.md`: testing overview and results
- `weekly_progress/`: weekly progress reports
- `models/`: machine learning models related to project
- `src/image_pipeline.py`: image and video processing and landmarking
- `src/main.py`: main program
- `src/model/`: building, training, and testing implemented machine learning model

## Possible flaws and improvements

The program can't recognize letters that require motion (J, Z, Ä, Ö, Å). This would be the next improvement, along with the program recognizing words instead of just letters.

## Performance of methods

Performance of model tested against PyTorch equivalents with n=10^8. Data propagations is slightly faster in own model than PyTorch's, but cross-entropy computation was ~9x slower (0.66s -> 5.37s) and softmax almost 2x (0.68s -> 1.20s).

## LLM usage

Duck.ai (GPT-5.6 Luna) used to correct isolated syntax errors and help identify the root causes of other errors. Although it struggled identifying the root causes, its suggestions helped me discover them myself.

## Final sources

-> MediaPipe hand landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker

-> Sign Language MNIST dataset (alphabet same in ASL and Finnish): https://www.kaggle.com/datasets/datamunge/sign-language-mnist

-> MLP geeksforgeeks (GFG): https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/

-> Adam GFG: https://www.geeksforgeeks.org/deep-learning/adam-optimizer/

-> Backpropagation GFG (Python implementation not viewed or used): https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/

-> Softmax and loss GFG: https://www.geeksforgeeks.org/machine-learning/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss/

-> Tests for neural networks: https://www.sebastianbjorkqvist.com/blog/writing-automated-tests-for-neural-networks/

-> Weight initialization techniques: https://medium.com/@piyushkashyap045/understanding-weight-initialization-techniques-in-neural-networks-582e80a1e839