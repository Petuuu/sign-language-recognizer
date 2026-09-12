## General

- **Programme:** Computer Science (TKT)

- **Problem:** Deaf and mute people may have difficulty communicating with people who do not know sign language. The proposed solution is an machine learning model that recognizes signing and converts it into text.

- **Programming languages:** Python

- **Other known languages:** C++, JavaScript

- **Libraries:** OpenCV, MediaPipe, NumPy

- **Input:** Initially, the program will accept only images. Later, it will be able to process raw live camera footage. The program identifies hand landmarks in the video, which a machine learning model classifies as a letter or as unknown.

- **Data structures:** Numpy arrays

- **Algorithms:**

-> Stream processing with OpenCV

-> Hand landmark detection with MediaPipe

-> Multi-Layer Perceptron (MLP) for static letters

-> Backpropagation training algorithm with the Adam optimizer

(-> Depending on the available time, a Gated Recurrent Unit (GRU) for dynamic letters)

- **Sources:**

-> MediaPipe hand landmarker: https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker

-> NumPy docs: https://numpy.org/doc/stable/

-> Sign Language MNIST dataset (alphabet same in ASL and Finnish): https://www.kaggle.com/datasets/datamunge/sign-language-mnist

-> MLP geeksforgeeks (GFG): https://www.geeksforgeeks.org/deep-learning/multi-layer-perceptron-learning-in-tensorflow/

-> Adam GFG: https://www.geeksforgeeks.org/deep-learning/adam-optimizer/

-> Backpropagation GFG (Python implementation not viewed or used): https://www.geeksforgeeks.org/machine-learning/backpropagation-in-neural-network/

(-> GRU GFG: https://www.geeksforgeeks.org/machine-learning/gated-recurrent-unit-networks/)


## Core

Building and training a machine learning model using the backpropagation algorithm and the Adam optimizer to classify sign language letters from hand landmarks detected in images or videos.