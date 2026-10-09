## General

**WIP**: Training and validation losses and accuracies are available as a plot and a text file in `model_results/`. The directory also contains results showing how the pretrained model classifies images and stream of me signing each valid letter. N.B I was not included in the training data.

## Coverage report

```
Name                        Stmts   Miss Branch BrPart  Cover   Missing
-----------------------------------------------------------------------
src/model/architecture.py     127      5     18      2    95%   166, 209, 274-276
src/model/helpers.py           43      0      4      0   100%
src/model/train.py             62      4     16      2    92%   108, 138, 168-169
-----------------------------------------------------------------------
TOTAL                         232      9     38      4    95%
```

## Test structure

```
sign-language-recognizer/
├── check_extracted_landmarks.py
├── test_architecture.py
|   └── TestModel
|       ├── test_architecture
|       ├── test_dropout
|       ├── test_error_checking
|       └── test_serialization
|
├── test_helpers.py
|   └── TestMethods
|       ├── test_cross_entropy_small
|       ├── test_cross_entropy
|       ├── test_relu_small
|       ├── test_relu
|       ├── test_softmax_small
|       ├── test_softmax
|       ├── test_adam_small
|       ├── test_adam
|       ├── test_create_dataset
|       └── test_classification
|
├── test_image_pipeline.py
|   └── TestPipeline
|       └── test_normalization
|
└── test_training.py
    └── TestTraining
        ├── test_propagation
        ├── test_gradients
        └── test_layers_change
```

### check_extracted_landmarks.py (manual test)

Ensures that landmarks extracted with `src/image_pipeline.py::landmarks_to_csv` are correct. This is done by writing unnormalized landmarks to a file using said function and then reading them and drawing them on their corresponding images.

### TestModel

Ensures that the implemented model's architecture is correct and its parameters are properly saved and loaded with the following tests:
- `test_architecture`: data is correctly forward and backward propagated. output data compared with equivalent PyTorch model
- `test_dropout`: the dropout layer functions correctly
- `test_error_checking`: invalid inputs and propagation orders are accounted for
- `test_serialization`: model parameters are correctly saved to and loaded from a JSON file and errors are handled

### TestMethods

- Ensures that implemented mathematical functions (cross entropy loss, ReLU, softmax, adam) work correctly and tests performance against PyTorch's implementations using inputs of size n=10 (functionality) and n=10^8 (performance)
- `test_create_dataset`: training and validation datasets are correctly generated
- `test_classification`: model output logits are classified to correct label

### TestPipeline

- `test_normalization`: landmarks are centered and normalized correctly

### TestTraining

Tests that the training loop correctly trains the model and metrics are properly plotted with the following tests:
- `test_propagation`: loss is propagated and the weight updates make the model improve
- `test_gradients`: gradients are non_zero and loss decreases
- `test_layers_change`: all model layers update after each optimizer step
- `test_plotting`: training results are saved as a PNG or JPG plot and errors are handled

## Test reproducibility

All tests can be run from the project root directory with
```bash
poetry run pytest
```

Individual test file can be run with
```bash
poetry run pytest tests/test_*NAME*.py
```

Individual test groups can by run with
```bash
poetry run pytest tests/test_*NAME*.py::*CLASS_NAME*
```

Individual tests can be run with
```bash
poetry run pytest tests/test_*NAME*.py::*CLASS_NAME*::*METHOD_NAME*
```

The coverage report can be composed with
```bash
poetry run coverage run --branch -m pytest
```

The report can then be displayed with `poetry run coverage report -m` or `poetry run coverage html`. To run the previous commands using Docker, simply replace `poetry run` with
```bash
docker run -it --rm sign-language-recognizer
```