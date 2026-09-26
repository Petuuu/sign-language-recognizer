## General

**WIP**: Training and validation losses and accuracies are available as a plot and a text file in `model_results/`. The directory also contains results showing how the pretrained model classifies images and video of me signing each valid letter. N.B I was not included in the training data.

## Coverage report

```
Name                        Stmts   Miss Branch BrPart  Cover   Missing
-----------------------------------------------------------------------
src/model/architecture.py      74      3      8      3    93%   126, 152, 188
src/model/helpers.py           14      0      0      0   100%
src/model/train.py              4      0      0      0   100%
-----------------------------------------------------------------------
TOTAL                          92      3      8      3    94%
```

## Test structure

```
test_model.py
├── TestModel
|   ├── test_architecture
|   ├── test_dropout
|   └── test_error_checking
├── TestTraining
|   ├── test_propagation
|   ├── test_gradients
|   └── test_layers_change
└── TestMethods
    ├── test_cross_entropy_small
    ├── test_cross_entropy
    ├── test_relu_small
    ├── test_relu
    ├── test_softmax_small
    └── test_softmax
```

### TestModel

Ensures that the implemented model's architecture is correct with the following tests:
- `test_architecture`: data is correctly forward and backward propagated. output data compared with equivalent PyTorch model
- `test_dropout`: the dropout layer functions correctly
- `test_error_checking`: invalid inputs and propagation orders are accounted for

### TestTraining

Tests that the training loop correctly trains the model with the following tests:
- `test_propagation`: loss is propagated and the weight updates make the model improve
- `test_gradients`: gradients are non_zero and loss decreases
- `test_layers_change`: all model layers update after each optimizer step

### TestMethods

Ensures that implemented mathematical functions (cross entropy loss, ReLU, softmax) work correctly and tests performance against PyTorch's implementations using inputs of size n=10 (functionality) and n=10^8 (performance)

## Test reproducibility

All tests can be run from the project root directory with
```bash
pytest
```

Individual test groups can by run with
```bash
pytest test_model.py::*CLASS_NAME*
```

Individual tests can be run with
```bash
pytest test_model.py::*CLASS_NAME*::*METHOD_NAME*
```

The coverage report can be composed with
```bash
coverage run --branch -m pytest
```

The report can then be displayed with `coverage report -m` or `coverage html`.