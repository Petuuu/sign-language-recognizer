# User Guide

**NOTE:** The image and video mode may not work on the command line interface on other than Linux (confirmed not to work on MacOS). If this is the case, start the web server to test these.

## Installation

### Requirements

- Python 3.11+
- Poetry

### Cloning the repository and setup

Clone the reposirory and navigate to the project with
```bash
git clone https://github.com/Petuuu/sign-language-recognizer.git
cd sign-language-recognizer
```

Unzip the dataset with, for example,
```bash
unzip dataset.zip
```

First, clone the repository, navigate to the project, and unzip `dataset.zip` to access the dataset used for training.

### Dependencies

Create the virtual environment with (may take several minutes)
```bash
poetry install
```

## Command line interface

Run the program from project root with
```bash
poetry run python -m src.main
```

Select one of the available modes when prompted:

- `0` saves hand landmarks from an image or image directory
- `1` detects letter(s) from an image or image directory
- `2` reads frames from the camera and detects letters continuously
- `3` runs training algorithm on model

In camera mode, cycle through images by pressing `q` or `Esc` in the OpenCV preview window. To stop the program, cycle through all images or press `Ctrl+C` in the terminal and cycle to the next image.

To stop the camera mode, press `q` or `Esc` in the OpenCV preview window or press
`Ctrl+C` in the terminal.

## Browser application

Start the development server with
```bash
poetry run python -m src.web
```

Open <http://localhost:8000>, click **Start camera**, and grant the browser camera permission. The browser captures frames and sends them to the container for MediaPipe landmark detection and classification. You can also select a JPG, JPEG, or PNG image to recognize it once. Stop the server with `Ctrl+C`.