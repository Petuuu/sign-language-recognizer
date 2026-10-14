# Overview

A computer vision program that recognizes finnish sign language letters (same as ASL with the addition of Ä, Ö, and Å) based on hand landmarks.

Only static letters can be recognizes, i.e. letters that don't require motion to sign. These are letters A-I and K-Y.

## Documentation

- [Specification document](docs/specification_document.md)
- [Implementation document](docs/implementation_document.md)
- [Test document](docs/test_document.md)
- [Pylint report](docs/pylint_report.txt)

### Weekly reports

- [Week 1](docs/weekly_progress/week_1.md)
- [Week 2](docs/weekly_progress/week_2.md)
- [Week 3](docs/weekly_progress/week_3.md)
- [Week 4](docs/weekly_progress/week_4.md)
- [Week 5](docs/weekly_progress/week_5.md)
- [Week 6](docs/weekly_progress/week_6.md)



# Usage

First, clone the repository, navigate to the project, and unzip `dataset.zip` to access the dataset used for training. The image and video mode may not work on the command line interface on other than Linux (confirmed not to work on MacOS). If this is the case, start the web server to test these.

## Command line

Create the virtual environment with (may take several minutes)
```bash
poetry install
```

Then, run the program from project root with
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

Start the development server on Docker with:
```bash
docker run -it --rm -p 8000:8000 \
  sign-language-recognizer python -m src.web
```

For local development without Docker, use:
```bash
poetry run python -m src.web
```

Open <http://localhost:8000>, click **Start camera**, and grant the browser camera permission. The browser captures frames and sends them to the container for MediaPipe landmark detection and classification. You can also select a JPG, JPEG, or PNG image to recognize it once. Stop the server with `Ctrl+C`.