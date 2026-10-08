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

First, clone the repository, navigate to the project, and unzip `dataset.zip` to access the dataset used for training.

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

## Docker container

If you don't have `Docker` already installed follow the installation process from https://docs.docker.com/engine/install/.

Build the Docker image from the project root (may take several minutes):
```bash
docker build -t sign-language-recognizer .
```

Run the interactive program with:
```bash
docker run -it --rm sign-language-recognizer python -m src.main
```

To use files from the host, mount a host directory:
```bash
docker run -it --rm \
  -v "$PWD/data:/usr/src/app/data" \
  sign-language-recognizer python -m src.main
```

Then enter a path under `/usr/src/app/data`, for example `data/sample/`.

### Camera and display access

The camera mode requires both the camera device and a display connection.
The following command is for Linux hosts using X11:
```bash
xhost +local:root

docker run -it --rm \
  --device=/dev/video0:/dev/video0 \
  -e DISPLAY="$DISPLAY" \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  sign-language-recognizer python -m src.main

xhost -local:root
```

Replace`/dev/video0` with the host camera device shown by `ls -l /dev/video*`, if necessary. The container exposes the device as `/dev/video0` because the application opens camera index `0`.

The `--device` option gives the container access to the camera. `DISPLAY` and the X11 socket mount allow OpenCV to create its preview window. The socket is mounted read-only because the application only needs to connect to the display server.

On Windows and macOS, Docker Desktop does not generally expose a host webcam to Linux containers through `--device`, and X11 is not provided by default. Run the camera mode directly on the host, or configure a separate camera/GUI bridge such as a Linux VM, WSL2, or an X server before using the equivalent container options. Docker image and landmark modes can still be used normally: if a GUI window is required, configure a display server and pass its `DISPLAY` connection to the container.
