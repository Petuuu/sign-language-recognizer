"""Manual tests for image pipeline"""

# pylint: disable=duplicate-code

import os
import sys
from pathlib import Path
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
import cv2 as cv
from src.image_pipeline import LABEL_TO_LETTER, LETTER_TO_LABEL, BASE_OPTIONS
from src.helpers import check_file_exists


def format_landmarks(points: list[float]) -> list[NormalizedLandmark]:
    """Formats a list of coordinates (floats) into a list of NormalizedLandmark objects

    Args:
        points (list[float]): coordinates

    Returns:
        landmarks (list[NormalizedLandmark]): coordinates as NormalizedLandmark objects
    """
    landmarks = []
    for x in range(0, len(points), 3):
        landmarks.append(
            NormalizedLandmark(
                x=float(points[x]),
                y=float(points[x + 1]),
                z=float(points[x + 2]),
                visibility=None,
                presence=None,
                name=None,
            )
        )
    return landmarks


def landmarks_to_csv(path: str, output_path: str) -> None:
    """Original landmarks_to_csv that doesn't normalize landmarks

    Args:
        path (str): path to an image directory or file
        output_path (str): path to output CSV file. Defaults to "dataset/unnormalized.csv"
    """
    check_file_exists(output_path)

    options = vision.HandLandmarkerOptions(
        base_options=BASE_OPTIONS, num_hands=2, running_mode=vision.RunningMode.IMAGE
    )
    with vision.HandLandmarker.create_from_options(options) as detector:
        with open(output_path, "w", encoding="utf-8") as f:
            if os.path.isdir(path):
                for file in path.rglob("*"):
                    if file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                        continue

                    # OpenCV uses BGR for images, while MediaPipe uses RGB
                    # -> colors need to be converted
                    img = cv.imread(str(file))
                    as_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=as_rgb)
                    res = detector.detect(mp_img)

                    # format result and save to file
                    if len(res.hand_landmarks) > 1:
                        raise ValueError("Only one hand allowed in a single picture")
                    label = LETTER_TO_LABEL[file.stem[0].upper()]
                    handedness = res.handedness[0][0].index
                    points = np.array(
                        [[lm.x, lm.y, lm.z] for lm in res.hand_landmarks[0]],
                        dtype=np.float64,
                    )
                    points = ",".join([str(x) for x in points.flatten().tolist()])
                    f.write(f"{label},{handedness},{points}\n")

            else:
                if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    print("Given file is not an image. Exiting...")
                    sys.exit()

                img = cv.imread(str(path))
                as_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=as_rgb)
                res = detector.detect(mp_img)

                if len(res.hand_landmarks) > 1:
                    raise ValueError("Only one hand allowed in a single picture")
                label = LETTER_TO_LABEL[path.stem[0].upper()]
                handedness = res.handedness[0][0].index
                points = np.array(
                    [[lm.x, lm.y, lm.z] for lm in res.hand_landmarks[0]],
                    dtype=np.float64,
                )
                points = ",".join([str(x) for x in points.flatten().tolist()])
                f.write(f"{label},{handedness},{points}\n")


def draw_landmarks(img: np.ndarray, landmarks: list[NormalizedLandmark]):
    """Original draw_landmarks function that draws landmarks from a list of
    NormalizedLandmark objects rather than a HandLandmarkerResult object

        Args:
            img (numpy.ndarray): image to draw landmarks onto
            landmarks (list[NormalizedLandmark]): normalized landmarks

        Returns:
            annotated (numpy.ndarray): copy of original image including landmarks and handedness
    """

    mp_hands = vision.HandLandmarksConnections
    mp_drawing = vision.drawing_utils
    mp_drawing_styles = vision.drawing_styles
    annotated = np.copy(img)

    mp_drawing.draw_landmarks(
        annotated,
        landmarks,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing_styles.get_default_hand_landmarks_style(),
        mp_drawing_styles.get_default_hand_connections_style(),
    )

    return annotated


def draw_extracted_landmarks(path: str) -> None:
    """Draw extracted landmarks on original picture to ensure validity

    Args:
        path (str): Path to file from which landmarks are read
    """
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            form = line.strip().split(",")
            landmarks = format_landmarks(form[2:])
            img = cv.imread(f"dataset/sample/{LABEL_TO_LETTER[int(form[0])]}.jpg")

            as_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=as_rgb)
            annotated = draw_landmarks(mp_img.numpy_view(), landmarks)
            as_bgr = cv.cvtColor(annotated, cv.COLOR_RGB2BGR)

            cv.imshow("Image", as_bgr)
            cv.waitKey(0)
            cv.destroyAllWindows()


if __name__ == "__main__":
    n = input(
        "Save unnormalized landmarks from images (0) or check extracted unnormalized"
        "landmarks (1, default)? "
    )
    if n == "0":
        main_path = Path("dataset/sample/")
        main_output_path = Path("dataset/unnormalized.csv")
        landmarks_to_csv(main_path, main_output_path)
    elif n in ("1", ""):
        draw_extracted_landmarks(Path("dataset/unnormalized.csv"))
    else:
        print("Invalid input. Exiting...")
