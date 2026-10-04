"""Manual tests for image pipeline"""

import os
import sys
import numpy as np
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarkerResult,
)
import cv2 as cv
from src.image_pipeline import (
    LABEL_TO_LETTER,
    LETTER_TO_LABEL,
    BASE_OPTIONS,
    HANDEDNESS_INDEX_TO_NAME,
)


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
    if os.path.exists(output_path):
        confirm = input("File already exists. Overide? [Y/n] ")
        if confirm not in ("Y", "y"):
            print("Exiting...")
            sys.exit()

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


def draw_landmarks(
    img: np.ndarray, landmarks: list[NormalizedLandmark], handedness: str
):
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

    # Add text next to landmarks indicating handedness
    height, width, _ = annotated.shape
    x = [landmark.x for landmark in landmarks]
    y = [landmark.y for landmark in landmarks]
    text_x = int(min(x) * width)
    text_y = int(min(y) * height)

    cv.putText(
        annotated,
        handedness,
        (text_x, text_y),
        cv.FONT_HERSHEY_COMPLEX_SMALL,
        1,
        (0, 136, 255),
        1,
        cv.LINE_AA,
    )

    return annotated


def draw_extracted_landmarks(path: str) -> None:
    """Draws extracted landmarks on original picture to ensure validity"""
    with open(path, "r") as f:
        for line in f:
            form = line.strip().split(",")
            landmarks = format_landmarks(form[2:])
            img = cv.imread(f"dataset/sample/{LABEL_TO_LETTER[int(form[0])]}.jpg")

            as_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=as_rgb)
            annotated = draw_landmarks(
                mp_img.numpy_view(), landmarks, HANDEDNESS_INDEX_TO_NAME[int(form[1])]
            )
            as_bgr = cv.cvtColor(annotated, cv.COLOR_RGB2BGR)

            cv.imshow("Image", as_bgr)
            cv.waitKey(0)
            cv.destroyAllWindows()


if __name__ == "__main__":
    n = input(
        "Save unnormalized landmarks from images (0) or check extracted unnormalized landmarks (1, default)? "
    )
    if n == "0":
        landmarks_to_csv(Path("dataset/sample/"))
    elif n in ("1", ""):
        draw_extracted_landmarks(Path("dataset/unnormalized.csv"))
    else:
        print("Invalid input. Exiting...")
