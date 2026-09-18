import os
import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision.hand_landmarker import (
    HandLandmarkerResult,
)
import cv2 as cv

BASE_OPTIONS = mp.tasks.BaseOptions(model_asset_path="models/hand_landmarker.task")


def draw_landmarks(img: np.ndarray, res: HandLandmarkerResult) -> np.ndarray:
    """Draw landmarks and handedness (which hand is in picutre) onto given image

    Args:
        img (numpy.ndarray): image to draw landmarks onto
        res (HandLandmarkerResult): result of landmark detection

    Returns:
        annotated (numpy.ndarray): copy of original image including landmarks and handedness
    """

    mp_hands = vision.HandLandmarksConnections
    mp_drawing = vision.drawing_utils
    mp_drawing_styles = vision.drawing_styles
    annotated = np.copy(img)

    for i in range(len(res.hand_landmarks)):
        landmarks = res.hand_landmarks[i]
        handedness = res.handedness[i]

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
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv.FONT_HERSHEY_COMPLEX_SMALL,
            1,
            (0, 136, 255),
            1,
            cv.LINE_AA,
        )

    return annotated


def image_detect(path: str) -> None:
    """Detect landmarks from images and displays them. Image must be jpg, jpeg, or png

    Args:
        path (str): path to an image directory or file
    """

    options = vision.HandLandmarkerOptions(
        base_options=BASE_OPTIONS, num_hands=2, running_mode=vision.RunningMode.IMAGE
    )
    with vision.HandLandmarker.create_from_options(options) as detector:
        if os.path.isdir(path):
            for file in path.iterdir():
                if file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                    continue

                # OpenCV uses BGR for images, while MediaPipe uses RGB -> colors need to be converted
                img = cv.imread(str(file))
                as_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=as_rgb)
                res = detector.detect(mp_img)
                if res:
                    print(res)

                annotated = draw_landmarks(mp_img.numpy_view(), res)
                as_bgr = cv.cvtColor(annotated, cv.COLOR_RGB2BGR)
                cv.imshow("Image", as_bgr)
                cv.waitKey(0)
                cv.destroyAllWindows()

        else:
            if file.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                print("Given file is not an image")
                exit()

            img = cv.imread(str(file))
            as_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=as_rgb)
            res = detector.detect(mp_img)
            if res:
                print(res)

            annotated = draw_landmarks(mp_img.numpy_view(), res)
            as_bgr = cv.cvtColor(annotated, cv.COLOR_RGB2BGR)
            cv.imshow("Image", as_bgr)
            cv.waitKey(0)
            cv.destroyAllWindows()


def stream_detect() -> None:
    """Detect landmarks from video stream and display them"""

    options = vision.HandLandmarkerOptions(
        base_options=BASE_OPTIONS, num_hands=2, running_mode=vision.RunningMode.VIDEO
    )
    with vision.HandLandmarker.create_from_options(options) as detector:
        cap = cv.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        # OpenCV video capture loop
        while True:
            ret, frame = cap.read()

            if not ret:
                print("Cannot receive frame (stream end?). Exiting...")
                break

            as_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=as_rgb)
            timestamp_ms = time.monotonic_ns() // 1000000
            res = detector.detect_for_video(mp_img, timestamp_ms)
            if res.hand_landmarks:
                print(res)

            annotated = draw_landmarks(mp_img.numpy_view(), res)
            as_bgr = cv.cvtColor(annotated, cv.COLOR_RGB2BGR)
            cv.imshow("Image", as_bgr)
            if cv.waitKey(1) == ord("q"):
                break

        cap.release()
        cv.destroyAllWindows()
