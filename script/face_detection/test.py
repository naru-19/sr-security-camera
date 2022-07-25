# streamlit run test.py

import streamlit as st
from numpysocket import NumpySocket
import cv2
from narutils import *
import numpy as np
from PIL import Image
from face_detector import FaceDetector


def main():
    face_detector = FaceDetector(upscale_factor=1)
    print("start")
    st.markdown("# Camera")
    np_sock = NumpySocket()
    np_sock.startServer(41234)
    start = now()
    time_loc = st.empty()
    detect_image_loc = st.empty()
    crop_image_loc = st.empty()

    while True:
        image_raw = np_sock.recieve()
        time_loc.text(str(now(start)))
        cropped_faces, bboxes = face_detector.crop_faces(image_raw)
        if len(cropped_faces) > 0:
            detect_img = cv2.drawContours(image_raw, bboxes, -1, (0, 255, 0), thickness=2)
            detect_image_rgb = Image.fromarray(cv2.cvtColor(detect_img, cv2.COLOR_BGR2RGB))
            crop_image_rgb = Image.fromarray(cv2.cvtColor(cropped_faces[0], cv2.COLOR_BGR2RGB))
            detect_image_loc.image(detect_image_rgb)
            crop_image_loc.image(crop_image_rgb)

        if cv2.waitKey() & 0xFF == ord("q"):
            break

    np_sock.close()


if __name__ == "__main__":
    main()
