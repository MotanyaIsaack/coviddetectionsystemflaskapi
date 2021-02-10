import numpy as np
import cv2
from PIL import Image


def preprocess_img(patient_xray_image):
    try:
        image = cv2.imdecode(
            np.frombuffer(patient_xray_image.read(), np.uint8), cv2.IMREAD_GRAYSCALE
        )
        img = cv2.resize(image, (150, 150))
        img = np.dstack([img, img, img])
        img = img.astype("float32") / 255
        final_image = []
        final_image.append(img)
        final_image = np.array(final_image)
        return True, final_image
    except Exception as e:
        return False, str(e)