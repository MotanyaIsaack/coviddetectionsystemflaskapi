# from flask import Flask, escape, request
# from flask import render_template

# app = Flask(__name__)

# @app.route('/')
# def hello_there(name = None):
#     return render_template(
#         "hello_there.html",
#         name=name,
#         # date=datetime.now()
#     )

import json
import numpy as np

#
from keras.models import load_model

# OR
# from tensorflow.keras.models import load_model
#
from flask import Flask, escape, request
from flask_cors import CORS, cross_origin
import cv2
from PIL import Image

import helper

flask_app = Flask(__name__)
CORS(flask_app)

model_path = "model/covid_final_model.h5"
DIAGNOSIS_MESSAGES = ["Pneumonia detected", "Covid19 detected", "Normal lungs detected"]


@flask_app.route("/test")
def default():
    return "api is running"


@flask_app.route("/", methods=["GET"])
def index_page():
    return_data = {"error": "0", "message": "Successful"}
    return flask_app.response_class(
        response=json.dumps(return_data), mimetype="application/json"
    )


@flask_app.route("/classify", methods=["POST"])
def classify_patient_xray_image():
    try:
        print(request.files)
        if (
            "patient_xray_image" in request.files
            and request.files["patient_xray_image"] is not None
        ):
            print("Good to go")
            patient_xray_image = request.files["patient_xray_image"]
            print(patient_xray_image)

            is_successful, preprocessed_image = helper.preprocess_img(
                patient_xray_image
            )
            if is_successful:
                # Load the covid_best_model
                covid_final_model = load_model("model/to_use_final_model.h5")
                # Do the prediction
                prediction = covid_final_model.predict(preprocessed_image)
                _prediction = round(prediction[0][0] * 100, 3)
                if _prediction > 50:
                    _prediction = DIAGNOSIS_MESSAGES[1]
                    return_data = {
                        "error": "0",
                        "message": "Successful",
                        "classification": DIAGNOSIS_MESSAGES[1],
                    }

                elif _prediction < 50:
                    _prediction = DIAGNOSIS_MESSAGES[2]
                    return_data = {
                        "error": "0",
                        "message": "Successful",
                        "classification": DIAGNOSIS_MESSAGES[2],
                    }
            else:
                return_data = {"error": "1", "message": "Image preprocessing error"}
        else:
            return_data = {"error": "1", "message": "Invalid parameters"}
    except Exception as e:
        return_data = {
            "error": "1",
            "message": f"[Error] : {e}",
            "message": f"[Error] : {e}",
        }
    return flask_app.response_class(
        response=json.dumps(return_data), mimetype="application/json"
    )


if __name__ == "__main__":
    flask_app.run(debug=False, host="0.0.0.0", port=5000)
