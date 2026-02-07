import tensorflow as tf
import numpy as np
import json
from PIL import Image


class FracturePredictor:

    def __init__(self):

        self.img_size = (224, 224)

        # -------- Load Detection Model --------
        self.detect_model = tf.keras.models.load_model(
            "model/fracture_detection_model.h5"
        )

        # -------- Load Classification Model --------
        self.classify_model = tf.keras.models.load_model(
            "model/fracture_classification_model.h5"
        )

        # -------- Load Class Maps --------
        with open("model/detect_classes.json") as f:
            self.detect_classes = json.load(f)

        with open("model/classify_classes.json") as f:
            self.classify_classes = json.load(f)

        # Reverse dictionary
        self.detect_labels = {v: k for k, v in self.detect_classes.items()}
        self.classify_labels = {v: k for k, v in self.classify_classes.items()}

        print("✅ Dual AI Models Loaded Successfully")


    # ---------------------------------------------
    # IMAGE PREPROCESSING
    # ---------------------------------------------
    def preprocess_image(self, image_path):

        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.img_size)

        img_array = np.array(img).astype("float32") / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array


    # ---------------------------------------------
    # MAIN PREDICTION FUNCTION
    # ---------------------------------------------
    def predict(self, image_path):

        img_array = self.preprocess_image(image_path)

        # -------- Stage 1: Fracture Detection --------
        detect_pred = self.detect_model.predict(img_array)
        detect_idx = np.argmax(detect_pred)
        detect_conf = float(np.max(detect_pred))

        detect_result = self.detect_labels[detect_idx]

        # -------- If Normal --------
        if detect_result == "normal":

            return {
                "fracture_type": "No Fracture",
                "severity": "Minor",
                "confidence": round(detect_conf * 100, 2)
            }

        # -------- Stage 2: Fracture Classification --------
        classify_pred = self.classify_model.predict(img_array)
        classify_idx = np.argmax(classify_pred)
        classify_conf = float(np.max(classify_pred))

        fracture_type = self.classify_labels[classify_idx]

        # -------- Severity Logic --------
        if classify_conf > 0.85:
            severity = "Severe"
        elif classify_conf > 0.65:
            severity = "Moderate"
        else:
            severity = "Minor"

        return {
            "fracture_type": fracture_type,
            "severity": severity,
            "confidence": round(classify_conf * 100, 2)
        }


    # ---------------------------------------------
    # CLASS LIST
    # ---------------------------------------------
    def get_classes(self):

        return {
            "Detection Classes": list(self.detect_classes.keys()),
            "Fracture Classes": list(self.classify_classes.keys())
        }
