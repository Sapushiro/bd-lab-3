import configparser
import traceback

import pandas as pd
import pickle
import sys

from src.logger import Logger

SHOW_LOG = True

FEATURE_NAMES = [
    "variance",
    "skewness",
    "curtosis",
    "entropy",
]

class Predictor:
    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.log = logger.get_logger(__name__)

        self.config = configparser.ConfigParser()
        self.config.read("config.ini")

        self.model_path = self.config["LOG_REG"]["path"]
        try:
            with open(self.model_path, "rb") as model_file:
                self.classifier = pickle.load(model_file)
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)

        self.log.info("Model is loaded from %s", self.model_path)

    def predict(self, features: dict[str, float]) -> int:
        input_data = pd.DataFrame([features], columns=FEATURE_NAMES)

        prediction = self.classifier.predict(input_data)
        predicted_class = int(prediction[0])

        self.log.info("Predicted class: %s", predicted_class)
        return predicted_class