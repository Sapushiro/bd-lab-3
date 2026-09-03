import configparser
import os
import pandas as pd
import pickle

from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from src.logger import Logger
import sys
import traceback

SHOW_LOG = True

class Model:
    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")

        self.X_train = pd.read_csv(
            self.config["SPLIT DATA"]["X_train"]
        )
        self.y_train = pd.read_csv(
            self.config["SPLIT DATA"]["y_train"]
        ).squeeze("columns")
        self.X_test = pd.read_csv(
            self.config["SPLIT DATA"]["X_test"]
        )
        self.y_test = pd.read_csv(
            self.config["SPLIT DATA"]["y_test"]
        ).squeeze("columns")

        self.project_path = os.path.join(os.getcwd(), "experiments")
        os.makedirs(self.project_path, exist_ok=True)
        self.model_path = os.path.join(self.project_path, "log_reg.sav")
        self.log.info("Model is ready")

    def log_reg(self, predict=False) -> None:
        classifier = LogisticRegression()
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            self.log.info("Logistic Regression accuracy: %.4f", accuracy)
        params = {"path": os.path.relpath(self.model_path, start=os.getcwd())}
        self.save_model(classifier, self.model_path, "LOG_REG", params)

    def save_model(self, classifier, path: str, name: str, params: dict) -> None:
        self.config[name] = params
        with open("config.ini", "w") as configfile:
            self.config.write(configfile)
        with open(path, "wb") as model_file:
            pickle.dump(classifier, model_file)
        self.log.info("%s is saved", path)


if __name__ == "__main__":
    model = Model()
    model.log_reg(predict=True)