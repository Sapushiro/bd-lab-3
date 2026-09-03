import configparser
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from src.train import Model


class TestModel(unittest.TestCase):

    def setUp(self) -> None:
        self.model = Model.__new__(Model)
        self.model.config = configparser.ConfigParser()
        self.model.log = Mock()

        self.model.X_train = pd.DataFrame(
            {
                "variance": [1.0, 2.0, -1.0, -2.0],
                "skewness": [2.0, 3.0, -2.0, -3.0],
                "curtosis": [0.5, 1.5, -0.5, -1.5],
                "entropy": [1.0, 0.0, -1.0, 0.5],
            }
        )
        self.model.y_train = pd.Series([0, 0, 1, 1])

        self.model.X_test = pd.DataFrame(
            {
                "variance": [1.5, -1.5],
                "skewness": [2.5, -2.5],
                "curtosis": [1.0, -1.0],
                "entropy": [0.5, -0.5],
            }
        )
        self.model.y_test = pd.Series([0, 1])

        self.model.model_path = "experiments/log_reg.sav"

    @patch("src.train.LogisticRegression")
    def test_log_reg_trains_classifier(self, mock_logistic_reg):
        classifier = Mock()
        mock_logistic_reg.return_value = classifier
        self.model.save_model = Mock()

        self.model.log_reg()

        mock_logistic_reg.assert_called_once_with()
        classifier.fit.assert_called_once_with(
            self.model.X_train,
            self.model.y_train,
        )

    @patch("src.train.LogisticRegression")
    def test_log_reg_saves_trained_classifier(self, mock_logistic_reg):
        classifier = Mock()
        mock_logistic_reg.return_value = classifier
        self.model.save_model = Mock()

        self.model.log_reg()

        self.model.save_model.assert_called_once_with(
            classifier,
            self.model.model_path,
            "LOG_REG",
            {"path": self.model.model_path},
        )

    @patch("src.train.accuracy_score")
    @patch("src.train.LogisticRegression")
    def test_log_reg_predicts_when_predict_is_true(
        self,
        mock_logistic_reg,
        mock_accuracy_score,
    ):
        classifier = Mock()
        classifier.predict.return_value = [0, 1]

        mock_logistic_reg.return_value = classifier
        mock_accuracy_score.return_value = 1.0
        self.model.save_model = Mock()

        self.model.log_reg(predict=True)

        classifier.predict.assert_called_once_with(self.model.X_test)

        mock_accuracy_score.assert_called_once_with(
            self.model.y_test,
            [0, 1],
        )

        self.model.log.info.assert_called_once_with(
            "Logistic Regression accuracy: %.4f",
            1.0,
        )

    @patch("src.train.LogisticRegression")
    def test_log_reg_does_not_predict_by_default(
        self,
        mock_logistic_reg,
    ):
        classifier = Mock()
        mock_logistic_reg.return_value = classifier
        self.model.save_model = Mock()

        self.model.log_reg()

        classifier.predict.assert_not_called()

    @patch("src.train.LogisticRegression")
    def test_log_reg_exits_when_training_fails(self, mock_logistic_reg):
        classifier = Mock()
        classifier.fit.side_effect = ValueError("Training failed")
        mock_logistic_reg.return_value = classifier
        self.model.save_model = Mock()

        with self.assertRaises(SystemExit) as exception:
            self.model.log_reg()

        self.assertEqual(exception.exception.code, 1)
        self.model.log.error.assert_called_once()
        self.model.save_model.assert_not_called()

    def test_save_model_creates_model_file(self):
        classifier = {
            "name": "fake_classifier",
        }
        params = {
            "path": "model.sav",
        }

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.sav"
            current_directory = os.getcwd()

            try:
                os.chdir(directory)

                self.model.save_model(
                    classifier=classifier,
                    path=str(model_path),
                    name="LOG_REG",
                    params=params,
                )
            finally:
                os.chdir(current_directory)

            self.assertTrue(model_path.is_file())

    def test_save_model_preserves_classifier(self):
        classifier = {
            "name": "fake_classifier",
        }
        params = {
            "path": "model.sav",
        }

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.sav"
            current_directory = os.getcwd()

            try:
                os.chdir(directory)

                self.model.save_model(
                    classifier=classifier,
                    path=str(model_path),
                    name="LOG_REG",
                    params=params,
                )
            finally:
                os.chdir(current_directory)

            with model_path.open("rb") as model_file:
                saved_classifier = pickle.load(model_file)

            self.assertEqual(saved_classifier, classifier)

    def test_save_model_creates_config_section(self):
        classifier = {
            "name": "fake_classifier",
        }
        params = {
            "path": "model.sav",
        }

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.sav"
            config_path = Path(directory) / "config.ini"
            current_directory = os.getcwd()

            try:
                os.chdir(directory)

                self.model.save_model(
                    classifier=classifier,
                    path=str(model_path),
                    name="LOG_REG",
                    params=params,
                )
            finally:
                os.chdir(current_directory)

            config = configparser.ConfigParser()
            config.read(config_path)

            self.assertIn("LOG_REG", config.sections())
            self.assertEqual(
                config["LOG_REG"]["path"],
                "model.sav",
            )

    def test_save_model_writes_log(self):
        classifier = {
            "name": "fake_classifier",
        }
        params = {
            "path": "model.sav",
        }

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.sav"
            current_directory = os.getcwd()

            try:
                os.chdir(directory)

                self.model.save_model(
                    classifier=classifier,
                    path=str(model_path),
                    name="LOG_REG",
                    params=params,
                )
            finally:
                os.chdir(current_directory)

            self.model.log.info.assert_called_once_with(
                "%s is saved",
                str(model_path),
            )


if __name__ == "__main__":
    unittest.main()