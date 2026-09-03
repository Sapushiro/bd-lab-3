import unittest
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src.predict import FEATURE_NAMES, Predictor


class TestPredictor(unittest.TestCase):

    def setUp(self) -> None:
        self.predictor = Predictor.__new__(Predictor)
        self.predictor.classifier = Mock()
        self.predictor.log = Mock()

        self.features = {
            "variance": 0.0,
            "skewness": 0.0,
            "curtosis": -2.8999,
            "entropy": -0.4445,
        }

    def test_predict_returns_classifier_result_as_integer(self):
        self.predictor.classifier.predict.return_value = np.array([1])

        result = self.predictor.predict(self.features)

        self.assertEqual(result, 1)
        self.assertIsInstance(result, int)

    def test_predict_calls_classifier_once(self):
        self.predictor.classifier.predict.return_value = np.array([0])

        self.predictor.predict(self.features)

        self.predictor.classifier.predict.assert_called_once()

    def test_predict_passes_dataframe(self):
        self.predictor.classifier.predict.return_value = np.array([0])

        self.predictor.predict(self.features)

        call_arguments = self.predictor.classifier.predict.call_args
        input_data = call_arguments.args[0]

        self.assertIsInstance(input_data, pd.DataFrame)

    def test_predict_passes_one_object(self):
        self.predictor.classifier.predict.return_value = np.array([0])

        self.predictor.predict(self.features)

        call_arguments = self.predictor.classifier.predict.call_args
        input_data = call_arguments.args[0]

        self.assertEqual(input_data.shape, (1, 4))

    def test_predict_preserves_feature_order(self):
        self.predictor.classifier.predict.return_value = np.array([0])

        self.predictor.predict(self.features)

        call_arguments = self.predictor.classifier.predict.call_args
        input_data = call_arguments.args[0]

        self.assertEqual(
            list(input_data.columns),
            FEATURE_NAMES,
        )

    def test_predict_preserves_feature_values(self):
        self.predictor.classifier.predict.return_value = np.array([0])

        self.predictor.predict(self.features)

        call_arguments = self.predictor.classifier.predict.call_args
        input_data = call_arguments.args[0]

        self.assertEqual(
            input_data.iloc[0].to_dict(),
            self.features,
        )

    def test_predict_writes_result_to_log(self):
        self.predictor.classifier.predict.return_value = np.array([1])

        self.predictor.predict(self.features)

        self.predictor.log.info.assert_called_once_with(
            "Predicted class: %s",
            1,
        )


if __name__ == "__main__":
    unittest.main()