import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

from src.api import app, get_predictor, get_database

class TestAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.predictor = Mock()
        self.database = Mock()

        app.dependency_overrides[get_predictor] = lambda: self.predictor
        app.dependency_overrides[get_database] = lambda: self.database

        self.client = TestClient(app)

        self.valid_features = {
            "variance": 0.0,
            "skewness": 0.0,
            "curtosis": -2.8999,
            "entropy": -0.4445,
        }

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_health_returns_ok(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok"}
        )

    def test_predict_returns_authentic_label(self):
        self.predictor.predict.return_value = 0

        response = self.client.post(
            "/predict",
            json=self.valid_features,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "prediction": 0,
                "label": "authentic",
            },
        )

    def test_predict_returns_forged_label(self):
        self.predictor.predict.return_value = 1

        response = self.client.post(
            "/predict",
            json=self.valid_features,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "prediction": 1,
                "label": "forged",
            },
        )

    def test_predict_passes_features_to_predictor(self):
        self.predictor.predict.return_value = 0

        self.client.post(
            "/predict",
            json=self.valid_features,
        )

        self.predictor.predict.assert_called_once_with(
            self.valid_features
        )

    def test_predict_returns_422_when_feature_is_missing(self):
        invalid_features = {
            "variance": 0.0,
            "skewness": 0.0,
            "curtosis": -2.8999,
        }

        response = self.client.post(
            "/predict",
            json=invalid_features,
        )

        self.assertEqual(response.status_code, 422)
        self.predictor.predict.assert_not_called()

    def test_predict_returns_422_for_extra_feature(self):
        invalid_features = {
            **self.valid_features,
            "unknown_feature": 10.0,
        }

        response = self.client.post(
            "/predict",
            json=invalid_features,
        )

        self.assertEqual(response.status_code, 422)
        self.predictor.predict.assert_not_called()

    def test_predict_returns_422_for_invalid_feature_type(self):
        invalid_features = {
            **self.valid_features,
            "variance": "not-a-number",
        }

        response = self.client.post(
            "/predict",
            json=invalid_features,
        )

        self.assertEqual(response.status_code, 422)
        self.predictor.predict.assert_not_called()


if __name__ == "__main__":
    unittest.main()
