import unittest
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

from fastapi.testclient import TestClient

from src.api import app, get_database, get_predictor


class TestApiDatabase(unittest.TestCase):

    def setUp(self) -> None:
        self.predictor = Mock()
        self.database = Mock()

        app.dependency_overrides[get_predictor] = lambda: self.predictor
        app.dependency_overrides[get_database] = lambda: self.database

        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_predict_saves_result_to_database(self) -> None:
        features = {
            "variance": 3.6216,
            "skewness": 8.6661,
            "curtosis": -2.8073,
            "entropy": -0.44699,
        }

        self.predictor.predict.return_value = 0
        self.database.save_prediction.return_value = 1

        response = self.client.post(
            "/predict",
            json=features,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "prediction": 0,
                "label": "authentic",
            },
        )

        self.predictor.predict.assert_called_once_with(
            features
        )
        self.database.save_prediction.assert_called_once_with(
            features=features,
            prediction=0,
            label="authentic",
        )

    def test_get_predictions_returns_database_records(
        self,
    ) -> None:
        self.database.get_predictions.return_value = [
            SimpleNamespace(
                id=1,
                variance=3.6216,
                skewness=8.6661,
                curtosis=-2.8073,
                entropy=-0.44699,
                prediction=0,
                label="authentic",
                created_at=datetime(
                    2026, 9, 2, 10, 30, 0
                ),
            )
        ]

        response = self.client.get("/predictions")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            [
                {
                    "id": 1,
                    "variance": 3.6216,
                    "skewness": 8.6661,
                    "curtosis": -2.8073,
                    "entropy": -0.44699,
                    "prediction": 0,
                    "label": "authentic",
                    "created_at": "2026-09-02T10:30:00",
                }
            ],
        )

        self.database.get_predictions.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()