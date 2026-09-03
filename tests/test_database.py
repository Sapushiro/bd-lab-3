import os
import unittest

import httpx

API_URL = os.getenv(
    "API_URL",
    "http://localhost:8000",
)

class TestDatabase(unittest.TestCase):
    def test_prediction_is_saved_to_database(self) -> None:
        features = {
            "variance": 1.234567,
            "skewness": -2.345678,
            "curtosis": 3.456789,
            "entropy": -4.567891,
        }

        history_before_response = httpx.get(
            f"{API_URL}/predictions",
            timeout=10.0
        )
        self.assertEqual(history_before_response.status_code, 200)

        ids_before = {record["id"] for record in history_before_response.json()}

        prediction_response = httpx.post(
            f"{API_URL}/predict",
            json=features,
            timeout=10.0
        )
        self.assertEqual(prediction_response.status_code, 200)
        prediction_result = prediction_response.json()

        history_after_response = httpx.get(
            f"{API_URL}/predictions",
            timeout=10.0
        )
        self.assertEqual(history_after_response.status_code, 200)
        new_records = [record for record in history_after_response.json()
                       if record["id"] not in ids_before]

        self.assertEqual(len(new_records), 1)
        saved_record = new_records[0]

        self.assertEqual(saved_record["variance"], features["variance"])
        self.assertEqual(saved_record["skewness"], features["skewness"])
        self.assertEqual(saved_record["curtosis"], features["curtosis"])
        self.assertEqual(saved_record["entropy"], features["entropy"])

        self.assertEqual(saved_record["prediction"], prediction_result["prediction"])
        self.assertEqual(saved_record["label"], prediction_result["label"])
        self.assertIsNotNone(saved_record["created_at"])


if __name__ == "__main__":
    unittest.main()

