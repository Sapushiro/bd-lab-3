import configparser
import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.preprocess import DataMaker

class TestDataMaker(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()

        self.cwd_patcher = patch("os.getcwd", return_value=self.temp_dir.name)
        self.cwd_patcher.start()

        self.data_maker = DataMaker()
        self.test_dataset = pd.DataFrame(
            {
                "variance": [3.6, 4.5, -2.3, 3.6, 1.2, -1.5, 2.7, -3.1, 0.8, 5.0],
                "skewness": [8.6, 8.1, -3.7, 8.6, 2.5, -1.2, 4.4, -5.0, 1.1, 7.2],
                "curtosis": [-2.8, -2.4, 5.6, -2.8, 1.5, 3.2, -1.1, 6.4, 0.5, -3.0],
                "entropy": [-0.4, -1.7, -0.9, -0.4, 0.3, 1.1, -2.2, 0.7, -0.6, -1.4],
                "class": [0, 0, 1, 0, 0, 1, 0, 1, 1, 1],
            }
        )

        self.test_dataset.to_csv(self.data_maker.data_path, index=False)

    def tearDown(self) -> None:
        self.cwd_patcher.stop()
        self.temp_dir.cleanup()

    def test_prepare_data_removes_duplicates(self):
        self.data_maker.prepare_data()

        clean_dataset = pd.read_csv(self.data_maker.data_clean_path)

        self.assertEqual(len(clean_dataset), 9)
        self.assertEqual(clean_dataset.duplicated().sum(), 0)

        self.assertIn("CLEAN DATA", self.data_maker.config)
        self.assertEqual(
            self.data_maker.config["CLEAN DATA"]["clean_dataset"],
            os.path.relpath(
                self.data_maker.data_clean_path,
                start=self.temp_dir.name,
            ),
        )

    def test_get_data_splits_features_and_target(self):
        self.data_maker.prepare_data()
        self.data_maker.get_data()

        X = pd.read_csv(self.data_maker.X_path)
        y = pd.read_csv(self.data_maker.y_path)

        self.assertEqual(
            list(X.columns),
            ["variance", "skewness", "curtosis", "entropy"],
        )
        self.assertEqual(list(y.columns), ["class"])

        self.assertEqual(len(X), 9)
        self.assertEqual(len(y), 9)

        self.assertIn("DATA", self.data_maker.config)
        self.assertEqual(
            self.data_maker.config["DATA"]["x_data"],
            os.path.relpath(
                self.data_maker.X_path,
                start=self.temp_dir.name,
            ),
        )
        self.assertEqual(
            self.data_maker.config["DATA"]["y_data"],
            os.path.relpath(
                self.data_maker.y_path,
                start=self.temp_dir.name,
            ),
        )

    def test_split_data_creates_train_and_test_files(self):
        self.data_maker.prepare_data()
        self.data_maker.get_data()
        self.data_maker.split_data(test_size=0.3)

        X_train = pd.read_csv(self.data_maker.train_path[0])
        y_train = pd.read_csv(self.data_maker.train_path[1])
        X_test = pd.read_csv(self.data_maker.test_path[0])
        y_test = pd.read_csv(self.data_maker.test_path[1])

        # После удаления дубликата остаётся 9 строк:
        # 6 попадает в train и 3 — в test.
        self.assertEqual(len(X_train), 6)
        self.assertEqual(len(y_train), 6)
        self.assertEqual(len(X_test), 3)
        self.assertEqual(len(y_test), 3)

        self.assertTrue(os.path.exists(self.data_maker.train_path[0]))
        self.assertTrue(os.path.exists(self.data_maker.train_path[1]))
        self.assertTrue(os.path.exists(self.data_maker.test_path[0]))
        self.assertTrue(os.path.exists(self.data_maker.test_path[1]))

        self.assertIn("SPLIT DATA", self.data_maker.config)

    def test_save_splitted_data_resets_index(self):
        dataframe = pd.DataFrame(
            {"value": [10, 20]},
            index=[5, 8],
        )
        save_path = os.path.join(self.temp_dir.name, "saved.csv")

        self.data_maker.save_splitted_data(dataframe, save_path)

        saved_dataframe = pd.read_csv(save_path)

        self.assertEqual(saved_dataframe.index.tolist(), [0, 1])
        self.assertEqual(saved_dataframe["value"].tolist(), [10, 20])
        self.assertNotIn("index", saved_dataframe.columns)

    def test_run_creates_all_expected_files(self):
        self.data_maker.run()

        expected_files = [
            self.data_maker.data_clean_path,
            self.data_maker.X_path,
            self.data_maker.y_path,
            self.data_maker.train_path[0],
            self.data_maker.train_path[1],
            self.data_maker.test_path[0],
            self.data_maker.test_path[1]
        ]

        for file_path in expected_files:
            with self.subTest(file_path=file_path):
                self.assertTrue(os.path.exists(file_path))

        self.assertTrue(os.path.exists(self.data_maker.config_path))

        config = configparser.ConfigParser()
        config.read(self.data_maker.config_path)

        self.assertIn("CLEAN DATA", config)
        self.assertIn("DATA", config)
        self.assertIn("SPLIT DATA", config)


if __name__ == "__main__":
    unittest.main()

