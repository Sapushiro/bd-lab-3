import configparser
import os
import sys

import pandas as pd
from sklearn.model_selection import train_test_split

from src.logger import Logger

TEST_SIZE = 0.3
RANDOM_STATE = 0
SHOW_LOG = True

class DataMaker:

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)

        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config_path = os.path.join(os.getcwd(), "config.ini")

        self.project_path = os.path.join(os.getcwd(), "data")
        os.makedirs(self.project_path, exist_ok=True)

        self.data_path = os.path.join(self.project_path, "BankNote_Authentication.csv")
        self.data_clean_path = os.path.join(self.project_path, "BankNote_Authentication_clean.csv")

        self.X_path = os.path.join(self.project_path, "BankNote_X.csv")
        self.y_path = os.path.join(self.project_path, "BankNote_y.csv")

        self.train_path = [os.path.join(self.project_path, "Train_BankNote_X.csv"),
                           os.path.join(self.project_path, "Train_BankNote_y.csv")]

        self.test_path = [os.path.join(self.project_path, "Test_BankNote_X.csv"),
                          os.path.join(self.project_path, "Test_BankNote_y.csv")]

        self.log.info("DataMaker is ready")

    def prepare_data(self) -> None:
        dataset = pd.read_csv(self.data_path)

        initial_size = len(dataset)
        dataset = dataset.drop_duplicates().reset_index(drop=True)
        removed_duplicates = initial_size - len(dataset)

        dataset.to_csv(self.data_clean_path, index=False)

        self.log.info(
            "Dataset preparation: %s duplicates removed",
            removed_duplicates,
        )

        self.config["CLEAN DATA"] = {
            "clean_dataset" : os.path.relpath(self.data_clean_path, start=os.getcwd())}

    def get_data(self) -> None:
        dataset = pd.read_csv(self.data_clean_path)

        X = dataset.iloc[:, :4].copy()
        y = dataset.iloc[:, 4].copy()

        X.to_csv(self.X_path, index=False)
        y.to_csv(self.y_path, index=False)

        self.log.info("X and y data is ready")

        self.config["DATA"] = {
            "X_data": os.path.relpath(self.X_path, start=os.getcwd()),
            "y_data": os.path.relpath(self.y_path, start=os.getcwd()),
        }

    def split_data(self, test_size=TEST_SIZE) -> None:
        X = pd.read_csv(self.X_path)
        y = pd.read_csv(self.y_path)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
        )

        self.save_splitted_data(X_train, self.train_path[0])
        self.save_splitted_data(y_train, self.train_path[1])
        self.save_splitted_data(X_test, self.test_path[0])
        self.save_splitted_data(y_test, self.test_path[1])
        self.config["SPLIT DATA"] = {
            "X_train": os.path.relpath(self.train_path[0], start=os.getcwd()),
            "y_train": os.path.relpath(self.train_path[1], start=os.getcwd()),
            "X_test": os.path.relpath(self.test_path[0], start=os.getcwd()),
            "y_test": os.path.relpath(self.test_path[1], start=os.getcwd()),
        }
        self.log.info("Train and test data is ready")
        with open(self.config_path, "w") as configfile:
            self.config.write(configfile)

    def save_splitted_data(self, df: pd.DataFrame, path: str) -> None:
        df = df.reset_index(drop=True)
        df.to_csv(path, index=False)
        self.log.info('%s is saved', path)

    def run(self) -> None:
        self.prepare_data()
        self.get_data()
        self.split_data()

if __name__ == "__main__":
    data_maker = DataMaker()
    try:
        data_maker.run()
    except FileNotFoundError:
        data_maker.log.exception("Data preparation failed")
        sys.exit(1)
