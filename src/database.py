import os
import re
from datetime import datetime

from sqlalchemy import (
    DateTime,
    Float,
    Integer,
    String,
    URL,
    create_engine,
    func,
    text,
    select
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

import hvac


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )
    variance: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    skewness: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    curtosis: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    entropy: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )
    prediction: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    label: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        nullable=False,
        server_default=func.sysdatetime(),
    )


class Database:
    def __init__(self) -> None:
        database_config = self._get_database_config()

        self.host = database_config["host"]
        self.port = database_config["port"]
        self.database_name = database_config["database"]
        self.user = database_config["username"]
        self.password = database_config["password"]

        if not re.fullmatch(r"[A-Za-z0-9_]+", self.database_name):
            raise ValueError(
                "DB_NAME can contain only letters, numbers and underscores"
            )

        self.master_engine = self._create_engine("master")
        self.engine = self._create_engine(self.database_name)

    def _get_vault_token(self) -> str:
        token_file = os.environ["VAULT_TOKEN_FILE"]

        with open(token_file, "r") as file:
            return file.read().strip()

    def _get_database_config(self) -> dict:
        client = hvac.Client(
            url=os.environ["VAULT_ADDR"],
            token=self._get_vault_token()
        )

        secret = client.secrets.kv.v2.read_secret_version(
            path="database",
            mount_point="secret"
        )

        return secret["data"]["data"]


    def _create_engine(self, database_name: str) -> Engine:
        connection_url = URL.create(
            drivername="mssql+pyodbc",
            username=self.user,
            password=self.password,
            host=self.host,
            port=self.port,
            database=database_name,
            query={
                "driver": "ODBC Driver 18 for SQL Server",
                "Encrypt": "yes",
                "TrustServerCertificate": "yes",
            },
        )

        return create_engine(
            connection_url,
            pool_pre_ping=True,
        )

    def check_connection(self) -> bool:
        with self.master_engine.connect() as connection:
            result = connection.execute(
                text("SELECT 1")
            ).scalar_one()

        return result == 1

    def create_database(self) -> None:
        with self.master_engine.connect().execution_options(
            isolation_level="AUTOCOMMIT"
        ) as connection:
            database_exists = connection.execute(
                text(
                    """
                    SELECT 1
                    FROM sys.databases
                    WHERE name = :database_name
                    """
                ),
                {
                    "database_name": self.database_name
                },
            ).scalar_one_or_none()

            if database_exists is None:
                connection.exec_driver_sql(
                    f"CREATE DATABASE [{self.database_name}]"
                )

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    def initialize(self) -> None:
        self.create_database()
        self.create_tables()

    def save_prediction(self, features: dict[str, float], prediction: int, label: str) -> int:
        prediction_record = Prediction(
            variance=features["variance"],
            skewness=features["skewness"],
            curtosis=features["curtosis"],
            entropy=features["entropy"],
            prediction=prediction,
            label=label
        )

        with Session(self.engine) as session:
            session.add(prediction_record)
            session.commit()
            session.refresh(prediction_record)

            return prediction_record.id

    def get_predictions(self) -> list[Prediction]:
        with Session(self.engine) as session:
            predictions = session.scalars(
                select(Prediction).order_by(
                    Prediction.id.desc()
                )
            ).all()

            return list(predictions)