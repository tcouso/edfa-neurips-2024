from typing import Tuple, List
from pathlib import Path
from dataclasses import dataclass
import pandas as pd
import pickle
import sqlite3
import os
import yaml

from src import paths


@dataclass
class PixelSimulation:
    index: Tuple[int, int]
    raw_signal: pd.Series
    true_signal: pd.Series
    pred_signal: pd.Series

    def __repr__(self) -> str:
        return f"PixelSimulation_{self.index[0]}_{self.index[1]}"

    def __str__(self) -> str:
        return f"PixelSimulation_{self.index[0]}_{self.index[1]}"


class PixelSimulationDB:
    def __init__(self, db_file: str):
        self.db_file = db_file
        self._initialize_db()

    def _initialize_db(self):
        """Initialize the SQLite database."""
        if not os.path.exists(self.db_file):
            try:
                with sqlite3.connect(self.db_file) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        """
                        CREATE TABLE IF NOT EXISTS simulations (
                            row INTEGER,
                            column INTEGER,
                            simulation BLOB,
                            PRIMARY KEY (row, column)
                        )
                    """
                    )
                    conn.commit()
            except Exception as e:
                print(f"Error initializing database: {e}")
                raise

    def _execute_query(self, query: str, params: tuple = (), commit: bool = False):
        """Execute a query with optional commit."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute(query, params)
                if commit:
                    conn.commit()
                return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            raise

    def insert_simulation(self, simulation: PixelSimulation):
        """
        Insert a PixelSimulation object into the database.

        Parameters
        ----------
        simulation : PixelSimulation
            The PixelSimulation object to insert.
        """
        serialized_simulation = pickle.dumps(simulation)
        try:
            self._execute_query(
                """
                INSERT OR REPLACE INTO simulations (row, column, simulation)
                VALUES (?, ?, ?)
            """,
                (simulation.index[0], simulation.index[1], serialized_simulation),
                commit=True,
            )
        except sqlite3.OperationalError as e:
            print(f"OperationalError during insert: {e}")
            raise
        except Exception as e:
            print(f"Unexpected error during insert: {e}")
            raise

    def get_simulation(self, index: Tuple[int, int]) -> PixelSimulation:
        """
        Retrieve a PixelSimulation object from the database by index.

        Parameters
        ----------
        index : Tuple[int, int]
            The (row, column) index of the PixelSimulation object.

        Returns
        -------
        PixelSimulation
            The retrieved PixelSimulation object.
        """
        rows = self._execute_query(
            """
            SELECT simulation FROM simulations
            WHERE row = ? AND column = ?
        """,
            (index[0], index[1]),
        )
        if rows:
            serialized_simulation = rows[0][0]
            return pickle.loads(serialized_simulation)
        else:
            raise ValueError(f"No simulation found for index {index}")

    def get_simulations_by_poly(self, poly_id: int) -> List[PixelSimulation]:
        """
        Retrieve all PixelSimulation objects from the database associated with a specific polygon.

        Parameters
        ----------
        poly_id : int
            The polygon id of the PixelSimulation objects to retrieve.

        Returns
        -------
        List[PixelSimulation]
            The list of retrieved PixelSimulation objects associated with the polygon.
        """
        rows = self._execute_query(
            """
            SELECT simulation FROM simulations
            WHERE row = ?
        """,
            (poly_id,),
        )
        return [pickle.loads(row[0]) for row in rows]


# DB instance global retrieval
with open(paths.config_dir("params.yaml"), "r") as file:
    params = yaml.safe_load(file)

esn_features_dim: int = params["esn_features_dim"]
selected_band: str = params["selected_band"]

db_filename = f"pixel_simulation_{selected_band}_{esn_features_dim}_features.db"
_db_instance = None


def get_db_instance(db_file: Path = paths.database_dir(db_filename)):
    global _db_instance
    if _db_instance is None:
        _db_instance = PixelSimulationDB(db_file)
    return _db_instance
