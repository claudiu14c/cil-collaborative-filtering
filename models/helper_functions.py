from typing import Callable, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import os
import torch


SEED = 42
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SUBMISSION_DIR = Path(__file__).resolve().parent.parent / "submissions"


def read_data_df(data_dir: str = DATA_DIR, seed: int = SEED, split: float = 0.25) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Reads in data and splits it into training and
    validation sets with a 75/25 split."""
    df = pd.read_csv(os.path.join(data_dir, "train_ratings.csv"))

    # Split sid_pid into sid and pid columns
    df[["sid", "pid"]] = df["sid_pid"].str.split("_", expand=True)
    df = df.drop("sid_pid", axis=1)
    df["sid"] = df["sid"].astype(int)
    df["pid"] = df["pid"].astype(int)

    if split == 0.0:
        return df
    # Split into train and validation dataset
    train_df, valid_df = train_test_split(df, test_size=split, random_state=seed)
    return train_df, valid_df


def read_tbr_df(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Reads the wishlist data."""
    tbr_df = pd.read_csv(os.path.join(data_dir, "train_tbr.csv"))
    return tbr_df


def read_full_training_data(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Reads the entire training dataset."""
    df = pd.read_csv(os.path.join(data_dir, "train_ratings.csv"))
    df[["sid", "pid"]] = df["sid_pid"].str.split("_", expand=True)
    df = df.drop("sid_pid", axis=1)
    df["sid"] = df["sid"].astype(int)
    df["pid"] = df["pid"].astype(int)
    return df


def read_data_matrix(df: pd.DataFrame) -> np.ndarray:
    """Returns matrix view of the training data, where columns are scientists (sid) and
    rows are papers (pid)."""

    return df.pivot(index="sid", columns="pid", values="rating").values


def read_full_data_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns matrix view of the training data, where columns are scientists
    (sid) and rows are papers (pid)."""

    pivot_df = df.pivot(index="sid", columns="pid", values="rating")
    row_ids = pivot_df.index.values
    col_ids = pivot_df.columns.values
    return pivot_df.values, row_ids, col_ids


def evaluate(valid_df: pd.DataFrame,
             pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> float:
    """
    Inputs:
        valid_df: Validation data, returned from read_data_df for example.
        pred_fn: Function that takes in arrays of sid and pid and outputs
           their rating predictions.

    Outputs: Validation RMSE
    """

    preds = pred_fn(valid_df["sid"].values, valid_df["pid"].values)
    return root_mean_squared_error(valid_df["rating"].values, preds)


def make_submission(pred_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
                    filename: str):
    """Makes a submission CSV file that can be submitted to kaggle.

    Inputs:
        pred_fn: Function that takes in arrays of sid and pid and
           outputs a score.
        filename: File to save the submission to.
    """

    df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    # Get sids and pids
    sid_pid = df["sid_pid"].str.split("_", expand=True)
    sids = sid_pid[0]
    pids = sid_pid[1]
    sids = sids.astype(int).values
    pids = pids.astype(int).values

    df["rating"] = pred_fn(sids, pids)
    directory = SUBMISSION_DIR / filename
    df.to_csv(directory, index=False)


def clip_and_make_submission(pred_fn_callable: Callable[[np.ndarray, np.ndarray], np.ndarray],
                             filename: str, data_dir: os.PathLike = DATA_DIR):
    """Creates a submission CSV file using the prediction function."""
    df_sub = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    sid_pid_split = df_sub["sid_pid"].str.split("_", expand=True)
    sids_sub_vals = sid_pid_split[0].astype(int).values
    pids_sub_vals = sid_pid_split[1].astype(int).values
    predictions = pred_fn_callable(sids_sub_vals, pids_sub_vals)
    df_sub["rating"] = np.clip(predictions, 1.0, 5.0)
    df_sub.to_csv(filename, index=False)
    print(f"Submission file created: {filename}")


def get_dataset(df: pd.DataFrame) -> torch.utils.data.Dataset:
    """Conversion from pandas data frame to torch dataset."""

    sids = torch.from_numpy(df["sid"].to_numpy())
    pids = torch.from_numpy(df["pid"].to_numpy())
    ratings = torch.from_numpy(df["rating"].to_numpy()).float()
    return torch.utils.data.TensorDataset(sids, pids, ratings)
