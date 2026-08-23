"""
data_split.py

Splits the feature dataset into train / validation / test sets.

Beneficiary-level leakage is already impossible by design: the dataset has
exactly one row per DESYNPUF_ID (aggregated back in merger.py), so a plain
row-level split IS a beneficiary-level split -- no special grouping logic
needed here.

What DOES need care: our proxy_label is heavily imbalanced (97.7% Approve /
2.3% Deny). A plain random split risks uneven class ratios across splits by
chance. We use stratified splitting to guarantee each split preserves the
same class ratio as the full dataset.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


def split_dataset(
    df: pd.DataFrame,
    target_col: str = "proxy_label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split df into train / validation / test sets, stratified on target_col.

    Parameters
    ----------
    df : pd.DataFrame
        Full feature dataset (one row per beneficiary).
    target_col : str
        Column to stratify on -- ensures class ratio is preserved across
        all three splits despite severe class imbalance.
    test_size, val_size : float
        Fractions of the FULL dataset (not of remaining data) held out for
        test and validation respectively. Train gets the remainder.
    random_state : int
        Fixed seed for reproducibility -- same split every time this runs.

    Returns
    -------
    (train_df, val_df, test_df)
    """
    logger.info(
        "Splitting dataset: %.0f%% train / %.0f%% val / %.0f%% test, stratified on '%s'.",
        (1 - test_size - val_size) * 100,
        val_size * 100,
        test_size * 100,
        target_col,
    )

    # First split off the test set.
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[target_col],
        random_state=random_state,
    )

    # Then split the remainder into train / validation. val_size must be
    # rescaled relative to train_val_df's size, not the original full size.
    relative_val_size = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        stratify=train_val_df[target_col],
        random_state=random_state,
    )

    _log_split_summary(train_df, val_df, test_df, target_col)

    return train_df, val_df, test_df


def _log_split_summary(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, target_col: str
) -> None:
    """Log row counts and class distribution per split, to verify stratification worked."""
    for name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        class_counts = split_df[target_col].value_counts(normalize=True) * 100
        logger.info(
            "%s: %d rows | Approve %.2f%% / Deny %.2f%%",
            name,
            len(split_df),
            class_counts.get(0, 0.0),
            class_counts.get(1, 0.0),
        )


if __name__ == "__main__":
    # Run with: python -m src.preprocessing.data_split
    from src.config.paths import FEATURE_DATASET_FILE, PROCESSED_DATA_DIR

    df = pd.read_parquet(FEATURE_DATASET_FILE)
    train_df, val_df, test_df = split_dataset(df)

    train_path = PROCESSED_DATA_DIR / "train.parquet"
    val_path = PROCESSED_DATA_DIR / "val.parquet"
    test_path = PROCESSED_DATA_DIR / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"\nSaved: {train_path}")
    print(f"Saved: {val_path}")
    print(f"Saved: {test_path}")