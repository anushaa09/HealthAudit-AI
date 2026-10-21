"""
build_dice_background.py

Builds the small background dataset used by the /recourse endpoint.

Why this exists: DiCE only needs a background dataset to learn realistic
feature ranges and category values -- it does not need the full training
split. The full split (data/processed/train.parquet) is large and git-ignored,
so it never reaches Render, which made /recourse fail in production. This
script condenses it into data/processed/dice_background.parquet, a file small
enough to commit to Git and deploy.

The sample is built to behave like the full split for DiCE's purposes:
  - stratified on the target, so the Approve/Deny ratio is preserved;
  - plus the rows holding the minimum and maximum of every numeric column,
    so feature ranges (and the permitted-range caps) match the full split;
  - plus one row per category value, so no category is lost.

Two modes:

  1. From the real training split (preferred, whenever a valid train.parquet exists):
        python -m src.explainability.build_dice_background
        python -m src.explainability.build_dice_background --source path/to/train.parquet --rows 8000

  2. From the trained model itself (fallback, if the training split is lost):
        python -m src.explainability.build_dice_background --from-model
     This builds a MODEL-DERIVED reference set -- it is NOT real patient data.
     It uses the feature means/standard deviations stored inside the fitted
     preprocessor, the real category values stored in its encoder, and plausible
     ceilings for dollar amounts and stays. DiCE only uses the background to know
     where it may search; every counterfactual it returns is still checked against
     the real trained model. The file is stamped with its origin, and the API logs
     a warning whenever it loads a model-derived file. Replace it with mode 1 as
     soon as the real training split is recovered.

Authors: Nandita Gaur, Anusha Sarkar
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.config.paths import DICE_BACKGROUND_FILE, TRAIN_DATASET_FILE, TRAINED_MODEL_FILE
from src.models.claim_classifier import ClaimClassifier
from src.training.train_baseline_models import CATEGORICAL_COLUMNS, TARGET_COLUMN
from src.utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_ROWS = 8000
RANDOM_STATE = 42

# Metadata key stamped into model-derived files so they can never be mistaken
# for real training data (read back by api/dependencies.py).
SOURCE_METADATA_KEY = b"healthaudit_background_source"
MODEL_DERIVED_LABEL = b"model-derived reference set (not real training data)"

# Plausible ceilings used ONLY in model-derived mode.
MAX_DOLLARS = 500_000.0      # any single dollar-valued feature
MAX_STAY_DAYS = 180          # total inpatient days
MIN_DENY_ROWS = 100          # the reference set must contain both outcomes


def _model_derived_bounds(name: str, mean: float, std: float) -> tuple[float, float, str]:
    """Return (low, high, kind) for one numeric feature in model-derived mode."""
    if name in ("BENE_ESRD_IND", "is_deceased") or name.startswith("SP_"):
        return 0.0, 1.0, "binary"
    if name.endswith("_CVRAGE_TOT_MONS") or name == "PLAN_CVRG_MOS_NUM":
        return 0.0, 12.0, "int"
    if name == "age":
        return 18.0, 105.0, "int"
    if name == "chronic_condition_count":
        return 0.0, 11.0, "int"
    if name.startswith(("total_", "distinct_")) and not name.endswith("_log"):
        return 0.0, float(np.ceil(mean + 6 * std)), "int"
    if name == "total_inpatient_days_log":
        return 0.0, float(np.log1p(MAX_STAY_DAYS)), "float"
    if name.endswith("_log"):
        return 0.0, float(np.log1p(MAX_DOLLARS)), "float"
    return 0.0, MAX_DOLLARS, "float"  # MEDREIMB_*, BENRES_*, PPPYMT_*, cost_per_claim


def build_model_derived_reference(
    classifier: ClaimClassifier, n_rows: int = 6000, random_state: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Build a reference dataset from the trained model's own preprocessing
    statistics (see the module docstring). Labels come from the real model.
    """
    preprocessor = classifier.pipeline.named_steps["preprocessor"]
    numeric_cols = list(preprocessor.transformers_[0][2])
    scaler = preprocessor.transformers_[0][1]
    categorical_cols = list(preprocessor.transformers_[1][2])
    encoder = preprocessor.transformers_[1][1]

    rng = np.random.default_rng(random_state)
    # ~1 in 4 rows is drawn from a wider spread so that high-utilisation
    # (Deny-like) claims exist in the reference set, as they do in real data.
    spread = rng.choice([1.0, 1.0, 1.0, 3.0], size=n_rows)

    data: dict[str, np.ndarray] = {}
    bounds: dict[str, tuple[float, float, str]] = {}
    for i, col in enumerate(numeric_cols):
        mean, std = float(scaler.mean_[i]), float(scaler.scale_[i])
        low, high, kind = _model_derived_bounds(col, mean, std)
        bounds[col] = (low, high, kind)
        if kind == "binary":
            values = (rng.random(n_rows) < min(max(mean, 0.0), 1.0)).astype(float)  # mean of a 0/1 column = its rate
        else:
            values = np.clip(mean + std * rng.standard_normal(n_rows) * spread, low, high)
            if kind == "int":
                values = np.round(values)
        data[col] = values

    df = pd.DataFrame(data)

    # Keep the derived count consistent with the flags it summarises.
    flag_cols = [c for c in numeric_cols if c.startswith("SP_")]
    df["chronic_condition_count"] = df[flag_cols].sum(axis=1)

    # Explicit minimum and maximum rows, so the feature ranges DiCE learns are
    # exactly the bounds above.
    edge_rows = []
    for edge in (0, 1):
        edge_rows.append({c: bounds[c][edge] for c in numeric_cols})
    df = pd.concat([df, pd.DataFrame(edge_rows)], ignore_index=True)

    n_total = len(df)
    for col, categories in zip(categorical_cols, encoder.categories_):
        cycled = np.resize(np.array(categories, dtype=object), n_total)  # every real category appears
        df[col] = rng.permutation(cycled)

    for col, (_, _, kind) in bounds.items():
        if kind in ("binary", "int"):
            df[col] = df[col].astype("int64")

    df = df[list(classifier.feature_columns)]
    df[TARGET_COLUMN] = classifier.predict(df).astype("int64").values

    n_deny = int(df[TARGET_COLUMN].sum())
    if n_deny < MIN_DENY_ROWS:
        raise RuntimeError(f"Reference set has only {n_deny} Deny rows; expected at least {MIN_DENY_ROWS}.")
    return df


def build_background_sample(
    df: pd.DataFrame,
    feature_columns: list[str],
    n_rows: int = DEFAULT_ROWS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Condense a full training split into a small DiCE background dataset.

    Parameters
    ----------
    df : pd.DataFrame
        The full training split. Must contain every model feature column and
        the target column.
    feature_columns : list[str]
        The model's feature columns (read from the trained model's metadata).
    n_rows : int
        Approximate size of the stratified random part of the sample. The
        range/category coverage rows are added on top, so the final size is
        slightly larger.
    """
    required = list(feature_columns) + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Source data is missing required columns: {missing}")

    df = df[required].reset_index(drop=True)

    if n_rows >= len(df):
        return df.copy()

    # 1. Stratified random sample -- keeps the Approve/Deny ratio.
    fraction = n_rows / len(df)
    sampled = df.groupby(TARGET_COLUMN, group_keys=False).sample(frac=fraction, random_state=random_state)
    keep = set(sampled.index)

    # 2. Rows holding each numeric column's min and max -- keeps feature ranges.
    numeric_columns = [c for c in feature_columns if c not in CATEGORICAL_COLUMNS]
    for col in numeric_columns:
        if df[col].notna().any():
            keep.add(df[col].idxmin())
            keep.add(df[col].idxmax())

    # 3. First row of each category value -- keeps every category.
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            keep.update(df.drop_duplicates(subset=col).index)

    return df.loc[sorted(keep)].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the DiCE background dataset for /recourse.")
    parser.add_argument("--source", type=Path, default=TRAIN_DATASET_FILE, help="Full training split (parquet).")
    parser.add_argument("--output", type=Path, default=DICE_BACKGROUND_FILE, help="Where to write the sample.")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Approximate sample size.")
    parser.add_argument(
        "--from-model",
        action="store_true",
        help="Fallback: build a model-derived reference set instead of reading a training split.",
    )
    args = parser.parse_args()

    classifier = ClaimClassifier.load(TRAINED_MODEL_FILE)

    if args.from_model:
        sample_df = build_model_derived_reference(classifier)
        source_df = None
    else:
        logger.info("Reading source data: %s", args.source)
        source_df = pd.read_parquet(args.source)
        logger.info("Source: %d rows, %d columns.", *source_df.shape)
        sample_df = build_background_sample(source_df, classifier.feature_columns, n_rows=args.rows)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(sample_df, preserve_index=False)
    if args.from_model:
        table = table.replace_schema_metadata({**(table.schema.metadata or {}), SOURCE_METADATA_KEY: MODEL_DERIVED_LABEL})
    pq.write_table(table, args.output)

    # Read the file back in full to prove it is valid before anyone commits it.
    check_df = pd.read_parquet(args.output)
    if check_df.shape != sample_df.shape:
        raise RuntimeError(f"Verification failed: wrote {sample_df.shape}, read back {check_df.shape}.")

    size_kb = args.output.stat().st_size / 1024
    deny_share = check_df[TARGET_COLUMN].mean() * 100
    print(f"\nSaved: {args.output}")
    if source_df is not None:
        print(f"Rows: {len(check_df)} (from {len(source_df)}) | Size: {size_kb:.0f} KB")
        print(f"Deny share: {deny_share:.2f}% in sample vs {source_df[TARGET_COLUMN].mean() * 100:.2f}% in source")
    else:
        print(f"Rows: {len(check_df)} | Size: {size_kb:.0f} KB | Deny share: {deny_share:.2f}%")
        print("NOTE: this is a MODEL-DERIVED reference set, not real training data. Replace it once the real split is recovered.")


if __name__ == "__main__":
    main()
