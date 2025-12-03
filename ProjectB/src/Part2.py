"""
Project B – Problem 2 using Surprise SVD
=======================================

This script:

1. Loads ratings_all_development_set.csv
2. Splits into train / valid / test (90% train-valid, 10% test)
3. Hyperparameter-tunes Surprise SVD for MAE using 3-fold CV
4. Trains the best SVD model on train+valid and reports test MAE
5. Trains final SVD on ALL dev data
6. Predicts ratings for ratings_masked_leaderboard_set.csv
7. Saves predictions to predicted_ratings_leaderboard.txt

IMPORTANT: 
This version correctly treats user_id and item_id as integers so predictions
do NOT collapse to the global mean.
"""

import os
import numpy as np
import pandas as pd

from surprise import Dataset, Reader, SVD
from surprise.model_selection import GridSearchCV


# ---------------------------------------------------------
# 1. Load development dataset
# ---------------------------------------------------------

def load_dev_dataframe(csv_path):
    df = pd.read_csv(csv_path)
    return df


def make_surprise_dataset(df):
    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)


# ---------------------------------------------------------
# 2. Create train/valid/test splits using pandas/sklearn
# ---------------------------------------------------------

def split_dev_data(df, valid_size=0.1, test_size=0.1, random_state=0):
    from sklearn.model_selection import train_test_split

    # First split dev → (train_valid + test)
    df_train_valid, df_test = train_test_split(
        df,
        test_size=test_size,
        shuffle=True,
        random_state=random_state
    )

    # Then split train_valid → (train + valid)
    rel_valid_size = valid_size / (1.0 - test_size)
    df_train, df_valid = train_test_split(
        df_train_valid,
        test_size=rel_valid_size,
        shuffle=True,
        random_state=random_state
    )

    return (
        df_train.reset_index(drop=True),
        df_valid.reset_index(drop=True),
        df_test.reset_index(drop=True)
    )


def build_trainset(df):
    """Convert a df into a Surprise Trainset."""
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[['user_id', 'item_id', 'rating']], reader)
    return data.build_full_trainset()


def build_testset(df):
    """Convert df into a Surprise-style testset list of (uid, iid, rating)."""
    return list(zip(
        df['user_id'],      # keep as int
        df['item_id'],      # keep as int
        df['rating'].astype(float)
    ))


# ---------------------------------------------------------
# 3. Hyperparameter tuning for SVD
# ---------------------------------------------------------

def tune_svd(df_train, df_valid, random_state=0):
    """Perform GridSearchCV on train+valid pooled data."""
    df_tv = pd.concat([df_train, df_valid], ignore_index=True)

    reader = Reader(rating_scale=(1, 5))
    data_tv = Dataset.load_from_df(df_tv[['user_id', 'item_id', 'rating']], reader)

    # A reasonable grid (adjust if you want deeper tuning)
    param_grid = {
        'n_factors': [20, 50, 100],
        'lr_all': [0.002, 0.005],
        'reg_all': [0.02, 0.05, 0.1],
        'n_epochs': [20, 40, 60],
    }

    gs = GridSearchCV(
        SVD,
        param_grid,
        measures=['mae'],
        cv=3,
        n_jobs=-1,
        joblib_verbose=1
    )
    gs.fit(data_tv)

    print("Best MAE score:", gs.best_score['mae'])
    print("Best params:", gs.best_params['mae'])

    return gs.best_params['mae']


# ---------------------------------------------------------
# 4. Train on train+valid, evaluate on test
# ---------------------------------------------------------

def train_and_eval(df_train, df_valid, df_test, best_params, random_state=0):
    df_tv = pd.concat([df_train, df_valid], ignore_index=True)

    trainset_tv = build_trainset(df_tv)
    testset = build_testset(df_test)

    algo = SVD(
        n_factors=best_params['n_factors'],
        lr_all=best_params['lr_all'],
        reg_all=best_params['reg_all'],
        n_epochs=best_params['n_epochs'],
        random_state=random_state,
        verbose=True
    )

    algo.fit(trainset_tv)
    predictions = algo.test(testset)

    abs_errors = [abs(p.r_ui - p.est) for p in predictions]
    mae = float(np.mean(abs_errors))

    print(f"Dev Test MAE = {mae:.5f}")

    return algo, mae


# ---------------------------------------------------------
# 5. Train on ALL dev data
# ---------------------------------------------------------

def train_on_all(df_all, best_params, random_state=0):
    reader = Reader(rating_scale=(1, 5))
    data_all = Dataset.load_from_df(df_all[['user_id', 'item_id', 'rating']], reader)
    trainset_all = data_all.build_full_trainset()

    algo = SVD(
        n_factors=best_params['n_factors'],
        lr_all=best_params['lr_all'],
        reg_all=best_params['reg_all'],
        n_epochs=best_params['n_epochs'],
        random_state=random_state,
        verbose=True
    )

    algo.fit(trainset_all)
    return algo


# ---------------------------------------------------------
# 6. Generate leaderboard predictions
# ---------------------------------------------------------

def load_leaderboard_df(csv_path):
    return pd.read_csv(csv_path)


def predict_leaderboard(algo, df_lb, out_path):
    preds = []
    for _, row in df_lb.iterrows():
        uid = row['user_id']   # must keep as int
        iid = row['item_id']   # must keep as int
        est = algo.predict(uid, iid).est
        preds.append(est)

    preds = np.clip(np.array(preds, dtype=float), 1.0, 5.0)
    np.savetxt(out_path, preds, fmt="%.6f")
    print(f"Saved {len(preds)} predictions to {out_path}")


# ---------------------------------------------------------
# 7. Main
# ---------------------------------------------------------

def main():

    # Update if your path differs
    dev_csv = os.path.join("data", "ratings_all_development_set.csv")
    leaderboard_csv = os.path.join("data", "ratings_masked_leaderboard_set.csv")

    df_all = load_dev_dataframe(dev_csv)
    print("Loaded dev ratings:", df_all.shape)

    # Split data
    df_train, df_valid, df_test = split_dev_data(df_all)
    print(f"Train: {len(df_train)}, Valid: {len(df_valid)}, Test: {len(df_test)}")

    # Hyperparameter tuning
    best_params = tune_svd(df_train, df_valid)

    # Train + evaluate on dev test split
    algo_tv, dev_test_mae = train_and_eval(df_train, df_valid, df_test, best_params)

    # Train on ALL dev data
    algo_all = train_on_all(df_all, best_params)

    # Predict leaderboard
    df_lb = load_leaderboard_df(leaderboard_csv)
    predict_leaderboard(algo_all, df_lb, "predicted_ratings_leaderboard.txt")


if __name__ == "__main__":
    main()
