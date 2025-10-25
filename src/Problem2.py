import numpy as np
import pandas as pd
import os

from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 0

def load_arr_from_npz(npz_path):
    npz_file_obj = np.load(npz_path)
    arr = npz_file_obj.f.arr_0.copy() # default name from np.savez
    npz_file_obj.close()
    return arr

def build_target(y_df):
    # Positive class: "Key Stage 4-5" → 1, else 0
    return (y_df["Coarse Label"] == "Key Stage 4-5").astype(int).values

def select_numeric_features(x_df):
    # Keep only numeric columns (drop obvious IDs/text columns if present)
    drop_like = {"text", "title", "author", "passage_id", "id"}
    candidate_cols = [c for c in x_df.columns if c not in drop_like]
    num_cols = [c for c in candidate_cols if np.issubdtype(x_df[c].dtype, np.number)]
    return x_df[num_cols].fillna(0.0).to_numpy(), num_cols

def prepare_matrices(x_train_df, x_test_df, xBERT_train, xBERT_test):
    Xnum_tr, num_cols = select_numeric_features(x_train_df)
    Xnum_te, _        = select_numeric_features(x_test_df)
    # Concatenate: [BERT | numeric]
    X_tr = np.hstack([xBERT_train, Xnum_tr])
    X_te = np.hstack([xBERT_test,  Xnum_te])
    return X_tr, X_te, num_cols

def make_search():
    # Fast, strong baseline for dense features
    base = LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=2000,
        n_jobs=1,              # lbfgs ignores n_jobs; keep API consistent
        random_state=RANDOM_STATE
    )
    params = {
        "C": loguniform(1e-3, 1e2),
        "class_weight": [None, "balanced"]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=base,
        param_distributions=params,
        n_iter=30,             # lighter but effective
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )
    return search

if __name__ == '__main__':
    data_dir = 'data'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df  = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    # Load BERT embeddings
    xBERT_train = load_arr_from_npz(os.path.join(data_dir, 'x_train_BERT_embeddings.npz'))
    xBERT_test  = load_arr_from_npz(os.path.join(data_dir, 'x_test_BERT_embeddings.npz'))
    assert xBERT_train.ndim == 2 and xBERT_test.ndim == 2
    assert len(xBERT_train) == len(x_train_df) and len(xBERT_test) == len(x_test_df)

    # Build y and feature matrices
    y = build_target(y_train_df)
    X_train_full, X_test_full, num_cols = prepare_matrices(x_train_df, x_test_df, xBERT_train, xBERT_test)

    # Train/validation split (small holdout; CV handles most validation)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train_full, y, test_size=0.10, stratify=y, random_state=RANDOM_STATE
    )

    # Standardize (dense): fit on train, apply to val and test
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr_std = scaler.fit_transform(X_tr)
    X_va_std = scaler.transform(X_va)
    X_te_std = scaler.transform(X_test_full)

    # Randomized search on the standardized features
    search = make_search()
    search.fit(X_tr_std, y_tr)

    best_model = search.best_estimator_
    # Refit on all training data (train split only), then evaluate on holdout
    best_model.fit(X_tr_std, y_tr)
    va_proba = best_model.predict_proba(X_va_std)[:, 1]
    val_auc = roc_auc_score(y_va, va_proba)
    print("Best params:", search.best_params_)
    print(f"Validation AUROC: {val_auc:.4f}")

    # Final predictions for the competition/test set
    yproba_test = best_model.predict_proba(X_te_std)[:, 1]
    np.savetxt("yproba1_test.txt", yproba_test, fmt="%.6f")
