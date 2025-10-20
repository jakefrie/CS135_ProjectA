import numpy as np
import pandas as pd
import os

import matplotlib.pyplot as plt
from scipy.stats import loguniform

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

RANDOM_STATE = 0

def make_pipeline():
    return Pipeline([
        # create our Bag of Words
        ("bow", CountVectorizer(
            lowercase=True,          # normalize case before tokenizing
            stop_words='english',    # drop common English stop words
            min_df=3,                # drop terms appearing in < 3 docs (can be overridden by search)
            max_df=0.9,              # drop terms appearing in > 90% of docs (too common)
            ngram_range=(1, 1),      # unigrams only (search may try (1,2) too)
            binary=False             # use integer counts, not just 0/1 presence
        )),
        ("clf", LogisticRegression(
            penalty="l2",            # default regularization (search may flip to "l1")
            solver="liblinear",      # supports l1/l2; good for smaller, sparse problems
            max_iter=5000,           # ensure convergence
            random_state=RANDOM_STATE
        ))
    ])

def make_RandomizedSearch():
    param_distributions = {
        # Vectorizer knobs (vocabulary pruning + feature encoding)
        "bow__min_df": [1, 3, 5, 10],
        "bow__max_df": [0.5, 0.7, 0.9],
        "bow__ngram_range": [(1,1), (1,2)],
        "bow__binary": [False, True],
        
        # Classifier knobs
        "clf__C": loguniform(1e-4, 1e4),
        "clf__penalty": ["l1", "l2"]
    }

    # K-fold Cross Validation with class-balance preserved in every fold
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    #   - draws n_iter random combos from param_distributions
    #   - for each combo: run 5-fold CV, score with AUROC, keep best
    search = RandomizedSearchCV(
        make_pipeline(),
        param_distributions=param_distributions,
        n_iter=25,                 # number of random combos to evaluate
        scoring="roc_auc",         # rank-based metric robust to class imbalance
        cv=cv,                     # 5-fold stratified CV
        n_jobs=-1,                 # use all cores
        random_state=RANDOM_STATE, # make the random draws reproducible
        verbose=1                  # print progress
    )

    return search

def create_model(x_df, y_df, x_te_df):  
    # --- 1) Extract features/labels from input DataFrames ---
    # Raw texts: fill NaN with empty strings so vectorizer doesn't choke
    text_series   = x_df["text"].fillna("")
    # Binary labels - positive class is level 4-5
    labels_series = (y_df["Coarse Label"] == "Key Stage 4-5").astype(int)
    x_te  = x_test_df["text"].fillna("")

    # --- 2) Hold-out validation split BEFORE any CV/search ---
    # Keep 20% aside for a final, untouched evaluation of the chosen model
    X_tr, X_va, y_tr, y_va = train_test_split(
        text_series, labels_series,
        test_size=0.2, stratify=labels_series, random_state=RANDOM_STATE
    )

    # --- 3) Build and run randomized hyperparameter search with 5-fold CV on TRAIN only ---
    search = make_RandomizedSearch()
    search.fit(X_tr, y_tr)
    best_model = search.best_estimator_
    best_model.fit(X_tr, y_tr)

    final_model = make_pipeline()
    final_model.set_params(**search.best_params_)
    final_model.fit(text_series, labels_series)

    # --- 5) Predict proba on the full test set and save exactly one float per line ---
    yproba_test = final_model.predict_proba(x_te)[:, 1]
    np.savetxt("yproba1_test.txt", yproba_test, fmt="%.6f")


if __name__ == '__main__':

    data_dir = 'data'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
    x_test_df = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

    create_model(x_train_df, y_train_df, x_test_df)