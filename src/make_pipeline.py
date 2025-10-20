import numpy as np
from scipy.stats import loguniform

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV

RANDOM_STATE = 0

def make_pipeline():
    return Pipeline([
        ("vec", CountVectorizer(
            lowercase=True,
            stop_words=None,
            min_df=3,
            max_df=0.9,
            ngram_range=(1, 1),
            binary=False
        )),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            solver="liblinear",      # supports l1/l2, good for sparse text
            max_iter=5000,
            random_state=RANDOM_STATE
        ))
    ])

def create_model(x_df, y_df):
    # Prepare data
    text_series   = x_df["text"].fillna("")
    labels_series = (y_df["Coarse Label"] == "Key Stage 4-5").astype(int)

    X_tr, X_va, y_tr, y_va = train_test_split(
        text_series, labels_series,
        test_size=0.2, stratify=labels_series, random_state=RANDOM_STATE
    )

    pipe = make_pipeline()

    # --- RandomizedSearchCV setup ---
    # NOTE: step names must match the pipeline: 'vec__...' and 'clf__...'
    param_distributions = {
        "vec__min_df":        [1, 2, 3, 5, 10],
        "vec__max_df":        [0.5, 0.7, 0.9],
        "vec__ngram_range":   [(1, 1), (1, 2)],
        "vec__binary":        [False, True],
        # Sample C on a log scale
        "clf__C":             loguniform(1e-4, 1e4),
        # Try both penalties (solver kept to liblinear which supports l1/l2)
        "clf__penalty":       ["l1", "l2"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=30,                 # adjust for your budget
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,                # refit on full training split with best params
        verbose=1
    )

    search.fit(X_tr, y_tr)

    # Optional: quick validation score on the held-out split
    val_acc = search.best_estimator_.score(X_va, y_va)
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Validation accuracy (held-out): {val_acc:.4f}")
    print("Best params:", search.best_params_)

    return search
