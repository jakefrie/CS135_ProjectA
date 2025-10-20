import numpy as np
from scipy.stats import loguniform

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score

RANDOM_STATE = 0

def make_pipeline():
    return Pipeline([
        ("bow", CountVectorizer(
            lowercase=True,
            stop_words='english',
            min_df=3,
            max_df=0.9,
            ngram_range=(1,1),
            binary=False
        )),
        ("clf", LogisticRegression(
            penalty="l2",
            solver="liblinear",
            max_iter=5000,
            random_state=RANDOM_STATE
        ))
    ])

def create_model(x_df, y_df):
    text_series   = x_df["text"].fillna("")
    labels_series = (y_df["Coarse Label"] == "Key Stage 4-5").astype(int)

    X_tr, X_va, y_tr, y_va = train_test_split(
        text_series, labels_series,
        test_size=0.2, stratify=labels_series, random_state=RANDOM_STATE
    )

    pipe = make_pipeline()

    param_distributions = {
        "bow__min_df": [1, 3, 5, 10],
        "bow__max_df": [0.5, 0.7, 0.9],
        "bow__ngram_range": [(1,1), (1,2)],
        "bow__binary": [False, True],
        "clf__C": loguniform(1e-4, 1e4),
        "clf__penalty": ["l1", "l2"]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        make_pipeline(),
        param_distributions=param_distributions,
        n_iter=25,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1
    )

    search.fit(X_tr, y_tr)
    print("Best AUROC:", search.best_score_)
    print("Best params:", search.best_params_)

    y_val_pred = search.best_estimator_.predict(X_va)
    y_val_proba = search.best_estimator_.predict_proba(X_va)[:,1]

    print("Validation AUROC:", roc_auc_score(y_va, y_val_proba))

    ConfusionMatrixDisplay(confusion_matrix(y_va, y_val_pred)).plot()

    return search
