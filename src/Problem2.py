import numpy as np
import pandas as pd
import os

from scipy.stats import loguniform
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 0
TEXT_COL = "text"   # <-- change this if your text column has a different name

# -------------------------------
# Utilities: labels and features
# -------------------------------

def build_target(y_df):
    """
    Map the 'Coarse Label' column to a binary target:
      'Key Stage 4-5' -> 1 (positive class), else 0.
    Returns a numpy array of shape (n_samples,).
    """
    return (y_df["Coarse Label"] == "Key Stage 4-5").astype(int).values

def list_numeric_feature_names(x_df):
    """
    Identify numeric columns to include as numeric features.
    We drop obvious text/id columns so they don't leak into numeric branch.
    """
    drop_like = {"text", "title", "author", "passage_id", "id"}
    candidate_cols = [c for c in x_df.columns if c not in drop_like]
    num_cols = [c for c in candidate_cols if np.issubdtype(x_df[c].dtype, np.number)]
    return num_cols

def make_tfidf_lr_search(num_cols):
    """
    Build a Pipeline:
      - ColumnTransformer:
          * 'tfidf': TfidfVectorizer applied to TEXT_COL
          * 'num'  : StandardScaler(with_mean=False) on numeric columns
        (with_mean=False keeps compatibility with sparse matrices)
      - LogisticRegression(solver='liblinear') as classifier
    Then wrap in RandomizedSearchCV over a reasonable space that explores
    underfitting -> sweet spot -> overfitting.

    We tune:
      - TF-IDF: n-grams, min_df, max_df
      - LR   : C (inverse regularization), class_weight
    """
    # Transformer for numeric columns: scale but preserve sparsity semantics
    # (with_mean=False is required if final matrix can be sparse)
    numeric_scaler = StandardScaler(with_mean=False)

    # Text vectorizer: TF-IDF
    tfidf = TfidfVectorizer(
        # defaults; specifics tuned via search space below
        lowercase=True,
        strip_accents=None
    )

    # ColumnTransformer wires each transformer to its column(s)
    preprocessor = ColumnTransformer(
        transformers=[
            ("tfidf", tfidf, TEXT_COL),            # apply TF-IDF to the text column
            ("num", numeric_scaler, num_cols),     # scale numeric feature columns
        ],
        remainder="drop",                          # ignore any other columns
        sparse_threshold=0.3                       # prefer sparse if large TF-IDF dominates
    )

    # Classifier: liblinear supports sparse input well for binary LR
    clf = LogisticRegression(
        penalty="l2",
        solver="liblinear",
        max_iter=2000,
        random_state=RANDOM_STATE
    )

    # Pipeline: preprocessing inside CV to avoid leakage
    from sklearn.pipeline import Pipeline
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", clf)
    ])

    # Hyperparameter search space:
    # - TfidfVectorizer:
    #     * ngram_range: unigrams vs unigrams+bigrams
    #     * min_df: drop very rare terms to reduce noise
    #     * max_df: drop very common terms (stop-word-ish)
    # - LogisticRegression:
    #     * C: inverse of regularization strength (log-uniform over wide range)
    #     * class_weight: balance classes if needed
    param_distributions = {
        "preprocess__tfidf__ngram_range": [(1, 1), (1, 2)],
        "preprocess__tfidf__min_df": [1, 2, 3, 5],
        "preprocess__tfidf__max_df": [0.85, 0.90, 0.95, 1.0],
        "clf__C": loguniform(1e-3, 1e2),
        "clf__class_weight": [None, "balanced"],
    }

    # 5-fold stratified CV, optimize ROC-AUC
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=35,                # a bit more breadth since space includes TF-IDF
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
        refit=True                # best model refit on the whole training subset used in CV
    )
    return search

# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    data_dir = "data"

    # 1) Load train/test frames
    x_train_df = pd.read_csv(os.path.join(data_dir, "x_train.csv"))
    y_train_df = pd.read_csv(os.path.join(data_dir, "y_train.csv"))
    x_test_df  = pd.read_csv(os.path.join(data_dir, "x_test.csv"))

    # 2) Sanity checks for required columns
    if TEXT_COL not in x_train_df.columns or TEXT_COL not in x_test_df.columns:
        raise ValueError(
            f"Expected a text column named '{TEXT_COL}'. "
            "Set TEXT_COL to the correct name."
        )

    # 3) Build target array
    y = build_target(y_train_df)

    # 4) Identify numeric feature columns (will be scaled and concatenated with TF-IDF)
    num_cols = list_numeric_feature_names(x_train_df)

    # 5) Keep only the columns we actually need for the model (text + numeric cols)
    #    This ensures ColumnTransformer sees the exact schema during fit/transform.
    cols_needed = [TEXT_COL] + num_cols
    X_train_full = x_train_df[cols_needed].copy()
    X_test_full  = x_test_df[cols_needed].copy()

    # 6) Small external holdout (10%) for a sanity check outside CV.
    #    CV remains the primary selection mechanism.
    X_tr_df, X_va_df, y_tr, y_va = train_test_split(
        X_train_full, y,
        test_size=0.10, stratify=y, random_state=RANDOM_STATE
    )

    # 7) Build randomized search over TF-IDF + numeric scaler + LR
    search = make_tfidf_lr_search(num_cols=num_cols)

    # 8) Fit the search on the train split (preprocessing is inside the pipeline)
    search.fit(X_tr_df, y_tr)

    print("Best params:", search.best_params_)

    # 9) Evaluate on the 10% external holdout
    va_proba = search.best_estimator_.predict_proba(X_va_df)[:, 1]
    val_auc = roc_auc_score(y_va, va_proba)
    print(f"Validation AUROC (10% holdout): {val_auc:.4f}")

    # 10) Train the best estimator on the same train split and produce test predictions
    #     (Optionally, you could refit on ALL training data before test prediction:
    #      search.refit=True already has the best model fit on all folds’ training portions.
    #      For a final fit on ALL of X_train_full, you can do best_model.fit(X_train_full, y).)
    best_model = search.best_estimator_
    yproba_test = best_model.predict_proba(X_test_full)[:, 1]

    # 11) Save leaderboard submission file
    np.savetxt("yproba1_test.txt", yproba_test, fmt="%.6f")
    print("Wrote yproba1_test.txt")
