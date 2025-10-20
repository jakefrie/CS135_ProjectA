"""
SUMMARY:
- read in the files 
- extract the text columns from the CSVs 
- partition into training and validation - all subsequent steps are done with just training data
- count up all the words in all texts 
- create a bag of words based on our specification (exclude words appearing in <3 cods, or >90% of docs, for example)
- vectorize each text into the form of an array with length = [size of our bag of words]
- cross validation - partition data into equal parts, and keep one part as "Validation" while training on the rest
    we then average the score for each partition to come up with a total score for that set of hyperparameters
    - repeat for each set of hyperparameters
- Choose the best set of hyperparameters (based on AUROC)
- Finally, run this on the original validation set (that we havent used yet)
- Output the confusion matrix for this set
"""

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

#fix all randomness
RANDOM_STATE = 0

def make_pipeline():
    """
    Build the end-to-end modeling pipeline:
      text (strings)
        -> CountVectorizer (tokenize, prune vocab, create sparse count matrix)
        -> LogisticRegression (linear classifier on sparse counts)
    """
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
    """
    Configure a randomized hyperparameter search over both vectorizer and classifier.
    Cross-validation (StratifiedKFold) estimates each hyperparameter combo's performance.
    """
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

def create_model(x_df, y_df):
    """
    Orchestrates:
      1) Extract text and labels
      2) Train/validation split (hold out 20% for final check)
      3) Hyperparameter search with CV on the training split
      4) Evaluate best model on held-out validation split
      5) Plot confusion matrix (threshold 0.5)
    """    
    # --- 1) Extract features/labels from input DataFrames ---
    # Raw texts: fill NaN with empty strings so vectorizer doesn't choke
    text_series   = x_df["text"].fillna("")
    # Binary labels - positive class is level 4-5
    labels_series = (y_df["Coarse Label"] == "Key Stage 4-5").astype(int)

    # --- 2) Hold-out validation split BEFORE any CV/search ---
    # Keep 20% aside for a final, untouched evaluation of the chosen model
    X_tr, X_va, y_tr, y_va = train_test_split(
        text_series, labels_series,
        test_size=0.2, stratify=labels_series, random_state=RANDOM_STATE
    )

    # --- 3) Build and run randomized hyperparameter search with 5-fold CV on TRAIN only ---
    search = make_RandomizedSearch()
    # This single call triggers:
    #   - vocabulary building (CountVectorizer.fit) on each CV train fold
    #   - vectorization (transform) for train/val folds
    #   - model fitting (LogisticRegression.fit) per fold
    #   - AUROC scoring per fold, averaged per hyperparameter combo

    search.fit(X_tr, y_tr)
    print("Best AUROC:", search.best_score_)
    print("Best params:", search.best_params_)

    # --- 4) Evaluate best found pipeline on the held-out validation split ---
    # Probabilities for AUROC; hard labels for confusion matrix
    y_val_pred = search.best_estimator_.predict(X_va)
    y_val_proba = search.best_estimator_.predict_proba(X_va)[:,1]

    print("Validation AUROC:", roc_auc_score(y_va, y_val_proba))

    # --- 5) Visual diagnostic: confusion matrix on the hold-out split ---

    ConfusionMatrixDisplay(confusion_matrix(y_va, y_val_pred)).plot()
    plt.show()


# Read in the files
if __name__ == '__main__':
    # Entry point: read CSVs and kick off the pipeline

    data_dir = 'data'
    x_train_df = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
    y_train_df = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))

    create_model(x_train_df, y_train_df)