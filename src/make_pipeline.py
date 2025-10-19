import numpy as np
import sklearn.linear_model
import sklearn.pipeline

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

RANDOM_STATE = 0

def make_pipeline():
    baseline_pipe = Pipeline([
        ("vec", CountVectorizer(
            lowercase=True,
            # Start simple; we’ll tune these later:
            stop_words=None,          # try 'english' later
            min_df=3,                 # try {1,3,5,10}
            max_df=0.9,               # try {0.9,0.7,0.5}
            ngram_range=(1,1),        # try (1,2) later
            binary=False              # try True later
        )),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,                    # we’ll sweep this later
            solver="liblinear",       # good for small/medium, supports l1/l2
            max_iter=5000,
            random_state=RANDOM_STATE
        ))
    ])
    return baseline_pipe

def create_model(x_df, y_df):
    text_series = x_df['text'].fillna("")
    labels_series = (y_df['Coarse Label'] == 'Key Stage 4-5').astype(int)
    X_tr, X_va, y_tr, y_va = train_test_split(
    text_series, labels_series, test_size=0.2, stratify=labels_series, random_state=RANDOM_STATE)

    pipe = make_pipeline()

    my_parameter_grid_by_name = dict()
    my_parameter_grid_by_name['my_bow_feature_extractor__min_df'] = [1, 2, 4]
    my_parameter_grid_by_name['my_classifier__C'] = np.logspace(-4, 4, 9)
    my_scoring_metric_name = 'accuracy'

    prng = np.random.RandomState(0)
    valid_ids = prng.choice(np.arange(N), size=100)
    valid_indicators_N = np.zeros(N)
    valid_indicators_N[valid_ids] = -1
    my_splitter = sklearn.model_selection.PredefinedSplit(valid_indicators_N)

    pipe = make_pipeline()
    baseline_pipe.fit(X_tr, y_tr)
    proba_va = baseline_pipe.predict_proba(X_va)[:, 1]

    grid_searcher = sklearn.model_selection.GridSearchCV(
        pipe,
        my_parameter_grid_by_name,
        scoring=my_scoring_metric_name,
        cv=my_splitter,
        refit=False)
     
    grid_searcher.fit(list_of_training_text_reviews, y_tr_N)
    return grid_searcher

# import numpy as np
# import pandas as pd

# from sklearn.pipeline import Pipeline
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split

# RANDOM_STATE = 0

# def make_pipeline():
#     return Pipeline([
#         ("vec", CountVectorizer(
#             lowercase=True,
#             stop_words=None,
#             min_df=3,
#             max_df=0.9,
#             ngram_range=(1,1),
#             binary=False
#         )),
#         ("clf", LogisticRegression(
#             penalty="l2",
#             C=1.0,
#             solver="liblinear",
#             max_iter=5000,
#             random_state=RANDOM_STATE
#         ))
#     ])

# def create_model(x_df, y_df):
#     # Prepare features/labels
#     X = x_df['text'].fillna("")
#     y = (y_df['Coarse Label'] == 'Key Stage 4-5').astype(int)

#     # Quick sanity split (optional; not used by GridSearchCV below)
#     X_tr, X_va, y_tr, y_va = train_test_split(
#         X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
#     )

#     # Pipeline
#     pipe = make_pipeline()

#     # Parameter grid — names MUST match step names ("vec", "clf")
#     param_grid = {
#         # Vectorizer knobs (keep modest to control runtime)
#         "vec__stop_words": [None, "english"],
#         "vec__min_df": [1, 3, 5],
#         "vec__max_df": [0.9, 0.7],
#         "vec__ngram_range": [(1,1), (1,2)],
#         "vec__binary": [False, True],

#         # Logistic Regression complexity
#         "clf__C": np.logspace(-3, 3, 7),  # ≥5 values
#         # optional:
#         # "clf__class_weight": [None, "balanced"],
#     }

#     # Cross-validation strategy (simple and solid)
#     cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

#     # Searcher — use AUROC, get train scores for under/overfit plots, refit best model
#     search = GridSearchCV(
#         estimator=pipe,
#         param_grid=param_grid,
#         scoring="roc_auc",
#         cv=cv,
#         n_jobs=-1,
#         verbose=1,
#         return_train_score=True,
#         refit=True   # keep the best-fitted pipeline available via search.best_estimator_
#     )

#     # Fit on the full training set (not the small holdout)
#     search.fit(X, y)

#     return search
