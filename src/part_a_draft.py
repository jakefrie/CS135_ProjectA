### 0. IMPORTS & SETUP ###

# Core
import numpy as np
import pandas as pd

# Modeling
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# Plotting (for quick looks; Part C will make formal figures)
import matplotlib.pyplot as plt

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

### 1. LOAD DATA ###

# Paths relative to your repo
X_TRAIN_PATH = "data/x_train.csv"
Y_TRAIN_PATH = "data/y_train.csv"

x_df = pd.read_csv(X_TRAIN_PATH)
y_df = pd.read_csv(Y_TRAIN_PATH)

# Sanity checks
display(x_df.head(2))
display(y_df.head(2))

# Keep text only for P1 (ignore numeric features for now)
# IMPORTANT: do NOT use the key columns (author, title, passage_id) as features
text_series = x_df['text'].fillna("")
labels_series = (y_df['Coarse Label'] == 'Key Stage 4-5').astype(int)  # 1 for upper-level

print("Train size:", len(text_series))
print("Class balance:\n", labels_series.value_counts(normalize=True))

### 2. VALIDATION SPLIT ###
X_tr, X_va, y_tr, y_va = train_test_split(
    text_series, labels_series, test_size=0.2, stratify=labels_series, random_state=RANDOM_STATE
)

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

baseline_pipe.fit(X_tr, y_tr)
proba_va = baseline_pipe.predict_proba(X_va)[:, 1]
print("Quick baseline AUROC:", roc_auc_score(y_va, proba_va))

### 3. PEEK AT VOCABULARY CHOICES ###
 def describe_vectorizer(vec, X=text_series):
    vec.fit(X)
    vocab_size = len(vec.vocabulary_)
    print(f"Vocab size: {vocab_size:,}")
    # top tokens by df are not directly exposed; use analyzer to estimate quickly if needed
    return vocab_size

# Compare a few reasonable configs
configs = [
    dict(stop_words=None, min_df=1, max_df=0.9, ngram_range=(1,1), binary=False),
    dict(stop_words='english', min_df=3, max_df=0.9, ngram_range=(1,1), binary=False),
    dict(stop_words='english', min_df=5, max_df=0.7, ngram_range=(1,1), binary=False),
    dict(stop_words='english', min_df=3, max_df=0.9, ngram_range=(1,2), binary=False),
    dict(stop_words='english', min_df=3, max_df=0.9, ngram_range=(1,1), binary=True),
]

sizes = []
for cfg in configs:
    vec = CountVectorizer(lowercase=True, **cfg)
    sizes.append(describe_vectorizer(vec))

sizes


### 4. K-FOLD AUROC ###
best_guess_pipe = Pipeline([
    ("vec", CountVectorizer(
        lowercase=True,
        stop_words='english',
        min_df=3,
        max_df=0.9,
        ngram_range=(1,1),
        binary=False
    )),
    ("clf", LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="liblinear",
        max_iter=5000,
        random_state=RANDOM_STATE
    ))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(best_guess_pipe, text_series, labels_series,
                            cv=cv, scoring="roc_auc", n_jobs=-1)
print("CV AUROC mean ± std:", f"{cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
