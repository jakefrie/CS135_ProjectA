import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------
# 1) Load core CSVs
# ---------------------------------------------------------
x_train = pd.read_csv("x_train.csv")
y_train = pd.read_csv("y_train.csv")
x_test  = pd.read_csv("x_test.csv")

# Binary target from Coarse Label
y = y_train["Coarse Label"].map({"Key Stage 2-3": 0, "Key Stage 4-5": 1}).values
groups = x_train["author"].values  # GroupKFold by author to mimic unseen authors

KEYS = ["author", "title", "passage_id"]
TEXT = "text"
base_num_cols = [c for c in x_train.columns if c not in KEYS + [TEXT]]

# ---------------------------------------------------------
# 2) Load PROVIDED BERT embeddings and merge by keys
#    Assumes files: x_train_bert.csv, x_test_bert.csv with the same KEYS + many bert_* columns
# ---------------------------------------------------------
train_bert_path = "x_train_bert.csv"
test_bert_path  = "x_test_bert.csv"

if not (os.path.exists(train_bert_path) and os.path.exists(test_bert_path)):
    raise FileNotFoundError(
        "Expected provided BERT embeddings as x_train_bert.csv and x_test_bert.csv. "
        "Please place them next to x_train.csv/x_test.csv."
    )

bert_train = pd.read_csv(train_bert_path)
bert_test  = pd.read_csv(test_bert_path)

# Merge
x_train = x_train.merge(bert_train, on=KEYS, how="inner", validate="one_to_one")
x_test  = x_test.merge(bert_test,  on=KEYS, how="inner", validate="one_to_one")

# Identify BERT columns (assume they start with 'bert_' or are named numerically)
bert_col_prefixes = ("bert_", "emb_", "vec_")
bert_cols = [c for c in x_train.columns if c not in KEYS + [TEXT] + base_num_cols]
# If your BERT columns are mixed with numeric feature names, filter by a prefix:
if not any(c.startswith(bert_col_prefixes) for c in bert_cols) and len(bert_cols) > 0:
    # Keep all newly-added cols as BERT
    pass
else:
    # Prefer explicit prefix filter
    bert_cols = [c for c in x_train.columns if c.startswith(bert_col_prefixes)]

if len(bert_cols) == 0:
    # Fallback: assume all non-key, non-text, non-base numeric columns that appeared after merge are BERT
    merged_num_cols = [c for c in x_train.columns if c not in KEYS + [TEXT]]
    bert_cols = [c for c in merged_num_cols if c not in base_num_cols]

# Numeric columns = original numeric (not BERT)
num_cols = base_num_cols

print(f"#Numeric features: {len(num_cols)}, #BERT dims: {len(bert_cols)}")

# ---------------------------------------------------------
# 3) Feature representation
#    - TF-IDF -> SVD -> scale  (dense, compact)
#    - Numeric -> scale
#    - BERT -> scale
# ---------------------------------------------------------
tfidf = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    token_pattern=r"\b[a-zA-Z]+\b",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    max_features=50000,     # you may lower to 30k if you want more speed
)

text_branch = Pipeline([
    ("tfidf", tfidf),
    ("svd", TruncatedSVD(n_components=300, random_state=RANDOM_STATE)),  # try 200–400
    ("scale", StandardScaler(with_mean=True))
])

num_branch = Pipeline([
    ("scale", StandardScaler(with_mean=True))
])

bert_branch = Pipeline([
    ("scale", StandardScaler(with_mean=True))
])

preprocess = ColumnTransformer(
    transformers=[
        ("text", text_branch, TEXT),
        ("num",  num_branch,  num_cols),
        ("bert", bert_branch, bert_cols),
    ],
    remainder="drop",
    sparse_threshold=0.0,  # force dense output (MLP needs dense anyway)
)

# ---------------------------------------------------------
# 4) MLP + caching (to reuse TF-IDF/SVD work across candidates)
# ---------------------------------------------------------
os.makedirs("cache_p2", exist_ok=True)
cache = joblib.Memory(location="cache_p2", verbose=0)

pipe_mlp = Pipeline([
    ("prep", preprocess),
    ("clf", MLPClassifier(
        hidden_layer_sizes=(256,),
        activation="relu",
        learning_rate_init=1e-4,
        alpha=1e-4,
        batch_size=512,
        max_iter=150,
        early_stopping=True,
        n_iter_no_change=8,
        tol=1e-3,
        random_state=RANDOM_STATE
    ))
], memory=cache)

# Small, informed search around your known best settings
param_dist = {
    "prep__text__svd__n_components": [200, 300, 400],
    "clf__hidden_layer_sizes": [(256,), (256,128)],    # allow a 2-layer variant
    "clf__alpha": [1e-5, 5e-5, 1e-4],
    "clf__learning_rate_init": [1e-4, 3e-4],
}

# ---------------------------------------------------------
# 5) CV and search
# ---------------------------------------------------------
gkf = GroupKFold(n_splits=5)
scorer = make_scorer(roc_auc_score, needs_proba=True)

search = RandomizedSearchCV(
    estimator=pipe_mlp,
    param_distributions=param_dist,
    n_iter=12,                       # compact search = fast
    scoring=scorer,
    cv=gkf.split(x_train, y, groups=groups),
    n_jobs=-1,
    verbose=2,
    refit=True,
    random_state=RANDOM_STATE,
    pre_dispatch="2*n_jobs"
)
search.fit(x_train, y)

print("\nBest params:", search.best_params_)
print("Best mean CV AUROC:", round(search.best_score_, 4))

os.makedirs("outputs_p2", exist_ok=True)
pd.DataFrame(search.cv_results_).to_csv("outputs_p2/mlp_tfidf_bert_cv_results.csv", index=False)

# ---------------------------------------------------------
# 6) Confusion matrix on held-out predictions (grouped)
# ---------------------------------------------------------
print("\nComputing held-out predictions for confusion matrix...")
y_proba_cv = cross_val_predict(
    search.best_estimator_,
    x_train,
    y,
    cv=gkf.split(x_train, y, groups=groups),
    method="predict_proba",
    n_jobs=-1
)[:, 1]

y_pred_cv = (y_proba_cv >= 0.5).astype(int)
cm = confusion_matrix(y, y_pred_cv)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["KS2-3", "KS4-5"])
disp.plot(values_format="d")
plt.title("Confusion Matrix — TF-IDF+SVD + Numeric + BERT → MLP (GroupKFold)")
plt.savefig("outputs_p2/mlp_tfidf_bert_confusion_matrix.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 7) Final fit + test predictions
# ---------------------------------------------------------
print("\nTraining final model on FULL train and writing yproba2_test.txt ...")
best_model = search.best_estimator_
best_model.fit(x_train, y)
test_proba = best_model.predict_proba(x_test)[:, 1]
np.savetxt("yproba2_test.txt", test_proba, fmt="%.7f")
print("Saved yproba2_test.txt")
