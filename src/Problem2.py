import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_predict
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, make_scorer
from sklearn.neural_network import MLPClassifier

RANDOM_STATE = 0
np.random.seed(RANDOM_STATE)

# ---------------------------------------------------------
# 1) Load data
# ---------------------------------------------------------
data_dir = 'data'
x_train = pd.read_csv(os.path.join(data_dir, 'x_train.csv'))
y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv'))
x_test = pd.read_csv(os.path.join(data_dir, 'x_test.csv'))

y = y_train["Coarse Label"].map({"Key Stage 2-3": 0, "Key Stage 4-5": 1}).values
groups = x_train["author"].values

KEYS = ["author", "title", "passage_id"]
TEXT = "text"
num_cols = [c for c in x_train.columns if c not in KEYS + [TEXT]]

# ---------------------------------------------------------
# 2) Feature representation
# ---------------------------------------------------------
tfidf = TfidfVectorizer(
    lowercase=True,
    strip_accents="unicode",
    token_pattern=r"\b[a-zA-Z]+\b",
    ngram_range=(1, 2),
    min_df=3,
    max_df=0.9,
    max_features=50000,
)

num_pipe = Pipeline([("scaler", StandardScaler(with_mean=False))])

preprocess = ColumnTransformer(
    transformers=[
        ("text", tfidf, TEXT),
        ("num", num_pipe, num_cols),
    ],
    remainder="drop",
    sparse_threshold=0.3,
)

# ---------------------------------------------------------
# 3) Define MLP pipeline + hyperparameter space
# ---------------------------------------------------------
pipe_mlp = Pipeline([
    ("prep", preprocess),
    ("clf", MLPClassifier(
        max_iter=150,
        early_stopping=True,
        n_iter_no_change=8,
        random_state=0,
    )),
])

param_dist_refined = {
    "clf__hidden_layer_sizes": [(256,), (256,128), (128,64)],
    "clf__alpha": [1e-5, 5e-5, 1e-4],
    "clf__learning_rate_init": [1e-4, 3e-4, 5e-4],
    "clf__activation": ["relu"],
}

# ---------------------------------------------------------
# 4) Cross-validation setup
# ---------------------------------------------------------
gkf = GroupKFold(n_splits=5)

# ---------------------------------------------------------
# 5) RandomizedSearchCV for hyperparameter tuning
# ---------------------------------------------------------
search = RandomizedSearchCV(
    pipe_mlp, param_dist_refined,
    n_iter=12, cv=gkf.split(x_train, y, groups=groups),
    scoring=scorer, n_jobs=-1, random_state=0, verbose=2
)
search.fit(x_train, y)

print("\nBest hyperparameters:", search.best_params_)
print("Best mean CV AUROC:", round(search.best_score_, 4))

# Save CV results
os.makedirs("outputs_p2", exist_ok=True)
pd.DataFrame(search.cv_results_).to_csv("outputs_p2/mlp_cv_results.csv", index=False)

# ---------------------------------------------------------
# 6) Confusion matrix on held-out predictions (CV)
# ---------------------------------------------------------
print("\nGenerating confusion matrix on held-out predictions...")
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
plt.title("MLP Confusion Matrix (GroupKFold held-out)")
plt.savefig("outputs_p2/mlp_confusion_matrix.png", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------
# 7) Final model fit + test predictions
# ---------------------------------------------------------
print("\nTraining final MLP on full data...")
best_model = search.best_estimator_
best_model.fit(x_train, y)

test_proba = best_model.predict_proba(x_test)[:, 1]
np.savetxt("yproba2_test.txt", test_proba, fmt="%.7f")
print("Saved yproba2_test.txt for leaderboard submission.")

# ---------------------------------------------------------
# 8) Optional: summarize CV performance curve
# ---------------------------------------------------------
print("\nTop 5 CV configs by AUROC:")
summary = pd.DataFrame(search.cv_results_).sort_values("mean_test_score", ascending=False).head(5)
print(summary[["mean_test_score", "param_clf__hidden_layer_sizes", "param_clf__alpha",
               "param_clf__learning_rate_init", "param_clf__activation"]])