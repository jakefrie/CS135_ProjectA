import numpy as np
import sklearn.linear_model
import sklearn.pipeline

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression


def make_pipeline():
    my_bow_classifier_pipeline = sklearn.pipeline.Pipeline([
        ('my_bow_feature_extractor', CountVectorizer(min_df=1, max_df=1.0, ngram_range=(1,1))),
        ('my_classifier', sklearn.linear_model.LogisticRegression(C=1.0, max_iter=20, random_state=101)),
    ])

def fit_grid(list_of_training_text_reviews, y_tr_N):
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
    grid_searcher = sklearn.model_selection.GridSearchCV(
        pipe,
        my_parameter_grid_by_name,
        scoring=my_scoring_metric_name,
        cv=my_splitter,
        refit=False)
     
    grid_searcher.fit(list_of_training_text_reviews, y_tr_N)
    return grid_searcher