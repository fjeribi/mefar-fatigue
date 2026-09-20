"""Conventional classifiers with the exact settings used in the paper (Table 5).

P1 models are fitted on pre-standardised inputs (scaled models) or raw inputs
(tree ensembles) by the calling script. P2 models wrap scaling inside a
Pipeline so that the scaler is fitted on the training participants only.
"""
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from .config import SEED

# Models that receive standardised inputs under P1
P1_SCALED = {"Logistic Regression", "KNN", "SVM", "Gaussian Naive Bayes"}


def p1_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2),
        "SVM": SVC(kernel="rbf", probability=True, random_state=SEED),
        "Gaussian Naive Bayes": GaussianNB(),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",
            random_state=SEED, n_jobs=-1),
    }


def _scaled(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def p2_models(include_rf=False, rf_params=None):
    models = {
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, objective="binary:logistic", eval_metric="logloss",
            random_state=SEED, n_jobs=-1, tree_method="hist"),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=3, random_state=SEED),
        "SVM": _scaled(SVC(kernel="rbf", probability=False, random_state=SEED)),
        "KNN": _scaled(KNeighborsClassifier(n_neighbors=5, n_jobs=-1)),
        "Gaussian Naive Bayes": _scaled(GaussianNB()),
        "Logistic Regression": _scaled(LogisticRegression(max_iter=2000, random_state=SEED)),
    }
    if include_rf:
        params = rf_params or dict(n_estimators=300, max_depth=None)
        models["Random Forest"] = RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    return models


def rf_candidates():
    """Candidate Random Forest configurations; one is selected on fold 1 (M5)."""
    return [
        dict(n_estimators=200, max_depth=None, min_samples_leaf=1),
        dict(n_estimators=400, max_depth=20, min_samples_leaf=2),
        dict(n_estimators=400, max_depth=None, min_samples_leaf=5),
    ]


def score(model, X):
    """Probability of the positive class, or decision-function value if unavailable."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except AttributeError:
            pass
    return model.decision_function(X)
