from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import NearestCentroid
from sklearn.svm import SVC

import pandas as pd

from Preprocessing import preprocessor
from ClassificationEvaluation import evaluate_classifier, get_cv_accuracy_scores


def run_classification(X, y):
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000),
            {"clf__C": [0.01, 0.1, 1, 10]}
        ),
        "LDA": (
            LinearDiscriminantAnalysis(),
            {}
        ),
        "Nearest Centroid": (
            NearestCentroid(),
            {"clf__metric": ["euclidean", "manhattan"]}
        ),
        "SVM (RBF)": (
            SVC(),
            {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", "auto"]}
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    rows = []

    cv_scores = {}

    for name, (model, params) in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("clf", model)
        ])

        grid = GridSearchCV(
            pipe,
            params,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )

        grid.fit(X, y)

        metrics = evaluate_classifier(
            grid.best_estimator_,
            X,
            y,
            cv
        )

        scores = get_cv_accuracy_scores(
            grid.best_estimator_,
            X,
            y,
            cv
        )

        cv_scores[name] = scores

        metrics["Model"] = name
        rows.append(metrics)

        print(f"{name}: best CV accuracy = {grid.best_score_:.4f}")

    results_df = pd.DataFrame(rows).set_index("Model")
    return results_df, cv_scores
