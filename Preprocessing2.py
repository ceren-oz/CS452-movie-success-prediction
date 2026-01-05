import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
    FunctionTransformer
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# --------------------------------------------------
# Load dataset (DO NOT MODIFY ORIGINAL TABLE)
# --------------------------------------------------
df = pd.read_csv("tmdb_movies_2000_2024.csv")

# --------------------------------------------------
# Split inputs and targets
# --------------------------------------------------
X = df.drop(columns=["revenue", "rating", "success_class"])
y_reg = df[["revenue", "rating"]]     # Regression target
y_clf = df["success_class"]            # Classification target

# --------------------------------------------------
# Column groups
# --------------------------------------------------
continuous_cols = ["budget", "runtime", "popularity"]
count_cols = ["vote_count", "cast_count", "crew_count"]
year_col = ["release_year"]

categorical_cols = ["original_language"]

list_columns = ["genres", "production_companies", "production_countries"]

# --------------------------------------------------
# Custom transformers (NO df modification)
# --------------------------------------------------
def zero_to_nan(X):
    X = X.copy()
    X[X == 0] = np.nan
    return X

def list_length(X):
    X = np.asarray(X).reshape(-1)
    return np.array([
        len(x) if isinstance(x, list) else 0 for x in X
    ]).reshape(-1, 1)



# --------------------------------------------------
# Pipelines
# --------------------------------------------------
continuous_pipeline = Pipeline([
    ("zero_to_nan", FunctionTransformer(zero_to_nan)),
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

count_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

year_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

list_pipeline = Pipeline([
    ("length", FunctionTransformer(list_length, validate=False)),
    ("scaler", StandardScaler())
])

# --------------------------------------------------
# ColumnTransformer (ALL columns handled)
# --------------------------------------------------
preprocessor = ColumnTransformer([
    ("continuous", continuous_pipeline, continuous_cols),
    ("counts", count_pipeline, count_cols),
    ("year", year_pipeline, year_col),
    ("categorical", categorical_pipeline, categorical_cols),
    ("genres", list_pipeline, ["genres"]),
    ("prod_companies", list_pipeline, ["production_companies"]),
    ("prod_countries", list_pipeline, ["production_countries"])
])

# --------------------------------------------------
# Train-test split (classification example)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_clf,
    test_size=0.2,
    random_state=42,
    stratify=y_clf
)




