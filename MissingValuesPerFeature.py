import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_csv("tmdb_movies_2000_2024.csv")

X = df.drop(columns=["revenue", "rating", "success_class"])
y_reg = df[["revenue", "rating"]]
y_clf = df["success_class"]

continuous_cols = ["budget", "runtime", "popularity"]
count_cols = ["vote_count", "cast_count", "crew_count"]
year_col = ["release_year"]
categorical_cols = ["original_language"]
list_columns = ["genres", "production_companies", "production_countries"]
# what abot production_companies, production_countries

# ----------------------------------
# Normalize fake missing values
# ----------------------------------
fake_zero_cols = ["budget", "runtime", "popularity"]
df[fake_zero_cols] = df[fake_zero_cols].replace(0, np.nan)


report = []

for col in df.columns:
    total = len(df)

    # 1. NaN count using value_counts
    vc = df[col].value_counts(dropna=False)
    nan_count = vc.get(np.nan, 0)

    # 2. Empty list count (only for list columns)
    empty_list_count = 0
    if col in list_columns:
        empty_list_count = df[col].apply(
            lambda x: isinstance(x, list) and len(x) == 0
        ).sum()

    # 3. Total missing
    total_missing = nan_count + empty_list_count

    report.append({
        "column": col,
        "nan_count": nan_count,
        "empty_list_count": empty_list_count,
        "total_missing": total_missing,
        "missing_percent": round(100 * total_missing / total, 2)
    })

missing_report = (
    pd.DataFrame(report)
    .sort_values("missing_percent", ascending=False)
    .reset_index(drop=True)
)

print(missing_report)



