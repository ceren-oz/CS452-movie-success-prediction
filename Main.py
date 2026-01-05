import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignore convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from Preprocessing2 import X_train, y_train
from Classification import run_classification
from ClassificationStatisticalTests import run_statistical_tests

results_df, cv_scores = run_classification(X_train, y_train)

print("\n=== Classification Results ===")
print(results_df)

run_statistical_tests(cv_scores)
