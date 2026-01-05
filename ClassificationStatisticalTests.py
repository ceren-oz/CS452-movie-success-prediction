import itertools
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon


def run_statistical_tests(cv_scores_dict):
    """
    cv_scores_dict:
        {
            "Logistic Regression": array([...]),
            "LDA": array([...]),
            ...
        }
    """

    # ---------- Friedman test ----------
    scores_list = list(cv_scores_dict.values())
    stat, p_value = friedmanchisquare(*scores_list)

    print("\n=== Friedman Test ===")
    print(f"Statistic = {stat:.4f}")
    print(f"p-value   = {p_value:.4f}")

    if p_value >= 0.05:
        print("No statistically significant difference found.")
        return
    else:
        print("Significant difference detected (p < 0.05)")

    # ---------- Post-hoc Wilcoxon tests ----------
    print("\n=== Pairwise Wilcoxon Tests ===")

    results = []

    for (name1, scores1), (name2, scores2) in itertools.combinations(
        cv_scores_dict.items(), 2
    ):
        w_stat, p = wilcoxon(scores1, scores2)

        results.append({
            "Model 1": name1,
            "Model 2": name2,
            "p-value": p
        })

        signif = "YES" if p < 0.05 else "NO"
        print(f"{name1} vs {name2}: p = {p:.4f} → Significant: {signif}")

    return pd.DataFrame(results)
