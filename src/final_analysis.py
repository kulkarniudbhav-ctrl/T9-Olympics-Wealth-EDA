"""Final-stage quality checks, feature selection, PCA and dashboard artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "output" / "figures"
TABLES = ROOT / "output" / "tables"
DASH = ROOT / "dashboard"
for folder in (FIGURES, TABLES, DASH): folder.mkdir(parents=True, exist_ok=True)

def main() -> None:
    preclean = pd.read_csv(PROCESSED / "olympics_wealth_preclean.csv")
    data = pd.read_csv(PROCESSED / "olympics_wealth_country_edition.csv")
    # Missingness and two documented imputation alternatives on the same two WDI features.
    missing = preclean.isna().mean().mul(100).sort_values(ascending=False).rename("missing_pct").reset_index().rename(columns={"index": "feature"})
    missing.to_csv(TABLES / "missingness_audit.csv", index=False)
    target = preclean[["gdp_per_capita_usd", "population", "Year", "Season"]].copy()
    global_median = target[["gdp_per_capita_usd", "population"]].fillna(target[["gdp_per_capita_usd", "population"]].median())
    by_year_season = target.groupby(["Year", "Season"])[["gdp_per_capita_usd", "population"]].transform("median")
    grouped_median = target[["gdp_per_capita_usd", "population"]].fillna(by_year_season).fillna(target[["gdp_per_capita_usd", "population"]].median())
    comparison = pd.DataFrame({
        "method": ["global median", "Olympic year-season median"],
        "gdp_per_capita_median": [global_median.gdp_per_capita_usd.median(), grouped_median.gdp_per_capita_usd.median()],
        "population_median": [global_median.population.median(), grouped_median.population.median()],
        "rows_imputed_gdp": [int(preclean.gdp_per_capita_usd.isna().sum())]*2,
        "rows_imputed_population": [int(preclean.population.isna().sum())]*2,
    })
    comparison.to_csv(TABLES / "imputation_comparison.csv", index=False)
    # Outlier comparison: IQR and modified z-score. Values are flagged, not deleted.
    rate = data.medals_per_million
    q1, q3 = rate.quantile([.25,.75]); iqr = q3-q1
    iqr_flag = (rate < q1-1.5*iqr) | (rate > q3+1.5*iqr)
    mad = np.median(np.abs(rate - np.median(rate)))
    mod_z = .6745*(rate-np.median(rate))/mad if mad else pd.Series(0, index=rate.index)
    mz_flag = mod_z.abs() > 3.5
    outliers = pd.DataFrame({"method": ["IQR", "Modified z-score", "Both"], "flagged_rows": [int(iqr_flag.sum()), int(mz_flag.sum()), int((iqr_flag&mz_flag).sum())]})
    outliers.to_csv(TABLES / "outlier_comparison.csv", index=False)
    # Feature selection: correlation and VIF (from inverse correlation matrix).
    features = ["log_gdp_per_capita", "log_population", "medals_per_million", "gold_share", "host_flag"]
    X = data[features].replace([np.inf, -np.inf], np.nan)
    X_imp = SimpleImputer(strategy="median").fit_transform(X)
    corr = pd.DataFrame(X_imp, columns=features).corr()
    corr.to_csv(TABLES / "feature_correlation.csv")
    inv = np.linalg.pinv(corr.to_numpy())
    vif = pd.DataFrame({"feature": features, "vif": np.diag(inv)})
    vif.to_csv(TABLES / "vif.csv", index=False)
    pca = PCA().fit(StandardScaler().fit_transform(X_imp))
    pca_table = pd.DataFrame({"component": [f"PC{i}" for i in range(1, len(features)+1)], "explained_variance_ratio": pca.explained_variance_ratio_, "cumulative_explained_variance": pca.explained_variance_ratio_.cumsum()})
    pca_table.to_csv(TABLES / "pca_summary.csv", index=False)
    pd.DataFrame(pca.components_.T, index=features, columns=pca_table.component).to_csv(TABLES / "pca_loadings.csv")
    # Residual overperformance from simple log-linear model, for descriptive benchmarking.
    design = np.column_stack([np.ones(len(data)), data.log_gdp_per_capita, data.log_population])
    coef, *_ = np.linalg.lstsq(design, data.log_medals, rcond=None)
    data["predicted_log_medals"] = design @ coef
    data["overperformance_residual"] = data.log_medals - data.predicted_log_medals
    top = data.sort_values("overperformance_residual", ascending=False).groupby("NOC", as_index=False).first().head(15)
    top[["NOC","Year","Season","medals","medals_per_million","overperformance_residual"]].to_csv(TABLES / "top_overperformers.csv", index=False)
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8,4.8)); ax.barh(missing.feature.head(12)[::-1], missing.missing_pct.head(12)[::-1], color="#2563eb"); ax.set_xlabel("Missing values (%)"); ax.set_title("Missingness before analytical-sample restriction"); fig.tight_layout(); fig.savefig(FIGURES / "missingness.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5,4.5)); ax.bar(outliers.method, outliers.flagged_rows, color=["#f59e0b", "#ef4444", "#7c3aed"]); ax.set_ylabel("Country-edition rows flagged"); ax.set_title("Outlier methods compared - medal rate per million"); fig.tight_layout(); fig.savefig(FIGURES / "outlier_methods.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.5,4.5)); ax.plot(range(1,6), pca.explained_variance_ratio_.cumsum(), marker="o", color="#1d4ed8"); ax.bar(range(1,6), pca.explained_variance_ratio_, alpha=.35, color="#60a5fa"); ax.set_xticks(range(1,6)); ax.set_ylim(0,1.05); ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance ratio / cumulative"); ax.set_title("PCA scree and cumulative explained variance"); fig.tight_layout(); fig.savefig(FIGURES / "pca_scree.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5)); plot_top = top.sort_values("overperformance_residual"); ax.barh(plot_top.NOC, plot_top.overperformance_residual, color="#15803d"); ax.set_xlabel("Residual log-medal performance"); ax.set_title("Top descriptive overperformers after GDP/population adjustment"); fig.tight_layout(); fig.savefig(FIGURES / "overperformers.png", dpi=180); plt.close(fig)
    result = {"imputation": comparison.to_dict("records"), "outliers": outliers.to_dict("records"), "pca": pca_table.to_dict("records"), "top_overperformers": top[["NOC","Year","Season","medals","overperformance_residual"]].to_dict("records")}
    (PROCESSED / "final_analysis_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    # One-page static dashboard; portable and dependency-free.
    core = json.loads((PROCESSED / "analysis_results.json").read_text())
    html = f'''<!doctype html><html><head><meta charset="utf-8"><title>Olympics x Wealth Dashboard</title><style>body{{font-family:Arial,sans-serif;margin:32px;color:#172033;background:#f7f9fc}}h1{{color:#102a56}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}.card{{background:white;border-radius:10px;padding:16px;box-shadow:0 1px 5px #dbe3f0}}.n{{font-size:28px;font-weight:bold;color:#1d4ed8}}img{{max-width:48%;background:white;margin:1%}}small{{color:#536174}}</style></head><body><h1>Olympics x wealth: evidence dashboard</h1><p>Country-Olympic-edition analysis, 1960-2016. Associations are descriptive, not causal.</p><div class="grid"><div class="card"><small>Complete records</small><div class="n">{core['sample']['country_edition_rows']:,}</div></div><div class="card"><small>GDP-medal-rate Spearman rho</small><div class="n">{core['h1_gdp_medals_per_million']['spearman_rho']:.3f}</div></div><div class="card"><small>Host advantage p-value</small><div class="n">{core['h3_host_advantage']['p_value']:.4f}</div></div><div class="card"><small>GDP/population model R-squared</small><div class="n">{core['log_linear_model']['r_squared']:.3f}</div></div></div><h2>Core evidence</h2><img src="../output/figures/gdp_medal_rate.png"><img src="../output/figures/income_quartile_medals.png"><h2>Quality and feature evidence</h2><img src="../output/figures/missingness.png"><img src="../output/figures/pca_scree.png"><p><b>Recommendations:</b> benchmark medal performance per population, compare host editions to country baselines rather than raw totals, and use GDP/population as context rather than deterministic targets.</p></body></html>'''
    (DASH / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps(result, indent=2))

if __name__ == "__main__": main()
