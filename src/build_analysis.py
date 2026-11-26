"""Build the integrated T9 Olympics x wealth dataset and statistical outputs."""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED, FIGURES = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "output" / "figures"
PROCESSED.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

IND = {"NY.GDP.PCAP.CD": "gdp_per_capita_usd", "SP.POP.TOTL": "population"}

# Historical teams are retained in the audit and mapped only for a clearly labelled
# sensitivity analysis; no historical entity is silently recoded in the raw files.
HISTORICAL_NOC = {"URS": "RUS", "GDR": "DEU", "FRG": "DEU", "EUN": "RUS", "YUG": "SRB", "TCH": "CZE", "SCG": "SRB"}
NOC_ALIASES = {"ANT": "ATG", "CGO": "COG", "IVB": "VGB", "SKN": "KNA", "VIN": "VCT", "YAR": "YEM", "YMD": "YEM"}
ALIASES = {
    "unitedstates": "USA", "unitedkingdom": "GBR", "russia": "RUS", "southkorea": "KOR", "northkorea": "PRK",
    "iran": "IRN", "venezuela": "VEN", "bolivia": "BOL", "moldova": "MDA", "tanzania": "TZA", "laos": "LAO",
    "brunei": "BRN", "vietnam": "VNM", "czechrepublic": "CZE", "slovakia": "SVK", "syria": "SYR",
    "ivorycoast": "CIV", "capeverde": "CPV", "swaziland": "SWZ", "micronesia": "FSM", "gambia": "GMB",
    "bahamas": "BHS", "gambia": "GMB", "congodrc": "COD", "democraticrepublicofthecongo": "COD",
    "congo": "COG", "kyrgyzstan": "KGZ", "macedonia": "MKD", "palestine": "PSE", "taiwan": "TWN",
    "kosovo": "XKX", "hongkong": "HKG", "capeverde": "CPV", "easttimor": "TLS", "northmacedonia": "MKD",
    "saoandprincipe": "STP", "stvincentandthegrenadines": "VCT", "antiguaandbarbuda": "ATG",
    "trinidadandtobago": "TTO", "saintkittsandnevis": "KNA", "dominicanrepublic": "DOM",
}
HOSTS = {
    (1960,"Summer"):"ITA", (1960,"Winter"):"USA", (1964,"Summer"):"JPN", (1964,"Winter"):"AUT",
    (1968,"Summer"):"MEX", (1968,"Winter"):"FRA", (1972,"Summer"):"DEU", (1972,"Winter"):"JPN",
    (1976,"Summer"):"CAN", (1976,"Winter"):"AUT", (1980,"Summer"):"RUS", (1980,"Winter"):"USA",
    (1984,"Summer"):"USA", (1984,"Winter"):"BIH", (1988,"Summer"):"KOR", (1988,"Winter"):"CAN",
    (1992,"Summer"):"ESP", (1992,"Winter"):"FRA", (1994,"Winter"):"NOR", (1996,"Summer"):"USA",
    (1998,"Winter"):"JPN", (2000,"Summer"):"AUS", (2002,"Winter"):"USA", (2004,"Summer"):"GRC",
    (2006,"Winter"):"ITA", (2008,"Summer"):"CHN", (2010,"Winter"):"CAN", (2012,"Summer"):"GBR",
    (2014,"Winter"):"RUS", (2016,"Summer"):"BRA",
}

def norm(value: str) -> str:
    return re.sub(r"[^a-z]", "", str(value).lower())

def ols(y: pd.Series, x1: pd.Series, x2: pd.Series) -> dict:
    frame = pd.concat([y, x1, x2], axis=1).dropna()
    target = frame.iloc[:, 0].to_numpy(); x = np.column_stack([np.ones(len(frame)), frame.iloc[:, 1], frame.iloc[:, 2]])
    beta, *_ = np.linalg.lstsq(x, target, rcond=None); pred = x @ beta
    ss_res = ((target-pred)**2).sum(); ss_tot = ((target-target.mean())**2).sum()
    return {"n": int(len(frame)), "intercept": float(beta[0]), "log_gdp_beta": float(beta[1]), "log_population_beta": float(beta[2]), "r_squared": float(1-ss_res/ss_tot)}

def main() -> None:
    athletes = pd.read_csv(RAW / "athlete_events.csv")
    noc = pd.read_csv(RAW / "noc_regions.csv")
    wdi = pd.read_csv(RAW / "wdi_indicators.csv")
    olympic = athletes.loc[(athletes["Year"] >= 1960) & athletes["Medal"].notna(), ["ID","Year","Season","Games","NOC","Event","Medal"]].copy()
    medals = olympic.drop_duplicates(["Games", "NOC", "Event", "Medal"])
    medal_counts = medals.groupby(["Year","Season","Games","NOC"], as_index=False).agg(
        medals=("Medal","size"), gold=("Medal", lambda s: (s == "Gold").sum()),
        silver=("Medal", lambda s: (s == "Silver").sum()), bronze=("Medal", lambda s: (s == "Bronze").sum()),
    )
    wdi = wdi[wdi["indicator_code"].isin(IND)].copy()
    wdi["year"] = pd.to_numeric(wdi["year"], errors="coerce")
    macro = wdi.pivot_table(index=["iso3", "country", "year"], columns="indicator_code", values="value", aggfunc="first").reset_index().rename(columns=IND)
    names = {norm(country): iso for country, iso in macro[["country", "iso3"]].drop_duplicates().itertuples(index=False)}
    valid_iso = set(macro["iso3"])
    noc["norm_region"] = noc["region"].map(norm)
    noc["iso3"] = noc["norm_region"].map(names).fillna(noc["norm_region"].map(ALIASES))
    noc["iso3"] = noc["iso3"].fillna(noc["NOC"].where(noc["NOC"].isin(valid_iso))).fillna(noc["NOC"].map(NOC_ALIASES))
    noc["iso3"] = noc.apply(lambda r: HISTORICAL_NOC.get(r["NOC"], r["iso3"]), axis=1)
    noc["mapping_source"] = np.where(noc["NOC"].isin(HISTORICAL_NOC), "historical sensitivity mapping", np.where(noc["norm_region"].isin(ALIASES), "documented alias", "WDI country-name match"))
    joined = medal_counts.merge(noc[["NOC","region","iso3","mapping_source"]], on="NOC", how="left")
    joined = joined.merge(macro, left_on=["iso3","Year"], right_on=["iso3","year"], how="left")
    joined["host_flag"] = [int(HOSTS.get((year, season)) == iso) for year, season, iso in joined[["Year","Season","iso3"]].itertuples(index=False)]
    joined["medals_per_million"] = joined["medals"] / joined["population"] * 1_000_000
    joined["log_gdp_per_capita"] = np.log1p(joined["gdp_per_capita_usd"])
    joined["log_population"] = np.log1p(joined["population"])
    joined["log_medals"] = np.log1p(joined["medals"])
    joined["gold_share"] = joined["gold"] / joined["medals"]
    joined["income_quartile"] = pd.qcut(joined["gdp_per_capita_usd"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"], duplicates="drop")
    joined.to_csv(PROCESSED / "olympics_wealth_preclean.csv", index=False)
    complete = joined.dropna(subset=["gdp_per_capita_usd", "population"]).copy()
    join_audit = pd.DataFrame([
        {"step": "Athlete medal rows, 1960-2016", "rows": len(olympic), "detail": "Athlete-level medal rows before team-event deduplication"},
        {"step": "Distinct medals", "rows": len(medals), "detail": "Unique Games + NOC + Event + Medal"},
        {"step": "Country-edition medal rows", "rows": len(medal_counts), "detail": "Aggregated NOC x Games"},
        {"step": "NOC-to-ISO3 matched", "rows": int(joined.iso3.notna().sum()), "detail": f"{int(joined.iso3.isna().sum())} rows lack a mappable NOC"},
        {"step": "Complete WDI analytical sample", "rows": len(complete), "detail": "Non-missing GDP per capita and population"},
    ])
    join_audit.to_csv(PROCESSED / "join_audit.csv", index=False)
    complete.to_csv(PROCESSED / "olympics_wealth_country_edition.csv", index=False)
    noc.to_csv(PROCESSED / "noc_to_iso3_crosswalk.csv", index=False)
    unmatched = noc[noc["iso3"].isna()][["NOC","region","notes"]]
    unmatched.to_csv(PROCESSED / "unmatched_nocs.csv", index=False)
    quality = pd.DataFrame({
        "feature": complete.columns,
        "missing_count": [int(complete[c].isna().sum()) for c in complete.columns],
        "missing_pct": [round(float(complete[c].isna().mean()*100),2) for c in complete.columns],
        "dtype": [str(complete[c].dtype) for c in complete.columns],
    })
    quality.to_csv(PROCESSED / "quality_audit.csv", index=False)
    rho_gdp = stats.spearmanr(complete["gdp_per_capita_usd"], complete["medals_per_million"], nan_policy="omit")
    rho_pop = stats.spearmanr(complete["population"], complete["medals"], nan_policy="omit")
    groups = [g["medals_per_million"].values for _, g in complete.groupby("income_quartile", observed=True)]
    kw = stats.kruskal(*groups)
    host_rows = []
    for iso, group in complete.groupby("iso3"):
        host = group.loc[group.host_flag.eq(1), "medals_per_million"]
        nonhost = group.loc[group.host_flag.eq(0), "medals_per_million"]
        if len(host) and len(nonhost): host_rows.append((iso, float(host.mean()), float(nonhost.median())))
    paired = pd.DataFrame(host_rows, columns=["iso3","host_rate","typical_nonhost_rate"])
    wilcoxon = stats.wilcoxon(paired["host_rate"], paired["typical_nonhost_rate"], alternative="greater") if len(paired) >= 6 else None
    stats_out = {
        "sample": {"country_edition_rows": int(len(complete)), "years": [int(complete.Year.min()), int(complete.Year.max())], "nocs_with_iso3": int(noc.iso3.notna().sum()), "unmatched_nocs": int(noc.iso3.isna().sum()), "athlete_medal_rows_before_dedup": int(len(olympic)), "team_and_individual_medal_rows_after_dedup": int(len(medals))},
        "h1_gdp_medals_per_million": {"spearman_rho": float(rho_gdp.statistic), "p_value": float(rho_gdp.pvalue)},
        "h2_population_total_medals": {"spearman_rho": float(rho_pop.statistic), "p_value": float(rho_pop.pvalue)},
        "h3_host_advantage": {"paired_countries": int(len(paired)), "median_host_rate": float(paired.host_rate.median()), "median_typical_nonhost_rate": float(paired.typical_nonhost_rate.median()), "wilcoxon_stat": float(wilcoxon.statistic) if wilcoxon else None, "p_value": float(wilcoxon.pvalue) if wilcoxon else None},
        "h4_income_quartiles": {"kruskal_stat": float(kw.statistic), "p_value": float(kw.pvalue)},
        "log_linear_model": ols(complete["log_medals"], complete["log_gdp_per_capita"], complete["log_population"]),
    }
    (PROCESSED / "analysis_results.json").write_text(json.dumps(stats_out, indent=2), encoding="utf-8")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8,5))
    scatter = ax.scatter(complete["gdp_per_capita_usd"], complete["medals_per_million"], c=complete["host_flag"], alpha=.55, cmap="coolwarm")
    ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=1); ax.set_xlabel("GDP per capita (US$, log scale)"); ax.set_ylabel("Medals per million people"); ax.set_title("Olympic medal rate and GDP per capita (1960-2016)")
    fig.colorbar(scatter, ax=ax, label="Host flag"); fig.tight_layout(); fig.savefig(FIGURES / "gdp_medal_rate.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5)); order = ["Q1 lowest","Q2","Q3","Q4 highest"]
    ax.boxplot([complete.loc[complete.income_quartile.eq(q), "medals_per_million"].dropna() for q in order], tick_labels=order, showfliers=False)
    ax.set_ylabel("Medals per million people"); ax.set_title("Medal-rate distribution by GDP-per-capita quartile"); fig.tight_layout(); fig.savefig(FIGURES / "income_quartile_medals.png", dpi=180); plt.close(fig)
    print(json.dumps(stats_out, indent=2))

if __name__ == "__main__":
    main()
