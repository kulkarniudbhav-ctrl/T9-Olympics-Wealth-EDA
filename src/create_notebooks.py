"""Create the four runnable project notebooks from the reproducible pipeline."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"; NB.mkdir(exist_ok=True)

def write(name, title, markdown, code):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}}
    nb.cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{markdown}"), nbf.v4.new_code_cell(code)]
    nbf.write(nb, NB / name)

common = """from pathlib import Path
import os, sys
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks': ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'tmp' / 'matplotlib'))
"""
write("01_acquisition_integration.ipynb", "01 - Acquisition and integration", "Downloads are reproducible through `src/fetch_data.py`. This notebook runs the integration pipeline, deduplicates team medals once per Games-NOC-Event-Medal, maps NOC to ISO3, and writes the country-edition dataset.", common + """\nfrom build_analysis import main
main()
import pandas as pd
integrated = pd.read_csv(ROOT / 'data/processed/olympics_wealth_country_edition.csv')
print('Integrated country-edition rows:', len(integrated))
display(integrated.head())
""")
write("02_quality_cleaning.ipynb", "02 - Quality and cleaning", "Reviews the generated crosswalk, unmatched NOCs, missingness, duplicate rule, and cleaned analytical dataset. The raw source files remain unchanged.", common + """\nimport pandas as pd
quality = pd.read_csv(ROOT / 'data/processed/quality_audit.csv')
unmatched = pd.read_csv(ROOT / 'data/processed/unmatched_nocs.csv')
data = pd.read_csv(ROOT / 'data/processed/olympics_wealth_country_edition.csv')
print('Analytical rows:', len(data), '| exact duplicate rows:', data.duplicated().sum())
display(quality)
print('Unmatched NOCs retained for audit:', len(unmatched))
display(unmatched)
""")
write("03_eda_hypothesis_tests.ipynb", "03 - EDA and hypothesis tests", "Loads actual results created by the pipeline. Tests use Spearman correlations, a paired Wilcoxon host comparison, and Kruskal-Wallis across GDP-per-capita quartiles. Results are observational, not causal.", common + """\nimport json
from IPython.display import Image, display
results = json.loads((ROOT / 'data/processed/analysis_results.json').read_text())
results
display(Image(filename=ROOT / 'output/figures/gdp_medal_rate.png'))
display(Image(filename=ROOT / 'output/figures/income_quartile_medals.png'))
""")
write("04_features_selection_pca.ipynb", "04 - Features, selection and PCA", "The dataset contains engineered medal rate, log GDP/capita, log population, host indicator, gold share, and GDP-per-capita quartile features. This starter cell prepares numeric features for PCA after imputation/scaling decisions are documented.", common + """\nimport pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
data = pd.read_csv(ROOT / 'data/processed/olympics_wealth_country_edition.csv')
features = ['log_gdp_per_capita','log_population','medals_per_million','gold_share','host_flag']
X = data[features].replace([float('inf'), float('-inf')], float('nan')).dropna()
pcs = PCA().fit(StandardScaler().fit_transform(X))
pd.DataFrame({'component': range(1, len(features)+1), 'explained_variance_ratio': pcs.explained_variance_ratio_, 'cumulative_variance': pcs.explained_variance_ratio_.cumsum()})
""")
print('Created:', *sorted(p.name for p in NB.glob('*.ipynb')), sep='\n- ')
