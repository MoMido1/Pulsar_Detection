# Pulsar Detection

A machine-learning coursework project exploring classification of pulsar candidates in an imbalanced dataset. The work focuses on feature analysis, data transformation, dimensionality reduction, probabilistic classification, and comparing classifiers under different evaluation settings.

## Project contents

- [Report.pdf](Report.pdf) — project report describing the motivation, methodology, and results.
- [main.py](main.py) — main analysis and experiment workflow.
- The modules and Library directories contain classifier implementations and supporting utilities used by the project.
- The scripts expect input data under data/; check the referenced file paths before running.

## Methodology covered

The code and report cover the following areas:

- Exploratory analysis of feature distributions and class imbalance
- Feature normalization / Gaussianization steps
- Principal Component Analysis (PCA)
- Comparison of classifiers, including Gaussian models, logistic regression, SVM variants, and Gaussian Mixture Models
- Score calibration and evaluation of classification results

Consult the report and source code for the exact experimental settings and metrics. This README does not name a best-performing classifier or numerical score without summarizing the corresponding results from the report.

## Running the project

The main script loads training and test data from paths under data/ and imports scientific Python packages, plotting tools, and project-local modules. Before running it:

1. Check that the expected data files exist at the paths referenced by main.py.
2. Install the required Python dependencies in an isolated environment.
3. Run the script from the repository root so relative paths resolve as expected.

The repository does not currently document a verified, pinned environment here. For reproducibility, record the tested Python version and dependency versions before relying on results in a new environment.

## Notes and limitations

This is an experimental analysis project rather than a packaged prediction service. Results depend on the data split, preprocessing, calibration, and evaluation protocol. Treat the report as the source for the original experimental conclusions and inspect the implementation before reusing it.

## Future improvements

- Add a dependency file with tested versions.
- Document the dataset source, license, and expected file layout.
- Provide a single command for reproducing the main experiments.
- Summarize key metrics from the report with the evaluation protocol and class-imbalance context.
