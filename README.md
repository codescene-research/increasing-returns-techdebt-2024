# Investigating the Returns of Highly Maintainable Code 2024

This repo contains a complete replication package, including raw data and scripts for the statistical analysis, for the paper "Increasing, not Diminishing: Investigating the Returns of Highly Maintainable Code" accepted at the [7th International Conference on Technical Debt (TechDebt)](https://conf.researchr.org/home/TechDebt-2024), Lisbon, Portugal, May 14-15, 2024.

## Authors

Markus Borg, Ilyana Pruvost, Enys Mones, and Adam Tornhill

## Abstract

Understanding and effectively managing Technical Debt (TD) remains a vital challenge in software engineering. While many studies on code-level TD have been published, few illustrate the business impact of low-quality source code. In this study, we combine two publicly available datasets and study the association between code quality on the one hand, and defect density and implementation time on the other hand. We introduce a value-creation model, derived from regression analyses, to explore relative changes from a baseline. Our results show how the association varies for different intervals of code quality. Furthermore, the value model suggests strong non-linearities at the ends of the code quality spectrum. We discuss the findings in light of TD and the broken windows theory. Finally, we argue that the value-creation model can be used to initiate discussions regarding the return on investment in refactoring efforts.

## Repository Content
- Three Jupyter Notebooks.
	- TechDebt24_increasing_returns.ipynb: The main Notebook that follows the paper.
	- regression_modeling.ipynb: Describes our work on candidate regression models.
	- data_pooling.ipynb: Presents our work on merging and preprocessing datasets.
- Functions.py: Various helper functions.
- increasing_returns.csv: Anonomous data in a csv-file.
