# Multiple Myeloma Risk Classifier

This repository contains a Python script for classifying multiple myeloma patients into ultra-high-risk and low-risk categories based on RNA-seq gene expression data and survival information.

## Data Requirements
- Expression data: TSV/CSV with GENE_ID as rows, sample IDs (e.g., MMRF_XXXX_1_BM) as columns. Values are TPM.
- Survival data: CSV with columns `public_id` (e.g., MMRF_XXXX) and `ttcos` (survival time in months).

## Installation
pip install -r requirements.txt

## Usage
python mmrf_ml_classifier.py --expression_file path/to/exp.tpm.tsv --survival_file path/to/survival_months.csv --output_dir output/

## Output
- Intermediate CSV files and plots in the specified output directory.
- Console output with model performance.

## Notes
- This is a simplified classification approach. For real clinical use, consider censoring in survival data and consult domain experts.
- Immunoglobulin genes are fetched via HGNC API and removed.
