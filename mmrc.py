# mmrf_ml_classifier.py
# 
# This script implements a classification algorithm for multiple myeloma risk stratification
# using RNA-seq gene expression data and patient survival data.
# 
# The algorithm performs the following steps:
# 1. Load and preprocess gene expression data (TPM values):
#    - Remove non-primary samples (keeping only '_1_BM' columns).
#    - Apply log2(TPM + 1) normalization.
#    - Filter genes with low variance (threshold: 0.25).
# 2. Transpose the data (samples as rows, genes as columns).
# 3. Remove immunoglobulin genes using HGNC API.
# 4. Merge with survival data.
# 5. Label patients based on survival time:
#    - Ultra-high-risk: survival < 24 months (label 1)
#    - Low-risk: survival >= 24 months (label 0)
#    Note: This assumes no censoring or handles it simplistically. In real survival analysis,
#          censored data should be considered carefully.
# 6. Perform feature selection using SelectKBest (top 30 genes).
# 7. Train and evaluate an SVM classifier with grid search and stratified cross-validation.
# 8. Output evaluation metrics (accuracy, ROC-AUC, classification report).
# 
# Dependencies:
# - pandas
# - numpy
# - matplotlib
# - scikit-learn
# - requests
# 
# Usage:
# python mmrc.py --expression_file path/to/exp.tpm.tsv --survival_file path/to/survival_months.csv --output_dir path/to/output
# 
# Output:
# - Processed data files in output_dir (e.g., normalized.csv, filtered.csv, etc.)
# - Plots: variance_distribution.png
# - Console: Model performance metrics

import argparse
import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import classification_report

def parse_arguments():
    parser = argparse.ArgumentParser(description="Multiple Myeloma Risk Classifier")
    parser.add_argument("--expression_file", required=True, help="Path to gene expression TPM file (TSV/CSV, genes as rows, samples as columns)")
    parser.add_argument("--survival_file", required=True, help="Path to survival data CSV (columns: public_id, ttcos for survival time in months)")
    parser.add_argument("--output_dir", default="output", help="Directory to save output files and plots")
    return parser.parse_args()

def load_and_preprocess_expression_data(file_path, output_dir):
    """Load expression data, remove non-primary samples, normalize, and save."""
    df = pd.read_csv(file_path, sep='\t' if file_path.endswith('.tsv') else ',')
    
    # Remove non-primary samples (keep only '_1_BM')
    columns_to_keep = [col for col in df.columns if '_1_BM' in col or col == 'GENE_ID']
    df = df[columns_to_keep]
    
    # Normalize: log2(TPM + 1), excluding GENE_ID
    df_normalized = df.copy()
    df_normalized.iloc[:, 1:] = np.log2(df_normalized.iloc[:, 1:] + 1)
    
    normalized_path = os.path.join(output_dir, "normalized.csv")
    df_normalized.to_csv(normalized_path, index=False)
    print(f"Normalized data saved to {normalized_path}")
    
    return df_normalized

def filter_low_variance_genes(df, threshold=0.25, output_dir=None):
    """Filter genes with variance below threshold and plot distribution."""
    row_variances = df.iloc[:, 1:].var(axis=1)
    
    # Plot variance distribution
    plt.figure(figsize=(8, 6))
    plt.hist(row_variances, bins=50, color='skyblue', edgecolor='black')
    plt.title('Distribution of Gene Expression Variances')
    plt.xlabel('Variance')
    plt.ylabel('Frequency (log scale)')
    plt.yscale('log')
    plt.grid(True)
    plot_path = os.path.join(output_dir, "variance_distribution.png")
    plt.savefig(plot_path)
    print(f"Variance distribution plot saved to {plot_path}")
    
    # Filter
    filtered_df = df[row_variances >= threshold]
    
    filtered_path = os.path.join(output_dir, "filtered_low_variance.csv")
    filtered_df.to_csv(filtered_path, index=False)
    print(f"Filtered data (variance >= {threshold}) saved to {filtered_path}")
    
    return filtered_df

def transpose_data(df, output_dir):
    """Transpose data (samples as rows, genes as columns)."""
    df_transposed = df.set_index('GENE_ID').T.reset_index()
    df_transposed.rename(columns={'index': 'public_id'}, inplace=True)
    df_transposed['public_id'] = df_transposed['public_id'].str.replace('_1_BM', '')
    
    transposed_path = os.path.join(output_dir, "transposed.csv")
    df_transposed.to_csv(transposed_path, index=False)
    print(f"Transposed data saved to {transposed_path}")
    
    return df_transposed

def fetch_immunoglobulin_genes():
    """Fetch Ensembl IDs of immunoglobulin genes from HGNC API."""
    url = "https://rest.genenames.org/search/symbol/IG*+AND+status:Approved"
    headers = {"Accept": "application/json"}
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        ensembl_ids = []
        for gene in data['response']['docs']:
            if 'ensembl_gene_id' in gene:
                ensembl_ids.append(gene['ensembl_gene_id'])
        print(f"Fetched {len(ensembl_ids)} immunoglobulin genes.")
        return ensembl_ids
    else:
        raise ValueError(f"Failed to fetch HGNC data: {response.status_code}")

def remove_immunoglobulin_genes(df, output_dir):
    """Remove immunoglobulin genes from the dataset."""
    ig_genes = fetch_immunoglobulin_genes()
    columns_to_drop = [col for col in df.columns if col in ig_genes]
    df_cleaned = df.drop(columns=columns_to_drop)
    
    cleaned_path = os.path.join(output_dir, "cleaned_no_ig_genes.csv")
    df_cleaned.to_csv(cleaned_path, index=False)
    print(f"Cleaned data (IG genes removed) saved to {cleaned_path}")
    
    return df_cleaned

def merge_with_survival(df, survival_file, output_dir):
    """Merge expression data with survival data."""
    survival_df = pd.read_csv(survival_file)
    merged_df = pd.merge(df, survival_df, on='public_id')
    
    merged_path = os.path.join(output_dir, "merged_with_survival.csv")
    merged_df.to_csv(merged_path, index=False)
    print(f"Merged data saved to {merged_path}")
    
    return merged_df

def label_risk_classes(df, threshold=24):
    """Label patients as ultra-high-risk (1) or low-risk (0) based on survival time."""
    df['target'] = df['ttcos'].apply(lambda x: 1 if x < threshold else 0)  # 1: ultra-high-risk, 0: low-risk
    return df

def perform_feature_selection(df, k=30):
    """Select top k features using ANOVA F-value."""
    X = df.drop(['public_id', 'ttcos', 'target'], axis=1)
    y = df['target']
    
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    
    selected_genes = X.columns[selector.get_support()].tolist()
    print(f"Selected {k} genes: {selected_genes}")
    
    return X_selected, y, selected_genes

def train_and_evaluate_svm(X, y):
    """Train SVM with grid search and evaluate using stratified CV."""
    svm_model = SVC(probability=True, random_state=42)
    
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': [1, 0.1, 0.01],
        'kernel': ['rbf', 'linear']
    }
    
    skf = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=skf, scoring='accuracy')
    grid_search.fit(X, y)
    
    print(f"Best parameters: {grid_search.best_params_}")
    best_model = grid_search.best_estimator_
    
    # Cross-validated metrics
    accuracies = cross_val_score(best_model, X, y, cv=skf, scoring='accuracy')
    roc_aucs = cross_val_score(best_model, X, y, cv=skf, scoring='roc_auc')
    
    print(f"Cross-validated accuracy: {accuracies.mean()}")
    print(f"Cross-validated ROC-AUC: {roc_aucs.mean()}")
    
    # Classification report
    y_pred = cross_val_predict(best_model, X, y, cv=skf)
    print(classification_report(y, y_pred, target_names=['low-risk', 'ultra-high-risk']))

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 1: Load and preprocess
    df = load_and_preprocess_expression_data(args.expression_file, args.output_dir)
    
    # Step 2: Filter low variance
    df_filtered = filter_low_variance_genes(df, output_dir=args.output_dir)
    
    # Step 3: Transpose
    df_transposed = transpose_data(df_filtered, args.output_dir)
    
    # Step 4: Remove IG genes
    df_cleaned = remove_immunoglobulin_genes(df_transposed, args.output_dir)
    
    # Step 5: Merge with survival
    df_merged = merge_with_survival(df_cleaned, args.survival_file, args.output_dir)
    
    # Step 6: Label risk classes
    df_labeled = label_risk_classes(df_merged)
    
    # Step 7: Feature selection
    X_selected, y, selected_genes = perform_feature_selection(df_labeled)
    
    # Step 8: Train and evaluate
    train_and_evaluate_svm(X_selected, y)

if __name__ == "__main__":
    main()
