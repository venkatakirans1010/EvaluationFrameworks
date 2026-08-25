import json
import os
from multi_run_evaluation import MultiRunRunner

def regenerate_metrics_for_run(model_folder, pdf_basename, run_id):
    """Regenerate metrics for a specific run"""
    
    # Paths
    gt_path = f"ground_truth/{pdf_basename}_ground_truth.md"
    run_dir = f"evaluation/results/{model_folder}/{pdf_basename}/run_{run_id}"
    raw_output_path = f"{run_dir}/raw_output.md"
    metrics_path = f"{run_dir}/metrics.json"
    
    # Check if files exist
    if not os.path.exists(gt_path):
        print(f"Ground truth not found: {gt_path}")
        return False
    
    if not os.path.exists(raw_output_path):
        print(f"Raw output not found: {raw_output_path}")
        return False
    
    # Read texts
    with open(gt_path, 'r', encoding='utf-8') as f:
        gt_text = f.read()
    
    with open(raw_output_path, 'r', encoding='utf-8') as f:
        ocr_text = f.read()
    
    # Compute new metrics
    runner = MultiRunRunner()
    metrics = runner._compute_metrics_from_texts(gt_text, ocr_text, run_id)
    
    # Save updated metrics
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics.__dict__, f, indent=2, ensure_ascii=False)
    
    print(f"Updated metrics for {model_folder}/{pdf_basename}/run_{run_id}")
    print(f"  Structural analysis: {metrics.structural_analysis}")
    return True

# Regenerate for Table_Based_PDF
for run_id in [1, 2, 3]:
    regenerate_metrics_for_run("Marker", "Table_Based_PDF", run_id)