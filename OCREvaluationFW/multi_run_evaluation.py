"""
Multi-Run Evaluation Consistency Strategy Implementation
========================================================

This module implements the Multi-Run Evaluation Consistency Strategy as outlined in section 8.
It provides functionality to:

1. Execute multiple independent runs (default: 3) for each PDF + model combination
2. Log exact model configuration for each run
3. Compute aggregated metrics (mean, stddev, variance trend, CCI)
4. Generate comprehensive reports with per-run and aggregated results
5. Ensure reproducibility and transparent evaluation

Key Components:
- MultiRunRunner: Orchestrates multiple evaluation runs
- MultiRunAggregator: Computes aggregated metrics and CCI
- MultiRunReporter: Generates reports and visualizations
"""

import json
import os
import time
import statistics
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, asdict
import tempfile
import shutil

# Import existing evaluation functions
from compare import evaluate_ocr_performance
from ocr_eval_utils import normalize_text


@dataclass
class RunConfig:
    """Configuration for a single evaluation run"""
    model_name: str
    model_version: str = "v1.0"
    model_family: str = "LLM"  # LLM or OCR
    timestamp: str = ""
    run_id: int = 1
    inference_parameters: Dict[str, Any] = None
    status: str = "pending"  # pending, success, failed
    error_message: Optional[str] = None
    duration_seconds: float = 0.0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()
        if self.inference_parameters is None:
            self.inference_parameters = {}


@dataclass
class RunMetrics:
    """Metrics computed for a single evaluation run"""
    run_id: int
    wer: float = 0.0
    mer: float = 0.0
    wil: float = 0.0
    cer: float = 0.0
    lev_distance: int = 0
    lev_norm: float = 0.0
    structural_accuracy: Dict[str, Any] = None
    structural_analysis: Dict[str, float] = None
    completeness: float = 0.0
    word_mismatches: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.structural_accuracy is None:
            self.structural_accuracy = {}
        if self.structural_analysis is None:
            self.structural_analysis = {}
        if self.word_mismatches is None:
            self.word_mismatches = []


@dataclass
class AggregatedMetric:
    """Aggregated statistics for a single metric across multiple runs"""
    mean: float
    std_dev: float
    cci: float  # Consistency Confidence Index
    variance_trend: str  # "stable", "increasing", "decreasing"
    values: List[float] = None
    
    def __post_init__(self):
        if self.values is None:
            self.values = []


@dataclass
class MultiRunSummary:
    """Summary of multi-run evaluation for a single PDF"""
    pdf_name: str
    model_name: str
    total_runs: int
    successful_runs: int
    failed_runs: int
    aggregated_metrics: Dict[str, AggregatedMetric]
    overall_cci: float
    stability_interpretation: str
    run_details: List[Dict[str, Any]]
    timestamp: str = ""
    # Composite aggregate scores (computed from aggregated metric means)
    text_score: Optional[float] = None
    structural_score: Optional[float] = None
    overall_score: Optional[float] = None
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class MultiRunAggregator:
    """Computes aggregated metrics and CCI from multiple runs"""
    
    @staticmethod
    def compute_aggregated_metric(values: List[float]) -> AggregatedMetric:
        """Compute aggregated statistics for a single metric"""
        # Filter out None values (NA metrics) and coerce to float safely
        valid_values: List[float] = []
        for v in values:
            if v is None:
                continue
            try:
                valid_values.append(float(v))
            except Exception:
                # Skip non-numeric values (e.g., 'NA')
                continue
        
        if not valid_values:
            # All values are None/NA - return NA aggregated metric
            return AggregatedMetric(None, None, None, "not_applicable", values)
        
        if len(valid_values) < len([v for v in values if v is not None]):
            # Some values are None/NA - note this in variance trend
            variance_trend_suffix = "_with_na"
        else:
            variance_trend_suffix = ""
        
        mean_val = float(statistics.mean(valid_values))
        std_dev = float(statistics.stdev(valid_values)) if len(valid_values) > 1 else 0.0
        
        # Compute CCI = 1 - (StdDev / Mean), guard against division by zero
        if float(mean_val) > 0:
            cci = max(0.0, 1.0 - (std_dev / mean_val))
        else:
            cci = 0.0
        
        # Determine variance trend (simple approach)
        if len(valid_values) < 3:
            variance_trend = "insufficient_data" + variance_trend_suffix
        else:
            # Check if values are generally increasing, decreasing, or stable
            first_half = valid_values[:len(valid_values)//2]
            second_half = valid_values[len(valid_values)//2:]
            first_mean = statistics.mean(first_half)
            second_mean = statistics.mean(second_half)
            
            diff_ratio = abs(second_mean - first_mean) / max(first_mean, 0.001)
            if diff_ratio < 0.05:  # Less than 5% change
                variance_trend = "stable" + variance_trend_suffix
            elif second_mean > first_mean:
                variance_trend = "increasing" + variance_trend_suffix
            else:
                variance_trend = "decreasing" + variance_trend_suffix
        
        return AggregatedMetric(mean_val, std_dev, cci, variance_trend, values.copy())
    
    @staticmethod
    def compute_overall_cci(aggregated_metrics: Dict[str, AggregatedMetric]) -> float:
        """Compute overall CCI across all metrics"""
        cci_values = [metric.cci for metric in aggregated_metrics.values() if metric.cci is not None and metric.cci > 0]
        return statistics.mean(cci_values) if cci_values else 0.0

    @staticmethod
    def compute_composite_scores(aggregated_metrics: Dict[str, AggregatedMetric]) -> Dict[str, Optional[float]]:
        """
        Compute composite aggregate scores per PDF using aggregated metric means.

        (a) Text Accuracy Score
            Text Score = 1 - average(WER, CER, MER, WIL)

        (b) Structural Score
            Structural Score = average(
                Heading Alignment,
                List Accuracy,
                Table Preservation (ignore NA),
                Link Correctness,
                Section Ordering
            )

        (c) Overall Extraction Score
            Overall Score = 0.6 * Text Score + 0.4 * Structural Score
        """
        # Helper to safely collect means
        def _collect_means(names: List[str]) -> List[float]:
            vals: List[float] = []
            for n in names:
                m = aggregated_metrics.get(n)
                if m is not None and m.mean is not None:
                    try:
                        vals.append(float(m.mean))
                    except Exception:
                        continue
            return vals

        # Text metrics are error rates/distances; lower is better
        text_components = _collect_means(['wer', 'cer', 'mer', 'wil'])
        text_score: Optional[float]
        if text_components:
            avg_err = statistics.mean(text_components)
            text_score = max(0.0, min(1.0, 1.0 - float(avg_err)))
        else:
            text_score = None

        # Structural metrics are accuracy/preservation scores in [0,1]; higher is better
        structural_components = _collect_means([
            'heading_alignment', 'list_accuracy', 'table_preservation', 'link_correctness', 'section_ordering'
        ])
        structural_score: Optional[float]
        if structural_components:
            structural_score = max(0.0, min(1.0, float(statistics.mean(structural_components))))
        else:
            structural_score = None

        # Overall score weighted combination
        overall_score: Optional[float]
        if text_score is not None and structural_score is not None:
            overall_score = max(0.0, min(1.0, 0.6 * text_score + 0.4 * structural_score))
        else:
            overall_score = None

        return {
            'text_score': text_score,
            'structural_score': structural_score,
            'overall_score': overall_score,
        }
    
    @staticmethod
    def interpret_stability(overall_cci: float) -> str:
        """Interpret stability based on CCI guidelines"""
        if overall_cci > 0.90:
            return "Highly stable"
        elif overall_cci >= 0.75:
            return "Moderately stable"
        else:
            return "Unstable results"
    
    @classmethod
    def aggregate_runs(cls, run_metrics_list: List[RunMetrics]) -> Dict[str, AggregatedMetric]:
        """Aggregate metrics from multiple runs"""
        if not run_metrics_list:
            return {}
        
        # Extract values for each metric
        metric_values = {
            'wer': [m.wer for m in run_metrics_list],
            'mer': [m.mer for m in run_metrics_list],
            'wil': [m.wil for m in run_metrics_list],
            'cer': [m.cer for m in run_metrics_list],
            'lev_norm': [m.lev_norm for m in run_metrics_list],
            'completeness': [m.completeness for m in run_metrics_list]
        }
        
        # Extract structural analysis metrics if available
        structural_metrics_available = any(m.structural_analysis for m in run_metrics_list)
        if structural_metrics_available:
            structural_metric_names = ['heading_alignment', 'list_accuracy', 'table_preservation', 'link_correctness', 'section_ordering']
            for metric_name in structural_metric_names:
                metric_values[metric_name] = [
                    m.structural_analysis.get(metric_name) if m.structural_analysis else None
                    for m in run_metrics_list
                ]
        
        # Compute aggregated metrics
        aggregated = {}
        for metric_name, values in metric_values.items():
            aggregated[metric_name] = cls.compute_aggregated_metric(values)
        
        return aggregated


class MultiRunRunner:
    """Orchestrates multiple evaluation runs for consistency analysis"""
    
    def __init__(self, num_runs: int = 3, output_base_dir: str = "evaluation/results"):
        self.num_runs = num_runs
        self.output_base_dir = output_base_dir
        self.aggregator = MultiRunAggregator()
    
    def _create_run_directory(self, model_folder: str, pdf_basename: str, run_id: int) -> str:
        """Create directory structure for a single run"""
        run_dir = os.path.join(self.output_base_dir, model_folder, pdf_basename, f"run_{run_id}")
        os.makedirs(run_dir, exist_ok=True)
        return run_dir
    
    def _save_run_config(self, run_dir: str, config: RunConfig) -> str:
        """Save run configuration to JSON file"""
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(config), f, indent=2, ensure_ascii=False)
        return config_path
    
    def _save_run_metrics(self, run_dir: str, metrics: RunMetrics) -> str:
        """Save run metrics to JSON file"""
        metrics_path = os.path.join(run_dir, "metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)
        return metrics_path
    
    def _save_raw_output(self, run_dir: str, raw_output: str) -> str:
        """Save raw OCR output to markdown file"""
        output_path = os.path.join(run_dir, "raw_output.md")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(raw_output)
        return output_path
    
    def _compute_metrics_from_texts(self, gt_text: str, ocr_text: str, run_id: int) -> RunMetrics:
        """Compute evaluation metrics from ground truth and OCR text"""
        # Use unified metrics computation function
        from ocr_eval_utils import compute_all_metrics
        metrics_result = compute_all_metrics(gt_text, ocr_text)
        
        # Extract structural metrics
        structural_accuracy = metrics_result['structural_accuracy']
        structural_analysis = metrics_result['structural_analysis']
        
        # Convert top mismatches to expected format
        top_mismatches = [
            {"gt": gt, "ocr": ocr, "count": count}
            for (gt, ocr), count in metrics_result['top_mismatches_raw']
        ]
        
        return RunMetrics(
            run_id=run_id,
            wer=metrics_result['wer'],
            mer=metrics_result['mer'],
            wil=metrics_result['wil'],
            cer=metrics_result['cer'],
            lev_distance=metrics_result['lev_distance'],
            lev_norm=metrics_result['lev_norm'],
            structural_accuracy=structural_accuracy,
            structural_analysis=structural_analysis,
            completeness=metrics_result['completeness'],
            word_mismatches=top_mismatches
        )
    
    def execute_single_run(
        self,
        run_id: int,
        pdf_basename: str,
        model_folder: str,
        gt_text: str,
        extraction_func,
        extraction_params: Dict[str, Any],
        progress_callback=None
    ) -> Tuple[RunConfig, RunMetrics, str]:
        """Execute a single evaluation run"""
        
        # Create run directory
        run_dir = self._create_run_directory(model_folder, pdf_basename, run_id)
        
        # Create run configuration (filter out non-serializable parameters)
        serializable_params = {}
        for key, value in extraction_params.items():
            if key not in ['file_bytes', 'model_name', 'model_version', 'model_family']:
                # Only include serializable values
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    serializable_params[key] = value
                else:
                    serializable_params[key] = str(value)
        
        config = RunConfig(
            model_name=extraction_params.get('model_name', model_folder),
            model_version=extraction_params.get('model_version', 'v1.0'),
            model_family=extraction_params.get('model_family', 'LLM' if 'Marker' in model_folder else 'OCR'),
            run_id=run_id,
            inference_parameters=serializable_params
        )
        
        start_time = time.time()
        
        try:
            if progress_callback:
                progress_callback(f"Run {run_id}: Starting extraction...")
            
            # Execute extraction (this should be a fresh call each time)
            raw_output = extraction_func(**extraction_params)
            
            if not raw_output or not raw_output.strip():
                raise ValueError("Empty or null output from extraction function")
            
            # Save raw output
            self._save_raw_output(run_dir, raw_output)
            
            if progress_callback:
                progress_callback(f"Run {run_id}: Computing metrics...")
            
            # Compute metrics
            metrics = self._compute_metrics_from_texts(gt_text, raw_output, run_id)
            
            # Update config with success status
            config.status = "success"
            config.duration_seconds = time.time() - start_time
            
            if progress_callback:
                progress_callback(f"Run {run_id}: Completed successfully")
            
        except Exception as e:
            # Handle failure
            config.status = "failed"
            config.error_message = str(e)
            config.duration_seconds = time.time() - start_time
            
            # Create empty metrics for failed run
            metrics = RunMetrics(run_id=run_id)
            
            # Save empty raw output
            self._save_raw_output(run_dir, f"# Run {run_id} Failed\n\nError: {str(e)}")
            
            if progress_callback:
                progress_callback(f"Run {run_id}: Failed - {str(e)}")
        
        # Save config and metrics
        self._save_run_config(run_dir, config)
        self._save_run_metrics(run_dir, metrics)
        
        return config, metrics, run_dir
    
    def execute_multi_run_evaluation(
        self,
        pdf_basename: str,
        model_folder: str,
        gt_text: str,
        extraction_func,
        extraction_params: Dict[str, Any],
        progress_callback=None
    ) -> MultiRunSummary:
        """Execute complete multi-run evaluation for a single PDF"""
        
        if progress_callback:
            progress_callback(f"Starting multi-run evaluation: {self.num_runs} runs")
        
        successful_configs = []
        successful_metrics = []
        all_run_details = []
        
        # Execute all runs
        for run_id in range(1, self.num_runs + 1):
            config, metrics, run_dir = self.execute_single_run(
                run_id, pdf_basename, model_folder, gt_text,
                extraction_func, extraction_params, progress_callback
            )
            
            # Track run details
            run_detail = {
                "run_id": run_id,
                "status": config.status,
                "error": config.error_message,
                "metrics_file": f"run_{run_id}/metrics.json",
                "duration_seconds": config.duration_seconds
            }
            all_run_details.append(run_detail)
            
            # Collect successful runs
            if config.status == "success":
                successful_configs.append(config)
                successful_metrics.append(metrics)
        
        # Check if we have enough successful runs (at least 2)
        if len(successful_metrics) < 2:
            if progress_callback:
                progress_callback(f"Insufficient successful runs ({len(successful_metrics)}/3). Cannot compute reliable aggregates.")
            
            # Create summary with minimal data
            summary = MultiRunSummary(
                pdf_name=pdf_basename,
                model_name=model_folder,
                total_runs=self.num_runs,
                successful_runs=len(successful_metrics),
                failed_runs=self.num_runs - len(successful_metrics),
                aggregated_metrics={},
                overall_cci=0.0,
                stability_interpretation="Insufficient data",
                run_details=all_run_details
            )
        else:
            if progress_callback:
                progress_callback("Computing aggregated metrics and CCI...")
            
            # Compute aggregated metrics
            aggregated_metrics = self.aggregator.aggregate_runs(successful_metrics)
            overall_cci = self.aggregator.compute_overall_cci(aggregated_metrics)
            stability_interpretation = self.aggregator.interpret_stability(overall_cci)
            # Compute composite aggregate scores
            composite = self.aggregator.compute_composite_scores(aggregated_metrics)
            
            # Create summary
            summary = MultiRunSummary(
                pdf_name=pdf_basename,
                model_name=model_folder,
                total_runs=self.num_runs,
                successful_runs=len(successful_metrics),
                failed_runs=self.num_runs - len(successful_metrics),
                aggregated_metrics=aggregated_metrics,
                overall_cci=overall_cci,
                stability_interpretation=stability_interpretation,
                run_details=all_run_details,
                text_score=composite.get('text_score'),
                structural_score=composite.get('structural_score'),
                overall_score=composite.get('overall_score')
            )
        
        # Save summary
        summary_dir = os.path.join(self.output_base_dir, model_folder, pdf_basename)
        os.makedirs(summary_dir, exist_ok=True)
        summary_path = os.path.join(summary_dir, "summary_multirun.json")
        
        # Convert summary to dict for JSON serialization
        summary_dict = asdict(summary)
        # Convert AggregatedMetric objects to dicts
        summary_dict['aggregated_metrics'] = {
            k: asdict(v) for k, v in summary.aggregated_metrics.items()
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_dict, f, indent=2, ensure_ascii=False)
        
        if progress_callback:
            progress_callback(f"Multi-run evaluation completed. Summary saved to: {summary_path}")
        
        return summary


# Utility functions for integration with existing codebase

def create_extraction_wrapper(submit_job_func, extract_markdown_func):
    """Create a wrapper function for extraction that can be called multiple times"""
    def extraction_wrapper(**params):
        """Wrapper that calls the API and extracts markdown"""
        # Filter out parameters that are not expected by the submit function
        api_params = {}
        for key, value in params.items():
            if key not in ['model_name', 'model_version', 'model_family']:
                api_params[key] = value
        
        # Call the API submission function
        result_payload = submit_job_func(**api_params)
        
        # Extract markdown from the result
        markdown = extract_markdown_func(result_payload)
        
        if not markdown:
            raise ValueError("No markdown content extracted from API response")
        
        return markdown
    
    return extraction_wrapper


def get_pdf_basename(filename: str) -> str:
    """Extract base name from PDF filename for directory naming"""
    return os.path.splitext(os.path.basename(filename))[0]


# Configuration integration

def get_multi_run_config() -> Dict[str, Any]:
    """Get multi-run configuration from config file or defaults"""
    try:
        from config import get_settings
        settings = get_settings()
        # Check if multi-run settings exist in the config
        # For now, return defaults - will be extended when config.py is updated
        return {
            'enabled': True,  # Will be configurable
            'num_runs': 3,
            'require_all_success': False  # Default behavior: compute aggregates if >=2 succeed
        }
    except Exception:
        # Fallback to defaults
        return {
            'enabled': True,
            'num_runs': 3,
            'require_all_success': False
        }