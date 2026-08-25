"""
Multi-Run Report Generation Module
==================================

This module generates comprehensive reports for multi-run evaluation results,
including per-run metrics, aggregated statistics, and visual charts.

Features:
- Per-run metric blocks with detailed breakdowns
- Aggregated tables with mean, stddev, and CCI
- Visual charts showing variance across runs
- Markdown and PNG export formats
- Consistency interpretation and recommendations
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import seaborn as sns
from dataclasses import asdict

from multi_run_evaluation import MultiRunSummary, AggregatedMetric, RunMetrics


class MultiRunReporter:
    """Generates comprehensive reports for multi-run evaluation results"""
    
    def __init__(self, output_base_dir: str = "evaluation/results"):
        self.output_base_dir = output_base_dir
        # Set up matplotlib style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def _fmt(self, v, digits=4):
        """Format a numeric metric or return 'NA' for None"""
        if v is None:
            return "NA"
        try:
            return f"{v:.{digits}f}"
        except Exception:
            return str(v)
    
    def _as_float_or_nan(self, v):
        """Convert a metric value to a float, or numpy.nan if None"""
        import numpy as _np
        return _np.nan if v is None else float(v)
    
    def generate_per_run_metrics_table(self, run_metrics_list: List[RunMetrics]) -> str:
        """Generate markdown table showing metrics for each run"""
        if not run_metrics_list:
            return "_No successful runs to display._\n"
        
        lines = []
        lines.append("| Run | WER | MER | WIL | CER | LEV-DIST (chars) | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |")
        lines.append("|-----|-----|-----|-----|-----|-------------------|--------------|-------------------|---------------|--------------------|------------------|------------------|")
        
        for metrics in run_metrics_list:
            structural_metrics = metrics.structural_analysis if metrics.structural_analysis else {}
            lines.append(
                f"| **{metrics.run_id}** | "
                f"{metrics.wer:.4f} | "
                f"{metrics.mer:.4f} | "
                f"{metrics.wil:.4f} | "
                f"{metrics.cer:.4f} | "
                f"{int(metrics.lev_distance)} | "
                f"{metrics.completeness:.4f} | "
                f"{self._fmt(structural_metrics.get('heading_alignment'))} | "
                f"{self._fmt(structural_metrics.get('list_accuracy'))} | "
                f"{self._fmt(structural_metrics.get('table_preservation'))} | "
                f"{self._fmt(structural_metrics.get('link_correctness'))} | "
                f"{self._fmt(structural_metrics.get('section_ordering'))} |"
            )
        
        return "\n".join(lines) + "\n"
    
    def generate_aggregated_metrics_table(self, aggregated_metrics: Dict[str, AggregatedMetric]) -> str:
        """Generate markdown table showing aggregated metrics with CCI"""
        if not aggregated_metrics:
            return "_No aggregated metrics available._\n"
        
        lines = []
        lines.append("| Metric | Mean | Std Dev | CCI | Variance Trend | Interpretation |")
        lines.append("|--------|------|---------|-----|----------------|----------------|")
        
        for metric_name, agg_metric in aggregated_metrics.items():
            # Interpret CCI
            cci_val = agg_metric.cci
            if cci_val is None:
                cci_interp = "NA"
            elif cci_val > 0.90:
                cci_interp = "🟢 Highly Stable"
            elif cci_val >= 0.75:
                cci_interp = "🟡 Moderately Stable"
            else:
                cci_interp = "🔴 Unstable"
            
            # Format variance trend
            trend_emoji = {
                "stable": "➡️",
                "increasing": "📈",
                "decreasing": "📉",
                "insufficient_data": "❓",
                "not_applicable": "NA"
            }
            trend_display = f"{trend_emoji.get(agg_metric.variance_trend, '❓')} {agg_metric.variance_trend.replace('_', ' ').title() if agg_metric.variance_trend else 'NA'}"
            
            lines.append(
                f"| **{metric_name.upper()}** | "
                f"{self._fmt(agg_metric.mean)} | "
                f"{self._fmt(agg_metric.std_dev)} | "
                f"{self._fmt(agg_metric.cci)} | "
                f"{trend_display} | "
                f"{cci_interp} |"
            )
        
        return "\n".join(lines) + "\n"
    
    def generate_run_details_table(self, run_details: List[Dict[str, Any]]) -> str:
        """Generate markdown table showing run execution details"""
        if not run_details:
            return "_No run details available._\n"
        
        lines = []
        lines.append("| Run | Status | Duration (s) | Error |")
        lines.append("|-----|--------|--------------|-------|")
        
        for detail in run_details:
            status_emoji = "✅" if detail["status"] == "success" else "❌"
            error_display = detail.get("error", "None") or "None"
            if len(error_display) > 50:
                error_display = error_display[:47] + "..."
            
            lines.append(
                f"| **{detail['run_id']}** | "
                f"{status_emoji} {detail['status'].title()} | "
                f"{detail.get('duration_seconds', 0):.2f} | "
                f"{error_display} |"
            )
        
        return "\n".join(lines) + "\n"
    
    def create_metrics_variance_chart(
        self,
        aggregated_metrics: Dict[str, AggregatedMetric],
        chart_path: str,
        pdf_name: str,
        model_name: str
    ) -> str:
        """Create a chart showing metric variance across runs"""
        if not aggregated_metrics:
            return ""
        
        # Prepare data - filter out metrics with NA mean
        metrics = []
        means = []
        std_devs = []
        ccis = []
        for metric_name, agg in aggregated_metrics.items():
            if agg.mean is None:
                continue
            metrics.append(metric_name)
            means.append(agg.mean)
            std_devs.append(agg.std_dev if agg.std_dev is not None else 0.0)
            ccis.append(agg.cci if agg.cci is not None else 0.0)
        
        if not metrics:
            return ""
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Chart 1: Mean values with error bars
        colors = plt.cm.Set3(np.linspace(0, 1, len(metrics)))
        bars1 = ax1.bar(metrics, means, yerr=std_devs, capsize=5, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_title(f'Metric Means with Standard Deviation\n{pdf_name} - {model_name}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Metric Value', fontsize=12)
        ax1.set_xlabel('Metrics', fontsize=12)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars1, means, std_devs):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (std if std is not None else 0.0) + 0.01,
                    f'{mean:.4f}±{(std if std is not None else 0.0):.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Chart 2: CCI values
        colors_cci = ['#2ecc71' if cci > 0.90 else '#f39c12' if cci >= 0.75 else '#e74c3c' for cci in ccis]
        bars2 = ax2.bar(metrics, ccis, color=colors_cci, alpha=0.8, edgecolor='black')
        ax2.set_title('Consistency Confidence Index (CCI)\nHigher = More Stable', fontsize=14, fontweight='bold')
        ax2.set_ylabel('CCI Value', fontsize=12)
        ax2.set_xlabel('Metrics', fontsize=12)
        ax2.set_ylim(0, 1.0)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add CCI threshold lines
        ax2.axhline(y=0.90, color='green', linestyle='--', alpha=0.7, label='Highly Stable (>0.90)')
        ax2.axhline(y=0.75, color='orange', linestyle='--', alpha=0.7, label='Moderately Stable (≥0.75)')
        ax2.legend(loc='upper right', fontsize=10)
        
        # Add value labels on CCI bars
        for bar, cci in zip(bars2, ccis):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{cci:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return chart_path
    
    def create_run_comparison_chart(
        self,
        run_metrics_list: List[RunMetrics],
        chart_path: str,
        pdf_name: str,
        model_name: str
    ) -> str:
        """Create a chart comparing metrics across individual runs"""
        if not run_metrics_list:
            return ""
        
        # Prepare data
        run_ids = [m.run_id for m in run_metrics_list]
        metrics_data = {
            'WER': [m.wer for m in run_metrics_list],
            'MER': [m.mer for m in run_metrics_list],
            'WIL': [m.wil for m in run_metrics_list],
            'CER': [m.cer for m in run_metrics_list],
            'LEV-DIST (chars)': [m.lev_distance for m in run_metrics_list],
            'Completeness': [m.completeness for m in run_metrics_list]
        }
        
        # Add structural analysis metrics if available
        structural_metrics_available = any(m.structural_analysis for m in run_metrics_list)
        if structural_metrics_available:
            metrics_data.update({
                'Heading Alignment': [self._as_float_or_nan(m.structural_analysis.get('heading_alignment')) if m.structural_analysis else self._as_float_or_nan(None) for m in run_metrics_list],
                'List Accuracy': [self._as_float_or_nan(m.structural_analysis.get('list_accuracy')) if m.structural_analysis else self._as_float_or_nan(None) for m in run_metrics_list],
                'Table Preservation': [self._as_float_or_nan(m.structural_analysis.get('table_preservation')) if m.structural_analysis else self._as_float_or_nan(None) for m in run_metrics_list],
                'Link Correctness': [self._as_float_or_nan(m.structural_analysis.get('link_correctness')) if m.structural_analysis else self._as_float_or_nan(None) for m in run_metrics_list],
                'Section Ordering': [self._as_float_or_nan(m.structural_analysis.get('section_ordering')) if m.structural_analysis else self._as_float_or_nan(None) for m in run_metrics_list]
            })
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot lines for each metric
        colors = plt.cm.tab10(np.linspace(0, 1, len(metrics_data)))
        for i, (metric_name, values) in enumerate(metrics_data.items()):
            # Plot; matplotlib will handle numpy.nan by breaking the line
            ax.plot(run_ids, values, marker='o', linewidth=2, markersize=8,
                   color=colors[i], label=metric_name, alpha=0.8)
            
            # Add value labels only for finite values
            for run_id, value in zip(run_ids, values):
                try:
                    if np.isfinite(value):
                        ax.annotate(f'{value:.3f}', (run_id, value),
                                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
                except Exception:
                    # value might be a python None or non-numeric; skip annotation
                    pass
        
        ax.set_title(f'Metric Values Across Runs\n{pdf_name} - {model_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Run Number', fontsize=12)
        ax.set_ylabel('Metric Value', fontsize=12)
        ax.set_xticks(run_ids)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Add note
        note_text = "Lower values indicate better OCR performance (except Completeness and Structural Analysis metrics)"
        plt.figtext(0.5, 0.02, note_text,
                   ha='center', fontsize=10, style='italic', color='gray')
        
        plt.tight_layout()
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return chart_path
    
    def generate_comprehensive_report(
        self, 
        summary: MultiRunSummary,
        run_metrics_list: List[RunMetrics],
        model_folder: str
    ) -> Tuple[str, str, List[str]]:
        """Generate comprehensive multi-run evaluation report"""
        
        # Create output directory
        report_dir = os.path.join(self.output_base_dir, model_folder, summary.pdf_name)
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate report content
        report_lines = []
        
        # Header
        report_lines.append(f"# 📊 Multi-Run Evaluation Report")
        report_lines.append(f"")
        report_lines.append(f"**📄 Document:** `{summary.pdf_name}`")
        report_lines.append(f"**🤖 Model:** `{summary.model_name}`")
        report_lines.append(f"**🕒 Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**📈 Total Runs:** {summary.total_runs}")
        report_lines.append(f"**✅ Successful Runs:** {summary.successful_runs}")
        report_lines.append(f"**❌ Failed Runs:** {summary.failed_runs}")
        report_lines.append(f"")
        
        # Executive Summary
        report_lines.append(f"## 🎯 Executive Summary")
        report_lines.append(f"")
        report_lines.append(f"**Overall Consistency Confidence Index (CCI):** `{summary.overall_cci:.4f}`")
        report_lines.append(f"**Stability Interpretation:** {summary.stability_interpretation}")
        report_lines.append(f"")
        
        # Interpretation guidelines
        if summary.overall_cci > 0.90:
            report_lines.append(f"🟢 **Excellent Consistency:** The model produces highly stable and reproducible results across multiple runs.")
        elif summary.overall_cci >= 0.75:
            report_lines.append(f"🟡 **Good Consistency:** The model shows moderate stability with some variance between runs.")
        else:
            report_lines.append(f"🔴 **Poor Consistency:** The model shows significant variance between runs, indicating potential reliability issues.")
        
        report_lines.append(f"")

        # Aggregate Scores (composite)
        try:
            # Backfill composite if missing by computing from aggregated metrics
            text_score = summary.text_score
            structural_score = summary.structural_score
            overall_score = summary.overall_score

            if (text_score is None or structural_score is None or overall_score is None) and summary.aggregated_metrics:
                # Compute on the fly using the same logic
                def _mean_or_none(metric_name: str) -> Optional[float]:
                    m = summary.aggregated_metrics.get(metric_name)
                    return None if (m is None or m.mean is None) else float(m.mean)

                text_components = [
                    _mean_or_none('wer'), _mean_or_none('cer'), _mean_or_none('mer'), _mean_or_none('wil'), _mean_or_none('lev_norm')
                ]
                text_components = [v for v in text_components if v is not None]
                if text_components:
                    avg_err = float(np.mean(text_components))
                    text_score = max(0.0, min(1.0, 1.0 - avg_err))

                structural_components = [
                    _mean_or_none('heading_alignment'), _mean_or_none('list_accuracy'), _mean_or_none('table_preservation'), _mean_or_none('link_correctness'), _mean_or_none('section_ordering')
                ]
                structural_components = [v for v in structural_components if v is not None]
                if structural_components:
                    structural_score = max(0.0, min(1.0, float(np.mean(structural_components))))

                if text_score is not None and structural_score is not None:
                    overall_score = max(0.0, min(1.0, 0.6 * text_score + 0.4 * structural_score))

            # Display Aggregate Scores section when available
            report_lines.append("## 🧮 Aggregate Scores")
            report_lines.append("")
            lines = []
            lines.append("| Metric | Score |")
            lines.append("|--------|-------|")
            lines.append(f"| **Text Accuracy Score** | {self._fmt(text_score, 4)} |")
            lines.append(f"| **Structural Score** | {self._fmt(structural_score, 4)} |")
            lines.append(f"| **Overall Extraction Score** | {self._fmt(overall_score, 4)} |")
            report_lines.extend(lines)
            report_lines.append("")
            report_lines.append("- Text Accuracy Score = 1 - average(WER, CER, MER, WIL, LEV-NORM)")
            report_lines.append("- Structural Score = average(Heading Alignment, List Accuracy, Table Preservation, Link Correctness, Section Ordering) — Table Preservation ignored if NA")
            report_lines.append("- Overall Extraction Score = 0.6 × Text Score + 0.4 × Structural Score")
            report_lines.append("")
        except Exception:
            # Non-fatal; skip aggregate scores if any error occurs
            pass
        
        # Run Execution Details
        report_lines.append(f"## 🔄 Run Execution Details")
        report_lines.append(f"")
        report_lines.append(self.generate_run_details_table(summary.run_details))
        
        # Per-Run Metrics
        if run_metrics_list:
            report_lines.append(f"## 📋 Per-Run Metrics")
            report_lines.append(f"")
            report_lines.append(self.generate_per_run_metrics_table(run_metrics_list))
        
        # Aggregated Metrics
        if summary.aggregated_metrics:
            report_lines.append(f"## 📊 Aggregated Metrics Analysis")
            report_lines.append(f"")
            report_lines.append(self.generate_aggregated_metrics_table(summary.aggregated_metrics))
            
            # Detailed metric analysis
            report_lines.append(f"### 🔍 Detailed Analysis")
            report_lines.append(f"")
            
            for metric_name, agg_metric in summary.aggregated_metrics.items():
                if agg_metric.mean is None:
                    stability = "not applicable"
                    report_lines.append(f"- **{metric_name.upper()}**: Mean = NA, StdDev = NA, CCI = NA ({stability})")
                else:
                    stability = "highly stable" if agg_metric.cci > 0.90 else "moderately stable" if agg_metric.cci >= 0.75 else "unstable"
                    report_lines.append(f"- **{metric_name.upper()}**: Mean = {self._fmt(agg_metric.mean)}, StdDev = {self._fmt(agg_metric.std_dev)}, CCI = {self._fmt(agg_metric.cci)} ({stability})")
            
            report_lines.append(f"")
        
        # Structural Analysis Section
        structural_metrics_available = any(
            run_metrics.structural_analysis for run_metrics in run_metrics_list
            if run_metrics.structural_analysis
        )
        
        if structural_metrics_available:
            report_lines.append(f"## 🏗️ Structural Analysis")
            report_lines.append(f"")
            report_lines.append(f"This section evaluates how well the OCR preserved the document's structural elements:")
            report_lines.append(f"")
            
            # Create structural analysis table
            lines = []
            lines.append("| Run | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |")
            lines.append("|-----|-------------------|---------------|--------------------|------------------|------------------|")
            
            for metrics in run_metrics_list:
                if metrics.structural_analysis:
                    structural = metrics.structural_analysis
                    lines.append(
                        f"| **{metrics.run_id}** | "
                        f"{self._fmt(structural.get('heading_alignment'))} | "
                        f"{self._fmt(structural.get('list_accuracy'))} | "
                        f"{self._fmt(structural.get('table_preservation'))} | "
                        f"{self._fmt(structural.get('link_correctness'))} | "
                        f"{self._fmt(structural.get('section_ordering'))} |"
                    )
            
            report_lines.extend(lines)
            report_lines.append(f"")
            
            # Add structural analysis aggregated metrics if available in summary
            if summary.aggregated_metrics:
                structural_agg_metrics = {k: v for k, v in summary.aggregated_metrics.items()
                                        if k in ['heading_alignment', 'list_accuracy', 'table_preservation',
                                               'link_correctness', 'section_ordering']}
                
                if structural_agg_metrics:
                    report_lines.append(f"### 📊 Structural Analysis Summary")
                    report_lines.append(f"")
                    
                    for metric_name, agg_metric in structural_agg_metrics.items():
                        if agg_metric.mean is None:
                            stability = "not applicable"
                            metric_display = metric_name.replace('_', ' ').title()
                            report_lines.append(f"- **{metric_display}**: Mean = NA, StdDev = NA, CCI = NA ({stability})")
                        else:
                            stability = "highly stable" if agg_metric.cci > 0.90 else "moderately stable" if agg_metric.cci >= 0.75 else "unstable"
                            metric_display = metric_name.replace('_', ' ').title()
                            report_lines.append(f"- **{metric_display}**: Mean = {self._fmt(agg_metric.mean)}, StdDev = {self._fmt(agg_metric.std_dev)}, CCI = {self._fmt(agg_metric.cci)} ({stability})")
                    
                    report_lines.append(f"")
            
            report_lines.append(f"### 🔍 Structural Analysis Interpretation")
            report_lines.append(f"")
            report_lines.append(f"- **Heading Alignment**: Measures how accurately heading hierarchy and formatting are preserved")
            report_lines.append(f"- **List Accuracy**: Evaluates preservation of ordered and unordered list structures")
            report_lines.append(f"- **Table Preservation**: Assesses how well table structures and data are maintained")
            report_lines.append(f"- **Link Correctness**: Measures accuracy of hyperlink preservation and formatting")
            report_lines.append(f"- **Section Ordering**: Evaluates whether document sections maintain their original sequence")
            report_lines.append(f"")
            report_lines.append(f"*Note: All structural analysis metrics range from 0.0 to 1.0, where higher values indicate better preservation.*")
            report_lines.append(f"")
        
        # Generate charts
        chart_paths = []
        
        if summary.aggregated_metrics:
            # Variance chart
            variance_chart_path = os.path.join(report_dir, "metrics_variance_chart.png")
            self.create_metrics_variance_chart(
                summary.aggregated_metrics, variance_chart_path, 
                summary.pdf_name, summary.model_name
            )
            chart_paths.append(variance_chart_path)
            
            report_lines.append(f"## 📈 Metrics Variance Analysis")
            report_lines.append(f"")
            report_lines.append(f"![Metrics Variance Chart](metrics_variance_chart.png)")
            report_lines.append(f"")
        
        if run_metrics_list:
            # Run comparison chart
            comparison_chart_path = os.path.join(report_dir, "run_comparison_chart.png")
            self.create_run_comparison_chart(
                run_metrics_list, comparison_chart_path,
                summary.pdf_name, summary.model_name
            )
            chart_paths.append(comparison_chart_path)
            
            report_lines.append(f"## 🔄 Run-by-Run Comparison")
            report_lines.append(f"")
            report_lines.append(f"![Run Comparison Chart](run_comparison_chart.png)")
            report_lines.append(f"")
        
        # Recommendations
        report_lines.append(f"## 💡 Recommendations")
        report_lines.append(f"")
        
        if summary.overall_cci > 0.90:
            report_lines.append(f"✅ **Model Performance:** Excellent consistency. This model can be trusted for production use.")
            report_lines.append(f"✅ **Deployment:** Safe to deploy with confidence in reproducible results.")
        elif summary.overall_cci >= 0.75:
            report_lines.append(f"⚠️ **Model Performance:** Good consistency with some variance. Consider additional testing.")
            report_lines.append(f"⚠️ **Deployment:** Suitable for production with monitoring of result variance.")
        else:
            report_lines.append(f"❌ **Model Performance:** Poor consistency. Investigate model configuration and parameters.")
            report_lines.append(f"❌ **Deployment:** Not recommended for production without addressing consistency issues.")
        
        report_lines.append(f"")
        
        # Technical Details
        report_lines.append(f"## 🔧 Technical Details")
        report_lines.append(f"")
        report_lines.append(f"**Consistency Confidence Index (CCI) Formula:** `CCI = 1 - (StdDev / Mean)`")
        report_lines.append(f"")
        report_lines.append(f"**Interpretation Guidelines:**")
        report_lines.append(f"- CCI > 0.90: Highly stable")
        report_lines.append(f"- CCI 0.75-0.90: Moderately stable")
        report_lines.append(f"- CCI < 0.75: Unstable results")
        report_lines.append(f"")
        report_lines.append(f"**Metrics Explanation:**")
        report_lines.append(f"- **WER**: Word Error Rate (lower is better)")
        report_lines.append(f"- **MER**: Match Error Rate (lower is better)")
        report_lines.append(f"- **WIL**: Word Information Lost (lower is better)")
        report_lines.append(f"- **CER**: Character Error Rate (lower is better)")
        report_lines.append(f"- **LEV-NORM**: Normalized Levenshtein Distance (lower is better)")
        report_lines.append(f"- **Completeness**: Ratio of OCR length to ground truth length (higher is better)")
        report_lines.append(f"")
        report_lines.append(f"**Structural Analysis Metrics:**")
        report_lines.append(f"- **Heading Alignment**: Accuracy of heading structure preservation (higher is better)")
        report_lines.append(f"- **List Accuracy**: Accuracy of list structure preservation (higher is better)")
        report_lines.append(f"- **Table Preservation**: Accuracy of table structure preservation (higher is better)")
        report_lines.append(f"- **Link Correctness**: Accuracy of link preservation (higher is better)")
        report_lines.append(f"- **Section Ordering**: Accuracy of section order preservation (higher is better)")
        
        # Save markdown report
        report_content = "\n".join(report_lines)
        report_md_path = os.path.join(report_dir, "multirun_report.md")
        with open(report_md_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        # Save text version for compatibility
        report_txt_path = os.path.join(report_dir, "multirun_report.txt")
        with open(report_txt_path, 'w', encoding='utf-8') as f:
            # Convert markdown to plain text (simplified)
            plain_text = report_content.replace('#', '').replace('**', '').replace('`', '')
            f.write(plain_text)
        
        return report_md_path, report_txt_path, chart_paths
    
    def load_and_generate_report(self, model_folder: str, pdf_basename: str) -> Tuple[str, str, List[str]]:
        """Load multi-run summary and generate comprehensive report"""
        
        # Load summary
        summary_path = os.path.join(self.output_base_dir, model_folder, pdf_basename, "summary_multirun.json")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Multi-run summary not found: {summary_path}")
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        
        # Reconstruct summary object (simplified)
        summary = MultiRunSummary(
            pdf_name=summary_data['pdf_name'],
            model_name=summary_data['model_name'],
            total_runs=summary_data['total_runs'],
            successful_runs=summary_data['successful_runs'],
            failed_runs=summary_data['failed_runs'],
            aggregated_metrics={},  # Will be reconstructed
            overall_cci=summary_data['overall_cci'],
            stability_interpretation=summary_data['stability_interpretation'],
            run_details=summary_data['run_details'],
            timestamp=summary_data.get('timestamp', ''),
            text_score=summary_data.get('text_score'),
            structural_score=summary_data.get('structural_score'),
            overall_score=summary_data.get('overall_score')
        )
        
        # Reconstruct aggregated metrics
        for metric_name, metric_data in summary_data['aggregated_metrics'].items():
            # Coerce reconstructed fields to proper types to avoid string/int comparison errors
            def _to_float_or_none(x):
                if x is None:
                    return None
                try:
                    return float(x)
                except Exception:
                    return None
            def _sanitize_values_list(vals):
                out = []
                for v in (vals or []):
                    if v is None:
                        out.append(None)
                        continue
                    try:
                        out.append(float(v))
                    except Exception:
                        out.append(None)
                return out

            summary.aggregated_metrics[metric_name] = AggregatedMetric(
                mean=_to_float_or_none(metric_data.get('mean')),
                std_dev=_to_float_or_none(metric_data.get('std_dev')),
                cci=_to_float_or_none(metric_data.get('cci')),
                variance_trend=str(metric_data.get('variance_trend') or 'not_applicable'),
                values=_sanitize_values_list(metric_data.get('values', []))
            )
        
        # Load individual run metrics
        run_metrics_list = []
        for run_detail in summary.run_details:
            if run_detail['status'] == 'success':
                metrics_path = os.path.join(
                    self.output_base_dir, model_folder, pdf_basename, 
                    run_detail['metrics_file']
                )
                if os.path.exists(metrics_path):
                    with open(metrics_path, 'r', encoding='utf-8') as f:
                        metrics_data = json.load(f)
                    
                    # Reconstruct RunMetrics object with explicit type conversion
                    run_metrics = RunMetrics(
                        run_id=int(metrics_data['run_id']),
                        wer=float(metrics_data['wer']),
                        mer=float(metrics_data['mer']),
                        wil=float(metrics_data['wil']),
                        cer=float(metrics_data['cer']),
                        lev_distance=int(metrics_data.get('lev_distance', 0)),
                        lev_norm=float(metrics_data.get('lev_norm', 0.0)),
                        structural_accuracy=metrics_data.get('structural_accuracy', {}),
                        completeness=float(metrics_data['completeness']),
                        word_mismatches=metrics_data.get('word_mismatches', []),
                        structural_analysis=metrics_data.get('structural_analysis', {})
                    )
                    run_metrics_list.append(run_metrics)
        
        # Generate comprehensive report
        return self.generate_comprehensive_report(summary, run_metrics_list, model_folder)