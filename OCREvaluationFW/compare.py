import os
import matplotlib.pyplot as plt
from datetime import datetime
import re

# ---------------------------------------------------------
# Main OCR Evaluation Function for Markdown Files
# ---------------------------------------------------------
def evaluate_ocr_performance(
    gt_file=None,
    ocr_file=None,
    output_dir="reports",
    custom_name=None,
    display_model_name=None,
    gt_display_name=None,
    ocr_display_name=None,
    gt_text=None,
    ignore_image_desc: bool = False,
):
    """
    Evaluate OCR performance by comparing ground truth and OCR output markdown files.

    Args:
        gt_file: Path to ground truth .md file
        ocr_file: Path to OCR output .md file
        output_dir: Directory to save evaluation reports
        custom_name: Optional custom name for the report files (overrides automatic naming)
        display_model_name: Optional human-friendly model name to show inside the report
        gt_display_name: Optional display name for the ground truth file (used in report)
        ocr_display_name: Optional display name for the ocr file (used in report)
    """
    # 1. Read files / accept gt_text directly (do not modify uploaded GT file)
    if gt_text is None:
        if not gt_file:
            raise ValueError("Either gt_file or gt_text must be provided")
        with open(gt_file, "r", encoding="utf-8") as f:
            gt_text = f.read().strip()
    else:
        # Use provided GT text as-is; do not alter the uploaded ground truth file
        gt_text = str(gt_text).strip() if gt_text else ""

    if not ocr_file:
        raise ValueError("ocr_file must be provided (path to OCR output .md)")
    with open(ocr_file, "r", encoding="utf-8") as f:
        ocr_text = f.read().strip()
    
    # Ensure both texts are strings
    gt_text = str(gt_text) if gt_text else ""
    ocr_text = str(ocr_text) if ocr_text else ""

    # Optional preprocessing to ignore image descriptions (captions, alt text)
    def _strip_image_descriptions(text: str) -> str:
        import re as _re
        if not isinstance(text, str):
            text = "" if text is None else str(text)
        # Remove explicit image markups
        text = _re.sub(r'!\[[^\]]*\]\([^\)]+\)', ' ', text)  # Markdown image
        text = _re.sub(r'<img\b[^>]*>', ' ', text, flags=_re.I)  # HTML <img>
        text = _re.sub(r'<figcaption\b[^>]*>.*?</figcaption>', ' ', text, flags=_re.I | _re.S)  # HTML figcaption
        # Remove likely caption/figure lines only if image tokens exist
        has_image_tokens = ('![' in text) or ('<img' in text.lower()) or ('<figure' in text.lower())
        if has_image_tokens:
            caption_line_re = _re.compile(r'^\s*(?:Figure|Fig\.|Image|Illustration)\s*\d*\s*[:.\-]\s*.*$', _re.I)
            labelled_line_re = _re.compile(r'^\s*(?:Caption|Image\s*Description|Alt\s*Text|Photo\s*Caption)\s*[:\-]\s*.*$', _re.I)
            filtered_lines = []
            for line in text.splitlines():
                if caption_line_re.match(line) or labelled_line_re.match(line):
                    continue
                filtered_lines.append(line)
            text = "\n".join(filtered_lines)
        return text

    # Keep original GT for creating a commented temporary file (do not modify original file)
    gt_text_original = gt_text
    if ignore_image_desc:
        gt_text = _strip_image_descriptions(gt_text)
        ocr_text = _strip_image_descriptions(ocr_text)

    # 2. Compute all metrics using unified function (with diagnostics)
    from ocr_eval_utils import compute_all_metrics
    try:
        metrics_result = compute_all_metrics(gt_text, ocr_text)
    except Exception as e:
        raise RuntimeError(f"compute_all_metrics failed: {e} | gt_type={type(gt_text).__name__}, ocr_type={type(ocr_text).__name__}")
    
    # Extract individual metrics
    wer = metrics_result['wer']
    mer = metrics_result['mer']
    wil = metrics_result['wil']
    cer_value = metrics_result['cer']
    lev_distance = metrics_result['lev_distance']
    lev_norm = metrics_result['lev_norm']
    completeness = metrics_result['completeness']
    structure_metrics = metrics_result['structural_accuracy']
    structural_analysis = metrics_result['structural_analysis']
    top_mismatches = metrics_result['top_mismatches_raw']

    # 4. Prepare report directory
    os.makedirs(output_dir, exist_ok=True)

    # Friendly model name from file (handle both .md and .txt extensions for compatibility)
    # model_name is used for filenames; model_display is used inside the report content
    if custom_name:
        model_name = custom_name
    else:
        model_name = os.path.basename(ocr_file).replace(".md", "").replace(".txt", "")

    # Display names for the report content
    model_display = display_model_name if display_model_name else model_name
    gt_display = gt_display_name if gt_display_name else gt_file
    ocr_display = ocr_display_name if ocr_display_name else ocr_file

    # 5. Additional markdown-specific metrics
    def compute_markdown_structure_similarity(gt_text, ocr_text):
        """Compute similarity of markdown structure elements and provide richer structural details."""
        # Helper to extract header texts and levels: returns list of (level, text)
        header_pattern = re.compile(r'^(#{1,6})\s*(.+)$', re.MULTILINE)
        gt_headers_matches = header_pattern.findall(gt_text)
        ocr_headers_matches = header_pattern.findall(ocr_text)
        gt_header_texts = [h[1].strip().lower() for h in gt_headers_matches]
        ocr_header_texts = [h[1].strip().lower() for h in ocr_headers_matches]
        
        # Counts
        gt_headers = len(gt_header_texts)
        ocr_headers = len(ocr_header_texts)
        
        # Count tables (simple pipe-based detection)
        table_pattern = re.compile(r'\|.*\|')
        gt_table_lines = table_pattern.findall(gt_text)
        ocr_table_lines = table_pattern.findall(ocr_text)
        gt_tables = len(gt_table_lines)
        ocr_tables = len(ocr_table_lines)
        
        # Count lists (unordered)
        list_pattern = re.compile(r'^[\*\-\+]\s', re.MULTILINE)
        gt_lists = len(list_pattern.findall(gt_text))
        ocr_lists = len(list_pattern.findall(ocr_text))
        
        # Extract links [text](url)
        link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
        gt_links = link_pattern.findall(gt_text)
        ocr_links = link_pattern.findall(ocr_text)
        gt_link_targets = [t[1].strip() for t in gt_links]
        ocr_link_targets = [t[1].strip() for t in ocr_links]
        
        return {
            'headers': {
                'gt': gt_headers,
                'ocr': ocr_headers,
                'gt_texts': gt_header_texts,
                'ocr_texts': ocr_header_texts
            },
            'tables': {
                'gt': gt_tables,
                'ocr': ocr_tables,
                'gt_table_lines': gt_table_lines,
                'ocr_table_lines': ocr_table_lines
            },
            'lists': {
                'gt': gt_lists,
                'ocr': ocr_lists
            },
            'links': {
                'gt': len(gt_link_targets),
                'ocr': len(ocr_link_targets),
                'gt_targets': gt_link_targets,
                'ocr_targets': ocr_link_targets
            }
        }
 
    # Helper to format metrics (show NA when metric is None)
    def _fmt_metric(v, digits=4):
        return "NA" if v is None else f"{v:.{digits}f}"

    # 6. Write text summary (plain text) and a markdown summary for nicer viewing
    try:
        report_text = f"""
OCR Evaluation Report (Markdown)
================================
Model: {model_display}
GT File: {gt_display}
OCR Output: {ocr_display}
Timestamp: {datetime.now()}

---- Text Similarity Metrics ----
WER       : {wer:.4f}
MER       : {mer:.4f}
WIL       : {wil:.4f}
CER       : {cer_value:.4f}
LEV (edits): {lev_distance}
Completeness: {completeness:.4f}

---- Markdown Structure Metrics ----
Headers - GT: {structure_metrics['headers']['gt']}, OCR: {structure_metrics['headers']['ocr']}
Tables  - GT: {structure_metrics['tables']['gt']}, OCR: {structure_metrics['tables']['ocr']}
Lists   - GT: {structure_metrics['lists']['gt']}, OCR: {structure_metrics['lists']['ocr']}
Links   - GT: {structure_metrics['links']['gt']}, OCR: {structure_metrics['links']['ocr']}

---- Structural Analysis ----
Heading Alignment: {_fmt_metric(structural_analysis['heading_alignment'])}
List Accuracy: {_fmt_metric(structural_analysis['list_accuracy'])}
Table Preservation: {_fmt_metric(structural_analysis['table_preservation'])}
Link Correctness: {_fmt_metric(structural_analysis['link_correctness'])}
Section Ordering: {_fmt_metric(structural_analysis['section_ordering'])}

---- Top Word Mismatches ----
{chr(10).join([f"Ground Truth: '{pair[0][0]}' vs OCR: '{pair[0][1]}' (x{pair[1]})" for pair in top_mismatches])}
"""
    except Exception as e:
        raise RuntimeError(f"building text report failed: {e}")

    # Plain text summary (backwards compatible)
    summary_txt_path = os.path.join(output_dir, f"{model_name}_report.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Markdown summary - nicer to render in UI and for saving as .md
    md_lines = []
    md_lines.append(f"# 📊 OCR Evaluation Report\n")
    md_lines.append(f"**🤖 Model:** `{model_display}`  ")
    md_lines.append(f"**📄 Ground Truth:** `{gt_display}`  ")
    md_lines.append(f"**📝 OCR Output:** `{ocr_display}`  ")
    md_lines.append(f"**🕒 Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n")

    # Unified single-run metrics table (aligns with Multi-Run per-run table)
    md_lines.append("## 📊 Unified Metrics\n")
    md_lines.append("| WER | MER | WIL | CER | LEV-DIST (chars) | Completeness | Heading Alignment | List Accuracy | Table Preservation | Link Correctness | Section Ordering |")
    md_lines.append("|-----|-----|-----|-----|-------------------|--------------|-------------------|---------------|--------------------|------------------|------------------|")
    md_lines.append(
        "| "
        f"{wer:.4f} | "
        f"{mer:.4f} | "
        f"{wil:.4f} | "
        f"{cer_value:.4f} | "
        f"{int(lev_distance)} | "
        f"{completeness:.4f} | "
        f"{_fmt_metric(structural_analysis['heading_alignment'])} | "
        f"{_fmt_metric(structural_analysis['list_accuracy'])} | "
        f"{_fmt_metric(structural_analysis['table_preservation'])} | "
        f"{_fmt_metric(structural_analysis['link_correctness'])} | "
        f"{_fmt_metric(structural_analysis['section_ordering'])} |"
    )
    md_lines.append("\n")
    md_lines.append("*Lower values indicate better OCR performance (except Completeness where 1.0 is ideal). Structural metrics range from 0.0–1.0 (higher is better).*\n")

    md_lines.append("## 🔍 Top Word Mismatches\n")
    if top_mismatches:
        md_lines.append("| Ground Truth | OCR Output | Count |")
        md_lines.append("|---|---|---:|")
        for (g, o), cnt in top_mismatches:
            md_lines.append(f"| `{g}` | `{o}` | **{cnt}** |")
    else:
        md_lines.append("✅ _No major word mismatches detected._")

    try:
        md_content = "\n".join(md_lines)
    except Exception as e:
        raise RuntimeError(f"building markdown report failed: {e}")

    summary_md_path = os.path.join(output_dir, f"{model_name}_report.md")
    with open(summary_md_path, "w", encoding="utf-8") as f_md:
        f_md.write(md_content)

    # 7. Visual Plot - include completeness metric
    metrics = ["WER", "MER", "WIL", "CER", "Completeness"]
    values = [wer, mer, wil, cer_value, completeness]

    # Ensure we have valid numeric values for plotting
    def safe_numeric_value(v):
        """Convert value to safe numeric value for plotting."""
        if v is None:
            return 0.0
        try:
            # Convert to float if it's a string or other type
            numeric_v = float(v)
            # Clamp to [0, 1] range for plotting
            return max(0.0, min(1.0, numeric_v))
        except (ValueError, TypeError):
            # If conversion fails, return 0
            return 0.0
    
    try:
        values = [safe_numeric_value(v) for v in values]
    except Exception as e:
        raise RuntimeError(f"normalizing metrics for plotting failed: {e} | values={values}")

    try:
        plt.figure(figsize=(14, 7))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#9B59B6']
        bars = plt.bar(metrics, values, color=colors[:len(metrics)])

        plt.title(f"📊 OCR Evaluation Metrics: {model_name}", fontsize=16, fontweight='bold', pad=20)
        plt.ylabel("Error Rate", fontsize=13)
        plt.xlabel("Metrics", fontsize=13)

        # Safe max calculation
        max_value = max(values) if values else 0.0
        # Guard against non-finite values
        if not isinstance(max_value, (int, float)):
            max_value = 0.0
        plt.ylim(0, float(max_value) + 0.1)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)

        # Add a subtle grid and improve styling
        plt.grid(axis='y', alpha=0.3, linestyle='--')
        plt.xticks(rotation=0, fontsize=11)
        plt.yticks(fontsize=11)

        # Add a note about lower being better
        plt.figtext(0.5, 0.02, "Lower values indicate better OCR performance",
                    ha='center', fontsize=10, style='italic', color='gray')

        plt.tight_layout()

        chart_path = os.path.join(output_dir, f"{model_name}_metrics.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        raise RuntimeError(f"plotting metrics failed: {e}")

    print("Report generated:")
    print("Text Summary:", summary_txt_path)
    print("Markdown Summary:", summary_md_path)
    print("Graph:", chart_path)
    return summary_txt_path, summary_md_path, chart_path


import argparse

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description="Run OCR evaluation")
   parser.add_argument("--gt_file", required=True, help="Path to ground truth file")
   parser.add_argument("--ocr_file", required=True, help="Path to OCR output file")
   parser.add_argument("--output_dir", default="evaluation/results", help="Output directory")
   
   args = parser.parse_args()
   evaluate_ocr_performance(
       gt_file=args.gt_file,
       ocr_file=args.ocr_file,
       output_dir=args.output_dir
   )
