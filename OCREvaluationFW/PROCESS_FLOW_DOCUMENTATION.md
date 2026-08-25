# OCR Evaluation Framework - Process Flow Documentation

This document describes the complete process flow for both Single Run and Multi Run evaluations in the OCR Evaluation Framework, starting from when a user uploads a PDF file and ground truth markdown file.

## Overview

The OCR Evaluation Framework provides two main evaluation modes:
1. **Single Run**: Performs one OCR extraction and evaluation
2. **Multi Run**: Performs multiple independent OCR extractions for consistency analysis

Both modes use identical metric computation logic to ensure consistent results.

---

## Single Run Process Flow

### Phase 1: File Upload and Validation
```
User Action: Upload PDF + Ground Truth (.md)
    ↓
Streamlit UI: Validate file types and content
    ↓
Display: PDF preview and GT preview
    ↓
User Action: Select model (Marker/Surya) and configure options
```

### Phase 2: OCR Processing
```
User Action: Click "Run" button
    ↓
streamlit_app.py: process_single_file()
    ↓
API Submission:
├─ Marker: _submit_marker_job() → Datalab Marker API
└─ Surya: _submit_ocr_job() → Datalab OCR API
    ↓
API Processing: Remote OCR extraction
    ↓
Response: _extract_markdown() extracts text from API response
    ↓
File Storage: Save to OCR_Output/{Model}/{filename}_ocr_output.md
```

### Phase 3: Evaluation
```
OCR Text Available
    ↓
streamlit_app.py: Call evaluate_ocr_performance()
    ↓
compare.py: evaluate_ocr_performance(gt_text=uploaded_gt_content, ocr_file=ocr_path)
    ↓
Ground Truth Processing:
├─ Use gt_text parameter directly (uploaded GT content)
└─ No file modifications or temporary files
    ↓
OCR Text Processing:
└─ Read from saved OCR file
    ↓
Metric Computation: ocr_eval_utils.compute_all_metrics(gt_text, ocr_text)
├─ Text preprocessing: preprocess_markdown_for_evaluation()
├─ Word-level metrics: jiwer (WER, MER, WIL)
├─ Character-level metrics: custom cer(), Levenshtein distance
├─ Structural analysis: analyze_markdown_structure()
└─ Word mismatches: compute_top_word_mismatches()
    ↓
Report Generation:
├─ Create metrics visualization chart
├─ Generate markdown report
├─ Generate text report
└─ Save to evaluation/results/{Model}/
```

### Phase 4: Results Display
```
Evaluation Complete
    ↓
Streamlit UI Display:
├─ Extracted OCR text
├─ Comprehensive evaluation report
├─ Metrics visualization chart
├─ Side-by-side text comparison
└─ Download options (reports, charts)
```

---

## Multi Run Process Flow

### Phase 1: File Upload and Validation
```
User Action: Upload PDF + Ground Truth (.md)
    ↓
Streamlit UI: Validate file types and content
    ↓
Display: PDF preview and GT preview
    ↓
User Action: Select "Multi-Run Consistency Evaluation"
    ↓
User Action: Select model (Marker/Surya) and configure options
```

### Phase 2: Multi-Run Orchestration
```
User Action: Click "Run" button
    ↓
streamlit_app.py: Initialize MultiRunRunner(num_runs=3)
    ↓
Ground Truth Preparation:
├─ Extract GT text: uploaded_gt.getvalue().decode('utf-8')
└─ Store as gt_text variable (no file modifications)
    ↓
Extraction Function Setup:
├─ Create wrapper: create_extraction_wrapper()
├─ Configure API parameters
└─ Add 5-second delay between runs
```

### Phase 3: Multiple OCR Extractions
```
MultiRunRunner.execute_multi_run_evaluation()
    ↓
For each run (1 to 3):
    ↓
    5-second delay (for deterministic spacing)
    ↓
    API Submission:
    ├─ Marker: _submit_marker_job() → Datalab API
    └─ Surya: _submit_ocr_job() → Datalab API
    ↓
    Response Processing: _extract_markdown()
    ↓
    Per-Run Storage:
    ├─ Save raw_output.md to evaluation/results/{Model}/{pdf_basename}/run_{n}/
    ├─ Compute metrics: _compute_metrics_from_texts(gt_text, ocr_text)
    │   └─ Uses: ocr_eval_utils.compute_all_metrics() (same as single run)
    ├─ Save metrics.json
    └─ Save config.json
    ↓
    Progress Update: Display run status to user
```

### Phase 4: Aggregation and Analysis
```
All Runs Complete
    ↓
MultiRunAggregator.aggregate_runs():
├─ Load all run metrics from metrics.json files
├─ Compute statistical aggregations (mean, std, min, max)
├─ Calculate Consistency Confidence Index (CCI)
└─ Determine stability interpretation
    ↓
Save Aggregated Results:
└─ summary_multirun.json
```

### Phase 5: Comprehensive Reporting
```
MultiRunReporter.generate_comprehensive_report():
    ↓
Report Generation:
├─ Create detailed markdown report
├─ Generate text report
├─ Create metrics variance chart
├─ Create run comparison chart
└─ Save to evaluation/results/{Model}/{pdf_basename}/
    ↓
Chart Generation:
├─ Metrics variance visualization
└─ Run-by-run comparison visualization
```

### Phase 6: Results Display
```
Multi-Run Complete
    ↓
Streamlit UI Display:
├─ Summary metrics (total runs, successful runs, overall CCI)
├─ Stability interpretation
├─ Comprehensive multi-run report
├─ Visual analysis charts
├─ Side-by-side comparisons for each run
└─ Download options (reports, charts, summary JSON)
```

---

## Key Components and Their Roles

### Core Files

#### streamlit_app.py
- **Role**: Main UI orchestration and user interaction
- **Key Functions**:
  - `process_single_file()`: Single run orchestration
  - `_submit_marker_job()` / `_submit_ocr_job()`: API communication
  - `_extract_markdown()`: Response processing
  - Multi-run initialization and progress tracking

#### compare.py
- **Role**: Single run evaluation logic
- **Key Function**: `evaluate_ocr_performance()`
- **Updated Signature**: Accepts `gt_text` parameter for direct GT content
- **Ensures**: No modifications to uploaded GT file

#### multi_run_evaluation.py
- **Role**: Multi-run orchestration and per-run metrics
- **Key Classes**:
  - `MultiRunRunner`: Orchestrates multiple extractions
  - `MultiRunAggregator`: Statistical analysis of runs
- **Key Function**: `_compute_metrics_from_texts()` uses same `compute_all_metrics()`

#### multi_run_reporter.py
- **Role**: Multi-run report generation and visualization
- **Key Function**: `generate_comprehensive_report()`
- **Outputs**: Markdown/text reports, variance charts, comparison charts

#### ocr_eval_utils.py
- **Role**: Unified metric computation (single source of truth)
- **Key Function**: `compute_all_metrics(gt_text, ocr_text)`
- **Metrics Computed**:
  - Word Error Rate (WER)
  - Match Error Rate (MER)
  - Word Information Lost (WIL)
  - Character Error Rate (CER)
  - Levenshtein distance and normalized distance
  - Structural accuracy and analysis
  - Word mismatch analysis

### Data Flow Consistency

#### Ground Truth Handling
- **Single Run**: Uses `gt_text` parameter with uploaded content directly
- **Multi Run**: Uses same uploaded GT content for all runs
- **Guarantee**: Both modes use identical GT text (no file modifications)

#### OCR Text Processing
- **Single Run**: Reads from saved OCR output file
- **Multi Run**: Uses extracted text directly, saves to run-specific files
- **Consistency**: Same text processing pipeline

#### Metric Computation
- **Both Modes**: Use `ocr_eval_utils.compute_all_metrics()`
- **Input**: Identical GT and OCR text strings
- **Output**: Identical metric values when same inputs used

---

## File Structure

```
OCREvaluationFW/
├── streamlit_app.py              # Main UI and orchestration
├── compare.py                    # Single run evaluation
├── multi_run_evaluation.py       # Multi-run orchestration
├── multi_run_reporter.py         # Multi-run reporting
├── ocr_eval_utils.py             # Unified metrics computation
├── config.py                     # Configuration management
├── OCR_Output/                   # OCR extraction outputs
│   ├── Marker/
│   └── Surya/
├── ground_truth/                 # Reference ground truth files
└── evaluation/
    └── results/                  # Evaluation reports and metrics
        ├── Marker/
        │   ├── {pdf_basename}_report.md     # Single run reports
        │   └── {pdf_basename}/              # Multi-run results
        │       ├── run_1/
        │       │   ├── raw_output.md
        │       │   ├── metrics.json
        │       │   └── config.json
        │       ├── run_2/ ...
        │       ├── run_3/ ...
        │       ├── multirun_report.md
        │       ├── summary_multirun.json
        │       └── *.png (charts)
        └── Surya/ (same structure)
```

---

## Quality Assurance

### Metric Consistency
- Both single and multi-run use identical `compute_all_metrics()` function
- Same text preprocessing pipeline
- Same ground truth content (no file modifications)
- Verified through debug testing and validation scripts

### Error Handling
- Network connectivity diagnostics
- API failure recovery with fallback endpoints
- Graceful degradation for offline usage
- Comprehensive error reporting

### User Experience
- Real-time progress tracking for multi-run
- Visual diff comparisons
- Multiple download formats
- Responsive UI with clear status indicators

---

## Technical Notes

### API Integration
- Supports both Marker (document conversion) and Surya (OCR) models
- Automatic failover between multiple endpoints
- Configurable timeouts and retry logic
- Language-specific processing options

### Performance Considerations
- Multi-run includes 5-second delays between runs for API stability
- Efficient file I/O with minimal temporary file usage
- Streaming progress updates for long-running operations
- Memory-efficient text processing

### Extensibility
- Modular design allows easy addition of new OCR models
- Configurable number of runs for multi-run evaluation
- Pluggable metric computation system
- Flexible report generation templates