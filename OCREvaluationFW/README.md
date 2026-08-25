## Streamlit Datalab OCR Playground

This Streamlit app uploads a PDF, forwards it to the [Datalab Marker / Chandra OCR API](https://documentation.datalab.to/docs/welcome/api#marker), and shows the resulting markdown next to the original document for quick side‑by‑side review. The model capabilities come from [datalab-to/chandra](https://huggingface.co/datalab-to/chandra).

### Setup
- Create a virtual environment (Python 3.10+ recommended).
- Install dependencies: `pip install -r requirements.txt`.
- Copy `config.example.toml` to `config.toml` and replace `YOUR_API_KEY` with your real Datalab API key. You can tune Marker (`endpoint`, `fallback_endpoints`) and Surya (`ocr_endpoint`, `ocr_fallback_endpoints`) URLs as well as the timeout settings there.
- **Multi-Run Evaluation Configuration**: In `config.toml`, you can also configure multi-run evaluation settings:
    - `multi_run_enabled`: Set to `true` to enable the multi-run feature by default.
    - `multi_run_runs`: Number of independent runs to perform (default is 3).
    - `multi_run_require_all_success`: If `true`, aggregation only occurs if all runs succeed.

### Running the app
```bash
streamlit run streamlit_app.py
```

Upload a PDF, choose between **Marker** (PDF → Markdown/HTML/JSON) or **Surya OCR** (line-level text with bounding boxes), tweak options such as output format, `use_llm`, `force_ocr`, or `max_pages`, and click **Run with Datalab**. The extracted text appears next to the embedded PDF preview along with a collapsible view of the raw API response to aid debugging.

---

## 📊 Multi-Run Evaluation Consistency Strategy

To ensure that model performance is stable and deterministic, each model can be evaluated multiple times on the same dataset. This reduces noise and builds confidence in the results. To ensure reproducibility and transparent evaluation, each of the execution runs will log the exact model configuration used during extraction. This eliminates ambiguity and ensures that output differences are not caused by configuration drift.

### 8.1 Objective

To ensure that model performance is stable and deterministic, each model is evaluated three times on the same dataset. This reduces noise and builds confidence in the results. To ensure reproducibility and transparent evaluation, each of the three execution runs will log the exact model configuration used during extraction. This eliminates ambiguity and ensures that output differences are not caused by configuration drift.

For every run, the following settings will be captured and included in the report:

- **Model details**: Name, version, model family (LLM/OCR).
- **Inference parameters**: Temperature, top-p, max tokens, penalties, etc.

### 8.2 Evaluation Process

For every PDF and every model:

- Run the extraction three independent times.
- For each run, compute all metrics.
- Store and label results as Run 1, Run 2, Run 3.
- Generate separate metric blocks for each run.
- No cached or intermediate results are reused.

### 8.3 Result Recording

Each run appears separately in the final report:

- Run 1 Metrics
- Run 2 Metrics
- Run 3 Metrics

Each includes:

- CER, WER, MER, WIL
- Levenshtein distance
- Structural accuracy
- Completeness
- Word mismatches

### 8.4 Consistency Scoring

The framework computes:

- Mean score across 3 runs
- Standard deviation
- Variance trend
- A Consistency Confidence Index (CCI)

`CCI = 1 - (StdDev / Mean Score)`

### 8.5 Interpretation Guidelines

- `CCI > 0.90` → Highly stable
- `0.75–0.90` → Moderately stable
- `< 0.75` → Unstable results

### 8.6 Reporting Format

The report includes:

- Comparative score tables for all 3 runs
- Visual graphs showing variance
- Summary observations
- Recommendation on model stability

---

## 🚀 Features

### Core Functionality
- **PDF Upload & Processing**: Upload single or multiple PDF files for OCR processing
- **Model Selection**: Choose between Marker (PDF → Markdown/HTML/JSON) or Surya OCR (text extraction)
- **Real-time Preview**: Side-by-side comparison of original PDF and extracted text
- **Manual File Comparison**: Upload ground truth and OCR output files for direct comparison

### Multi-Run Evaluation
- **Consistency Analysis**: Run multiple independent evaluations to assess model stability
- **Comprehensive Metrics**: WER, MER, WIL, CER, Levenshtein distance, structural accuracy, completeness
- **Visual Reports**: Charts showing metric variance and run-by-run comparisons
- **Stability Scoring**: Consistency Confidence Index (CCI) with interpretation guidelines
- **Detailed Logging**: Per-run configuration and metadata tracking

### Evaluation Metrics
- **WER (Word Error Rate)**: Measures word-level accuracy
- **MER (Match Error Rate)**: Evaluates sequence matching performance
- **WIL (Word Information Lost)**: Assesses information preservation
- **CER (Character Error Rate)**: Character-level accuracy measurement
- **Levenshtein Distance**: Edit distance between texts
- **Structural Accuracy**: Preservation of document structure (headers, tables, lists)
- **Completeness**: Ratio of OCR output length to ground truth length

---

## 📁 Directory Structure

```
OCREvaluationFW/
├── streamlit_app.py              # Main Streamlit application
├── multi_run_evaluation.py       # Multi-run evaluation engine
├── multi_run_reporter.py         # Report generation and visualization
├── compare.py                     # Evaluation metrics computation
├── config.py                     # Configuration management
├── test_multi_run.py             # Comprehensive test suite
├── requirements.txt              # Python dependencies
├── config.example.toml           # Configuration template
├── evaluation/
│   └── results/                  # Evaluation results and reports
│       ├── Marker/               # Marker model results
│       └── Surya/                # Surya model results
├── OCR_Output/                   # OCR extraction outputs
│   ├── Marker/                   # Marker outputs with metadata
│   └── Surya/                    # Surya outputs with metadata
└── ground_truth/                 # Ground truth reference files
```

### Multi-Run Directory Structure

For each multi-run evaluation, the following structure is created:

```
evaluation/results/{model}/{pdf_name}/
├── summary_multirun.json         # Aggregated results summary
├── multirun_report.md            # Comprehensive markdown report
├── multirun_report.txt           # Plain text report
├── metrics_variance_chart.png    # Variance analysis chart
├── run_comparison_chart.png      # Run-by-run comparison chart
├── run_1/                        # First run artifacts
│   ├── config.json               # Run configuration
│   ├── metrics.json              # Computed metrics
│   └── raw_output.md             # Raw OCR output
├── run_2/                        # Second run artifacts
│   └── ...
└── run_3/                        # Third run artifacts
    └── ...
```

---

## 🔧 Configuration

### Basic Configuration (`config.toml`)

```toml
# API Configuration
api_key = "YOUR_API_KEY"
endpoint = "https://api.datalab.to/api/v1/marker"
ocr_endpoint = "https://api.datalab.to/api/v1/ocr"

# Timeout Settings
request_timeout_seconds = 300
poll_timeout_seconds = 600
poll_interval_seconds = 5

# Multi-Run Evaluation Settings
multi_run_enabled = true
multi_run_runs = 3
multi_run_require_all_success = false
```

### Configuration Options

- **`multi_run_enabled`**: Enable multi-run evaluation by default in the UI
- **`multi_run_runs`**: Number of independent runs to perform (default: 3)
- **`multi_run_require_all_success`**: If true, aggregation only occurs if all runs succeed

---

## 📊 Usage Examples

### Single PDF Evaluation

1. **Upload PDF**: Use the file uploader to select your PDF document
2. **Select Model**: Choose between Marker or Surya OCR
3. **Configure Options**: Set language, processing parameters, etc.
4. **Enable Multi-Run**: Check "Enable Multi-Run Consistency Evaluation" for stability analysis
5. **Run Evaluation**: Click "Run Multi-Run Evaluation" or "Run Auto Extract & Evaluate"

### Manual File Comparison

1. **Upload Ground Truth**: Upload your reference markdown file
2. **Upload OCR Output**: Upload the OCR-generated markdown file
3. **Run Comparison**: Click "Run Comparison" to generate evaluation metrics
4. **View Results**: Review metrics, charts, and detailed difference analysis

### Interpreting Results

#### Consistency Confidence Index (CCI)
- **CCI > 0.90**: 🟢 Highly stable - Excellent consistency, suitable for production
- **CCI 0.75-0.90**: 🟡 Moderately stable - Good consistency with some variance
- **CCI < 0.75**: 🔴 Unstable - Significant variance, investigate model configuration

#### Metric Interpretation
- **Lower is better**: WER, MER, WIL, CER, Levenshtein Distance
- **Higher is better**: Completeness, Structural Accuracy matches

---

## 🧪 Testing

### Run Simple Tests
```bash
python test_multi_run.py --simple
```

### Run Full Test Suite
```bash
python -m unittest test_multi_run.py -v
```

### Test Coverage
The test suite includes:
- **Unit Tests**: Individual component testing (RunConfig, RunMetrics, Aggregator, etc.)
- **Integration Tests**: End-to-end multi-run workflow validation
- **Mock Testing**: API interaction simulation
- **Error Handling**: Failure scenario testing
- **Chart Generation**: Visualization testing with matplotlib mocking

---

## 🔍 Troubleshooting

### Common Issues

#### Charts Not Displaying
- **Issue**: Metrics Variance Chart or Run Comparison Chart not loading
- **Solution**: Ensure matplotlib and seaborn are installed; check file permissions in evaluation/results/

#### Network Connectivity
- **Issue**: "All configured Datalab endpoints failed"
- **Solution**: Use the "Run Full Network Diagnostics" feature to identify connectivity issues
- **Alternative**: Use Manual File Comparison for offline evaluation

#### Insufficient Successful Runs
- **Issue**: "Insufficient successful runs (X/3). Cannot compute reliable aggregates."
- **Solution**: Check API connectivity, verify PDF format, ensure sufficient API quota

### Debug Features
- **Network Diagnostics**: Comprehensive connectivity testing for Datalab endpoints
- **Endpoint Testing**: Quick reachability checks for API services
- **Detailed Error Messages**: Specific guidance for common failure scenarios
- **Progress Tracking**: Real-time status updates during multi-run evaluation

---

## 📈 Performance Considerations

### Multi-Run Evaluation
- **Duration**: Each run involves independent API calls, so total time = (single run time) × (number of runs)
- **API Limits**: Ensure sufficient API quota for multiple runs
- **Storage**: Each run generates ~3-5 files, plan storage accordingly

### Optimization Tips
- **Parallel Processing**: Future enhancement - currently runs are sequential for consistency
- **Caching**: Disabled by design to ensure independent runs
- **Batch Processing**: Consider processing multiple PDFs in separate sessions

---

## 🤝 Contributing

### Development Setup
1. Clone the repository
2. Create virtual environment: `python -m venv .venv`
3. Activate environment: `.venv\Scripts\activate` (Windows) or `source .venv/bin/activate` (Unix)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy configuration: `cp config.example.toml config.toml`
6. Run tests: `python test_multi_run.py --simple`

### Adding New Features
- Follow the existing code structure and patterns
- Add comprehensive tests for new functionality
- Update documentation and configuration examples
- Ensure backward compatibility with existing evaluations

---

## 📄 License

This project is part of the OCR Evaluation Framework for assessing and comparing OCR model performance with multi-run consistency analysis.

