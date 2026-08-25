# Prompt Evaluation Framework

A robust, extensible system for evaluating prompts across a wide range of state-of-the-art AI models. Built with Streamlit, it provides a user-friendly interface for prompt management, multi-model evaluation, and in-depth, automated response analysis.

---

## Key Features

### 🔹 Prompt Management
- **Text Input & File Upload:** Enter prompts directly or upload from `.txt`, `.md`, or `.json` files.
- **Prompt Library:** Save, load, and organize prompts for repeated use.
- **Prompt Statistics:** Instantly view character, word, and line counts.

### 🔹 Multi-Model Evaluation
- **Unified LLM Routing:** Seamlessly send prompts to multiple models via RouteLLM/AbacusAI, OpenAI, or Anthropic APIs.
- **Custom Model Configurations:** Set temperature, top_p, max_tokens, and more for each model.
- **Batch & Parallel Evaluation:** Evaluate a single prompt across many models simultaneously.
- **Response Comparison:** Side-by-side display of all model outputs for easy comparison.

### 🔹 Automated Metrics & Analysis
- **BLEU Score:** Measures n-gram overlap (BLEU-1 to BLEU-4 and overall BLEU) between model output and reference, indicating precision and phrase similarity.
- **ROUGE Score:** Computes ROUGE-1, ROUGE-2, and ROUGE-L (precision, recall, F1) for recall and sequence similarity.
- **Comprehensive Summary:** Generates a detailed, human-readable report with:
  - Text length analysis
  - Word and phrase overlap
  - Precision/recall balance
  - Key insights and actionable feedback
- **Exportable Reports:** Download results and summaries as JSON, CSV, or markdown.

### 🔹 Results Management
- **Automatic Storage:** Every evaluation is saved with full metadata (timestamp, prompt, model configs, scores).
- **History & Search:** Browse, search, and filter all past evaluations.
- **Export Options:** Download individual or bulk results for further analysis.

---

## Supported Models

Evaluate prompts on a wide selection of top-tier models, including but not limited to:

- **OpenAI:** GPT-4o, GPT-4.1, GPT-5, and their mini/nano variants
- **Anthropic:** Claude 3.7 Sonnet, Claude Opus 4, Claude Haiku 4.5, and more
- **Google Gemini:** Gemini 2.0 Flash, Gemini 2.5 Pro/Flash
- **Meta Llama:** Llama 3.1, Llama 4 Maverick, Llama 3.3 Versatile
- **Qwen:** Qwen 2.5, Qwen3, Qwen3 Coder
- **xAI Grok:** Grok 4, Grok Code Fast
- **DeepSeek, Kimi, and others** via RouteLLM/AbacusAI

See `models_info.json` for the full, up-to-date list with pricing and descriptions.

---

## Evaluation Metrics Explained

### BLEU (Bilingual Evaluation Understudy)
- **BLEU-1 to BLEU-4:** Measures n-gram precision (word and phrase overlap).
- **Overall BLEU:** Standard 4-gram BLEU for holistic similarity.
- **Interpretation:** High BLEU = strong word/phrase match; low BLEU = more paraphrasing or missing content.

### ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
- **ROUGE-1:** Unigram (word-level) overlap (precision, recall, F1).
- **ROUGE-2:** Bigram (2-word phrase) overlap.
- **ROUGE-L:** Longest common subsequence (sequence similarity).
- **Interpretation:** High ROUGE = strong recall and sequence match; low ROUGE = missing or reordered content.

### Automated Summary & Insights
- **Text Length Analysis:** Compares reference and response lengths.
- **Word & Phrase Overlap:** Highlights missing/extra words, bigram/trigram matches.
- **Precision/Recall Balance:** Diagnoses if responses are too short, verbose, or off-topic.
- **Key Insights:** Actionable feedback for improving prompts or model selection.

---

## How It Works

1. **Configure API Keys:** Enter your RouteLLM/AbacusAI, OpenAI, and Anthropic keys in the "API Keys" tab.
2. **Set Up Models:** Add and customize model configurations in the "Model Config" tab.
3. **Create or Upload Prompts:** Use the "Prompts" tab to manage your prompt library.
4. **Evaluate:** In the "Evaluate" tab, select prompts and models, then run evaluations.
5. **Review Results:** View detailed metrics, summaries, and download reports in the "Results History" tab.

---

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

3. Open your browser to [http://localhost:8501](http://localhost:8501)

---

## Security

- **API keys** are stored locally in `data/api_keys.json`. Do **not** commit this file to version control.
- For production, consider using environment variables or secure vaults.

---

## Extensibility

- **Add new models** by updating `models_info.json` and model config UI.
- **Custom metrics** can be added in `evaluation_metrics.py`.
- **Flexible storage:** All results are JSON-serializable for easy integration with other tools.

---

## Roadmap

- More advanced metrics (e.g., BERTScore, METEOR)
- Cost tracking and optimization
- Batch and multi-prompt evaluation
- Custom evaluation criteria and user-defined metrics

---

**Prompt Evaluation Framework** — The fastest way to compare, analyze, and improve your prompts across the latest AI models.

---

## Previous README (for reference)

<details>
<summary>Click to expand previous README content</summary>

...existing code...

</details>

