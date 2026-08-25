

import streamlit as st
import json
import pandas as pd
import re
import os
import importlib
from collections import Counter
from math import sqrt
from io import BytesIO
from statistics import median

from langdetect import detect_langs, DetectorFactory
import pycountry

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except Exception:
    TfidfVectorizer = None
    SKLEARN_AVAILABLE = False

try:
    bert_score = importlib.import_module("bert_score").score
    BERTSCORE_AVAILABLE = True
except Exception:
    bert_score = None
    BERTSCORE_AVAILABLE = False


# Make language detection deterministic
DetectorFactory.seed = 0

# Set page metadata early so the browser tab/header uses the app name.
st.set_page_config(page_title="Attribute Evaluation Framework")


def get_language_stats(text):
    """Return a mapping of language code -> percentage of text.

    Uses langdetect.detect_langs to get probabilities and normalizes
    them to integer percentages that sum to ~100.
    """
    if not text or not str(text).strip():
        return {}
    try:
        detections = detect_langs(text)
    except Exception:
        return {}

    totals = {}
    for det in detections:
        try:
            code = det.lang
            prob = float(det.prob)
        except Exception:
            continue
        if not code:
            continue
        totals[code] = totals.get(code, 0.0) + prob

    total_prob = sum(totals.values())
    if not total_prob:
        return {}

    stats = {}
    for code, prob in totals.items():
        pct = int(round((prob / total_prob) * 100))
        if pct > 0:
            stats[code] = pct
    return stats


def get_language_name(code):
    """Map a language code like 'en' or 'ca' to a human-readable name."""
    if not code:
        return "Unknown"
    try:
        # Try ISO 639-1
        lang = pycountry.languages.get(alpha_2=code.lower())
        if lang and hasattr(lang, "name"):
            return lang.name
        # Try ISO 639-3
        lang = pycountry.languages.get(alpha_3=code.lower())
        if lang and hasattr(lang, "name"):
            return lang.name
    except Exception:
        pass

    # Fallback for common codes
    fallback = {
        "en": "English",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "it": "Italian",
        "pt": "Portuguese",
        "zh": "Chinese",
        "ja": "Japanese",
        "ru": "Russian",
        "ar": "Arabic",
        "ca": "Catalan",
        "nl": "Dutch",
        "sv": "Swedish",
        "fi": "Finnish",
        "no": "Norwegian",
        "da": "Danish",
        "pl": "Polish",
        "tr": "Turkish",
        "el": "Greek",
        "ko": "Korean",
        "he": "Hebrew",
        "hi": "Hindi",
        "id": "Indonesian",
        "th": "Thai",
        "vi": "Vietnamese",
        "uk": "Ukrainian",
        "cs": "Czech",
        "ro": "Romanian",
        "hu": "Hungarian",
        "bg": "Bulgarian",
        "fa": "Persian",
        "ur": "Urdu",
        "ta": "Tamil",
        "te": "Telugu",
        "ml": "Malayalam",
        "bn": "Bengali",
        "pa": "Punjabi",
        "gu": "Gujarati",
        "kn": "Kannada",
        "mr": "Marathi",
        "or": "Odia",
        "si": "Sinhala",
        "my": "Burmese",
        "km": "Khmer",
        "lo": "Lao",
        "am": "Amharic",
        "zu": "Zulu",
        "xh": "Xhosa",
        "st": "Southern Sotho",
        "tn": "Tswana",
        "ts": "Tsonga",
        "ss": "Swati",
        "ve": "Venda",
        "nr": "South Ndebele",
        "nd": "North Ndebele",
        "rw": "Kinyarwanda",
        "so": "Somali",
        "yo": "Yoruba",
        "ig": "Igbo",
        "ha": "Hausa",
        "sw": "Swahili",
        "af": "Afrikaans",
    }
    return fallback.get(code.lower(), code.upper())


def format_language_stats(stats):
    """Format language stats dict into a human-readable multi-line string."""
    if not stats:
        return "Language detected: Unknown"
    lines = ["Language detected"]
    for lang, pct in sorted(stats.items(), key=lambda x: -x[1]):
        name = get_language_name(lang)
        lines.append(f"{name}\n{pct}%")
    return "\n".join(lines)


def language_stats_to_df(stats):
    """Convert language stats dict to a small DataFrame.

    Columns: Language, Code, Percentage.
    """
    rows = []
    for code, pct in sorted(stats.items(), key=lambda x: -x[1]):
        rows.append({
            "Language": get_language_name(code),
            "Code": code.upper(),
            "Percentage": pct,
        })
    if not rows:
        return pd.DataFrame(columns=["Language", "Code", "Percentage"])
    return pd.DataFrame(rows, columns=["Language", "Code", "Percentage"])


def _iter_generated_artifact_blocks(topic):
    artifacts = topic.get("generated_artifacts", []) if isinstance(topic, dict) else []
    if not isinstance(artifacts, list):
        return []
    return [b for b in artifacts if isinstance(b, dict)]


def _iter_summary_payloads_from_block(block):
    summary_block = block.get("summary")
    if not isinstance(summary_block, dict):
        return []

    content = summary_block.get("content")
    if isinstance(content, list):
        return [item for item in content if isinstance(item, dict)]

    # Legacy summary schema.
    return [summary_block]


def _iter_faq_entries_from_block(block):
    entries = []

    # Legacy schema: {"faqs": [{question, answer}, ...]}
    legacy = block.get("faqs")
    if isinstance(legacy, list):
        entries.extend(item for item in legacy if isinstance(item, dict))

    # New schema: {"faq": {"content": [{"faqs": [...]}, ...]}}
    faq_block = block.get("faq")
    if isinstance(faq_block, dict):
        content = faq_block.get("content")
        if isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue
                inner = c.get("faqs")
                if isinstance(inner, list):
                    entries.extend(item for item in inner if isinstance(item, dict))

    return entries


def _iter_keyword_entries_from_block(block):
    entries = []

    # Legacy schema: {"keywords": [{keyword, ...}, ...]}
    legacy = block.get("keywords")
    if isinstance(legacy, list):
        entries.extend(item for item in legacy if isinstance(item, dict))

    # New schema: {"keyword": {"content": [{"keywords": [...]}, ...]}}
    kw_block = block.get("keyword")
    if isinstance(kw_block, dict):
        content = kw_block.get("content")
        if isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue
                inner = c.get("keywords")
                if isinstance(inner, list):
                    entries.extend(item for item in inner if isinstance(item, dict))

    return entries


def _iter_entity_entries_from_block(block):
    entries = []

    # Legacy schema: {"entities": [{label, ...}, ...]}
    legacy = block.get("entities")
    if isinstance(legacy, list):
        entries.extend(item for item in legacy if isinstance(item, dict))

    # New schema: {"entity": {"content": [{"entities": [...]}, ...]}}
    ent_block = block.get("entity")
    if isinstance(ent_block, dict):
        content = ent_block.get("content")
        if isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue
                inner = c.get("entities")
                if isinstance(inner, list):
                    entries.extend(item for item in inner if isinstance(item, dict))

    return entries


def _iter_classification_entries_from_block(block):
    entries = []

    # Legacy schema: {"classifications": [{classification, ...}, ...]}
    legacy = block.get("classifications")
    if isinstance(legacy, list):
        entries.extend(item for item in legacy if isinstance(item, dict))

    # New schema: {"classification": {"content": [{"classifications": [...]}, ...]}}
    cls_block = block.get("classification")
    if isinstance(cls_block, dict):
        content = cls_block.get("content")
        if isinstance(content, list):
            for c in content:
                if not isinstance(c, dict):
                    continue
                inner = c.get("classifications")
                if isinstance(inner, list):
                    entries.extend(item for item in inner if isinstance(item, dict))

    return entries


def extract_summary_components_from_topic(topic):
    components = []
    found_generated_title = False

    if not isinstance(topic, dict):
        return components

    for block in _iter_generated_artifact_blocks(topic):
        for payload in _iter_summary_payloads_from_block(block):
            title = payload.get("title")
            if title and isinstance(title, str) and title.strip():
                components.append(("title", title.strip()))
                found_generated_title = True

            long_sum = payload.get("long_summary")
            if long_sum and isinstance(long_sum, str) and long_sum.strip():
                components.append(("long_summary", long_sum.strip()))

            short_sum = payload.get("short_summary")
            if short_sum and isinstance(short_sum, str) and short_sum.strip():
                components.append(("short_summary", short_sum.strip()))

    # Backward-compatible fallback when only page title is available.
    if not found_generated_title:
        page_title = topic.get("title")
        if page_title and isinstance(page_title, str) and page_title.strip():
            components.append(("title", page_title.strip()))

    return components


def extract_primary_summary_from_topic(topic):
    if not isinstance(topic, dict):
        return ""
    for block in _iter_generated_artifact_blocks(topic):
        for payload in _iter_summary_payloads_from_block(block):
            text = payload.get("long_summary") or payload.get("short_summary") or payload.get("summary") or ""
            if text and str(text).strip():
                return str(text)
    return ""


def extract_faq_texts_from_topic(topic):
    faqs_text = []
    if not isinstance(topic, dict):
        return faqs_text
    for block in _iter_generated_artifact_blocks(topic):
        for faq in _iter_faq_entries_from_block(block):
            q = faq.get("question") or ""
            a = faq.get("answer") or ""
            if q or a:
                faqs_text.append(f"Q: {q}\nA: {a}")
    return faqs_text


def extract_keywords_from_topic(topic):
    texts = []
    if not isinstance(topic, dict):
        return texts
    for block in _iter_generated_artifact_blocks(topic):
        for kw in _iter_keyword_entries_from_block(block):
            word = kw.get("keyword") or kw.get("key_word") or ""
            if word and str(word).strip():
                texts.append(str(word))
    return texts


def extract_entities_from_topic(topic):
    texts = []
    if not isinstance(topic, dict):
        return texts
    for block in _iter_generated_artifact_blocks(topic):
        for ent in _iter_entity_entries_from_block(block):
            label = ent.get("label") or ""
            if label and str(label).strip():
                texts.append(str(label))
    return texts


def extract_classifications_from_topic(topic):
    texts = []
    if not isinstance(topic, dict):
        return texts
    for block in _iter_generated_artifact_blocks(topic):
        for cls in _iter_classification_entries_from_block(block):
            name = cls.get("classification") or cls.get("name") or ""
            if name and str(name).strip():
                texts.append(str(name))
    return texts


def _compute_artifact_stats_generic(data):
    stats = {
        "derived.total_faqs": 0,
        "derived.total_summaries": 0,
        "derived.total_classifications": 0,
        "derived.total_keywords": 0,
        "derived.total_sentiments": 0,
        "derived.total_entities": 0,
    }
    topics = data.get("topics", []) if isinstance(data, dict) else []
    if not isinstance(topics, list):
        return stats

    for topic in topics:
        if not isinstance(topic, dict):
            continue
        for block in _iter_generated_artifact_blocks(topic):
            stats["derived.total_faqs"] += len(_iter_faq_entries_from_block(block))
            stats["derived.total_keywords"] += len(_iter_keyword_entries_from_block(block))
            stats["derived.total_entities"] += len(_iter_entity_entries_from_block(block))
            stats["derived.total_classifications"] += len(_iter_classification_entries_from_block(block))
            stats["derived.total_summaries"] += len(_iter_summary_payloads_from_block(block))

            sentiment_data = block.get("sentiment")
            if isinstance(sentiment_data, dict):
                stats["derived.total_sentiments"] += len(_normalize_sentiment_entries(sentiment_data))

    return stats


def extract_all_text(data):
    """Collect all relevant text from a CRD JSON document for language detection."""
    texts = []
    if not isinstance(data, dict):
        return ""
    topics = data.get("topics", [])
    if not isinstance(topics, list):
        return ""
    for topic in topics:
        if not isinstance(topic, dict):
            continue
        ctx = topic.get("context")
        if ctx:
            texts.append(str(ctx))
        title = topic.get("title")
        if title:
            texts.append(str(title))
        for comp_name, comp_text in extract_summary_components_from_topic(topic):
            if comp_name and comp_text:
                texts.append(str(comp_text))

        texts.extend(extract_faq_texts_from_topic(topic))
        texts.extend(extract_keywords_from_topic(topic))
        texts.extend(extract_entities_from_topic(topic))
        texts.extend(extract_classifications_from_topic(topic))

        for block in _iter_generated_artifact_blocks(topic):
            sentiment_data = block.get("sentiment")
            if isinstance(sentiment_data, dict):
                for entry in _normalize_sentiment_entries(sentiment_data):
                    summary_text = entry.get("summary") or ""
                    if summary_text and str(summary_text).strip():
                        texts.append(str(summary_text))

    return "\n".join(texts)


# ========== Metric Explanations and Info Display Helper ==========

METRIC_EXPLANATIONS = {
    "Summary": {
        "Base Metrics": {
            "Faithfulness": "Measures hallucination rate. Low hallucination = higher score. Range: 1.0-5.0",
            "Coverage": "How well the summary covers context content. Range: 1.0-5.0",
            "Relevance": "Relevance of summary to context. Range: 1.0-5.0",
            "Conciseness": "Score based on compression near target ratio 0.20. Range: 1.0-5.0",
            "Clarity": "Average sentence length score, target ~18 words. Range: 1.0-5.0",
            "Structure": "Score based on sentence count, target 4 sentences. Range: 1.0-5.0",
            "Overall": "Median of six dimensions above. Range: 1.0-5.0",
            "Classification": "Auto-label: Excellent (≥4.5), Acceptable (≥4.0), Mixed, or Needs review",
        },
        "Additional Metrics": {
            "ROUGE-1": "Unigram overlap recall. Higher = better content coverage. Range: 0-1",
            "ROUGE-L": "Longest Common Subsequence overlap. Captures sequence alignment. Range: 0-1",
            "BERTScore": "Semantic similarity via contextual embeddings. Higher = better semantic match. Range: 0-1 or N/A",
            "Context_Precision": "Groundedness of generated text in context: overlap/generated tokens. Range: 0-1",
            "Context_Recall": "Coverage of context by generated text: overlap/context tokens. Range: 0-1",
            "Compression_Ratio": "Summary tokens / context tokens. Typically 0.15-0.35 for summaries",
            "TFIDF_Overlap": "Weighted lexical similarity emphasizing informative terms. Range: 0-1",
            "Answer_Groundedness": "Fraction of sentences supported by context where each generated sentence is matched to its best context sentence/chunk (TF-IDF cosine similarity ≥ 0.6). Higher = fewer hallucinations. Range: 0-1",
            "Hallucination_Rate": "Fraction of sentences NOT supported by context using the same sentence-level support rule. Lower = better. Range: 0-1",
            "Consistency_Score": "Mean pairwise TF-IDF similarity across outputs for the same input (e.g. across system variants). Higher = more stable generation. Range: 0-1",
        }
    },
    "FAQ": {
        "Base Metrics": {
            "Faithfulness": "Hallucination rate in FAQ content. Range: 1.0-5.0",
            "Coverage": "Content coverage across Q+A pairs. Range: 1.0-5.0",
            "Relevance": "FAQ relevance to context. Range: 1.0-5.0",
            "Conciseness": "Compression ratio of FAQ text. Range: 1.0-5.0",
            "Clarity": "Clarity of question/answer phrasing. Range: 1.0-5.0",
            "Structure": "Structure of FAQ format. Range: 1.0-5.0",
            "Overall": "Median of six dimensions. Range: 1.0-5.0",
            "Classification": "Auto-label based on quality thresholds",
            "FAQ_Count": "Number of FAQ entries found",
        },
        "Additional Metrics": {
            "Question_Relevance": "TF-IDF similarity of questions to context. Range: 0-1",
            "Answer_Correctness": "TF-IDF similarity of answers to context. Range: 0-1",
            "Redundancy_Score": "Average pairwise similarity (lower = less repetition). Range: 0-1",
        }
    },
    "Keyword": {
        "Base Metrics": {
            "Faithfulness": "Whether keywords appear in context. Range: 1.0-5.0",
            "Coverage": "Context term coverage by keywords. Range: 1.0-5.0",
            "Relevance": "Keyword relevance to context. Range: 1.0-5.0",
            "Conciseness": "Score based on keyword count efficiency. Range: 1.0-5.0",
            "Clarity": "Clarity/quality of keyword terms. Range: 1.0-5.0",
            "Structure": "Organizational quality of keyword list. Range: 1.0-5.0",
            "Overall": "Median of six dimensions. Range: 1.0-5.0",
            "Classification": "Auto-label based on quality thresholds",
            "Keyword_Count": "Number of keywords found",
        },
        "Additional Metrics": {
            "Precision@K": "Fraction of top-K keywords matching context. Range: 0-1",
            "Recall@K": "Fraction of important context terms in top-K. Range: 0-1",
            "TFIDF_Overlap": "Weighted similarity between keyword set and context. Range: 0-1",
        }
    },
    "Entity": {
        "Base Metrics": {
            "Faithfulness": "Entity hallucination check. Range: 1.0-5.0",
            "Coverage": "Entity coverage of context. Range: 1.0-5.0",
            "Relevance": "Entity relevance to context. Range: 1.0-5.0",
            "Conciseness": "Entity extraction efficiency. Range: 1.0-5.0",
            "Clarity": "Entity clarity and quality. Range: 1.0-5.0",
            "Structure": "Entity annotation structure quality. Range: 1.0-5.0",
            "Overall": "Median of six dimensions. Range: 1.0-5.0",
            "Classification": "Auto-label based on quality thresholds",
            "Entity_Count": "Number of entities found",
        },
        "Additional Metrics": {
            "NER_F1": "F1 score for Named Entity Recognition (requires ground truth). Shows: N/A or 0-1",
            "Entity_Linking_Accuracy": "Accuracy of entity linking to references (requires ground truth). Shows: N/A or 0-1",
        }
    },
    "Classification": {
        "Base Metrics": {
            "Faithfulness": "Classification accuracy vs context. Range: 1.0-5.0",
            "Coverage": "Coverage of context categories. Range: 1.0-5.0",
            "Relevance": "Classification relevance. Range: 1.0-5.0",
            "Conciseness": "Conciseness of classifications. Range: 1.0-5.0",
            "Clarity": "Clarity of class labels. Range: 1.0-5.0",
            "Structure": "Classification structure quality. Range: 1.0-5.0",
            "Overall": "Median of six dimensions. Range: 1.0-5.0",
            "Classification": "Auto-label based on quality thresholds",
            "Classification_Count": "Number of classifications found",
        },
        "Additional Metrics": {
            "Accuracy": "Classification accuracy (requires ground truth labels). Shows: N/A or 0-1",
            "Precision": "Precision per class (requires ground truth). Shows: N/A or 0-1",
            "Recall": "Recall per class (requires ground truth). Shows: N/A or 0-1",
            "F1": "F1 score per class (requires ground truth). Shows: N/A or 0-1",
            "Confusion_Matrix": "Prediction vs actual breakdown (requires ground truth). Shows: N/A or matrix",
        }
    },
    "Sentiment": {
        "Metrics": {
            "Sentiment_Count": "Number of sentiment entries found in generated artifacts",
            "Sentiment_Language": "Language status: 'English' = evaluated, 'Ignored (Language)' = non-English (excluded), 'Partially Ignored' = mixed languages, 'No Sentiment Data' = none found",
            "Sentiment_Score_Validity": "Whether sentiment score is in valid range [0.0, 1.0]. 1.0 = valid, 0.5 = outside range, 0.0 = missing/invalid. Range: 0.0-1.0",
            "Sentiment_Summary_Relevance": "TF-IDF overlap between sentiment summary and context. Measures relevance to source material. Only evaluated for English sentiments. Range: 0.0-1.0",
        }
    }
}


@st.dialog("Metric Definitions")
def _open_metric_info_dialog(attribute_type):
    if attribute_type not in METRIC_EXPLANATIONS:
        st.write("No metric definitions found.")
        return

    explanations = METRIC_EXPLANATIONS[attribute_type]

    if attribute_type == "Sentiment":
        for metric, description in explanations["Metrics"].items():
            st.write(f"**{metric}**: {description}")
    else:
        if "Base Metrics" in explanations:
            st.write("**Base Metrics (1-5 Scale):**")
            for metric, description in explanations["Base Metrics"].items():
                st.write(f"  • **{metric}**: {description}")

        if "Additional Metrics" in explanations:
            st.write("")
            st.write("**Additional Metrics:**")
            for metric, description in explanations["Additional Metrics"].items():
                st.write(f"  • **{metric}**: {description}")


def show_metric_info(attribute_type, widget_key):
    """Render a tiny info button that opens metric definitions in a dialog."""
    if st.button("i", key=f"metric_info_{widget_key}", type="tertiary"):
        _open_metric_info_dialog(attribute_type)


st.title("Attribute Evaluation Framework")
st.write("Evaluate and compare attribute generation metrics across systems.")

# Style tiny info buttons to look like compact circular info icons.
st.markdown(
    """
    <style>
    div[data-testid="stButton"] button[kind="tertiary"] {
        min-height: 24px;
        height: 24px;
        min-width: 24px;
        width: 24px;
        border: 1.5px solid #0EA5E9;
        border-radius: 999px;
        background: #E0F2FE;
        color: #0369A1;
        font-size: 14px;
        font-weight: 900;
        font-style: italic;
        line-height: 1;
        padding: 0;
    }
    div[data-testid="stButton"] button[kind="tertiary"] p {
        font-weight: 900 !important;
        font-style: italic !important;
    }
    div[data-testid="stButton"] button[kind="tertiary"]:hover {
        background: #0EA5E9;
        color: #ffffff;
        border-color: #0368A9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.header("Navigation")
    nav = st.radio(
        "Go to section",
        [
            "Comparison",
            "Quality Evaluation",
            "Single File Evaluation",
            "Dell Configuration Evaluation",
            "Dell_Config_no_GT",
        ],
        index=0,
    )
    keyword_top_k = st.number_input("Keyword Top-K", min_value=1, max_value=50, value=10, step=1)
    st.header("Reporting")
    reporting_mode = st.selectbox(
        "Report View",
        ["Executive", "Balanced", "Detailed"],
        index=1,
        help="Executive: focused summary columns. Balanced: core metrics + key quality signals. Detailed: full raw metric tables.",
    )
    show_diagnostics = st.toggle(
        "Show diagnostics",
        value=False,
        help="Show outlier and diagnostic detail tables.",
    )

REPORTING_MODE = reporting_mode
SHOW_DIAGNOSTICS = show_diagnostics


def _existing_columns(df, cols):
    return [c for c in cols if c in (df.columns if df is not None else [])]


def _curated_columns(attribute_type):
    executive_cols = {
        "Summary": [
            "Page",
            "System",
            "Component",
            "Overall",
            "Classification",
            "Context_Precision",
            "Context_Recall",
            "ROUGE-L",
            "TFIDF_Overlap",
        ],
        "FAQ": ["Page", "System", "FAQ_Count", "Overall", "Classification", "Question_Relevance", "Answer_Correctness"],
        "Keyword": ["Page", "System", "Keyword_Count", "Overall", "Classification", "Precision@K", "Recall@K"],
        "Entity": ["Page", "System", "Entity_Count", "Overall", "Classification"],
        "Classification": ["Page", "System", "Classification_Count", "Overall", "Classification"],
        "Sentiment": ["Page", "System", "Sentiment_Count", "Sentiment_Language", "Sentiment_Score_Validity", "Sentiment_Summary_Relevance"],
    }

    balanced_cols = {
        "Summary": [
            "Page",
            "System",
            "Component",
            "Overall",
            "Classification",
            "Faithfulness",
            "Coverage",
            "Relevance",
            "Conciseness",
            "Clarity",
            "Structure",
            "ROUGE-1",
            "ROUGE-L",
            "Context_Precision",
            "Context_Recall",
            "TFIDF_Overlap",
            "Compression_Ratio",
            "Answer_Groundedness",
            "Hallucination_Rate",
            "Consistency_Score",
        ],
        "FAQ": [
            "Page",
            "System",
            "FAQ_Count",
            "Overall",
            "Classification",
            "Faithfulness",
            "Coverage",
            "Relevance",
            "Question_Relevance",
            "Answer_Correctness",
            "Redundancy_Score",
        ],
        "Keyword": [
            "Page",
            "System",
            "Keyword_Count",
            "Overall",
            "Classification",
            "Faithfulness",
            "Coverage",
            "Relevance",
            "Precision@K",
            "Recall@K",
            "TFIDF_Overlap",
        ],
        "Entity": ["Page", "System", "Entity_Count", "Overall", "Classification", "Faithfulness", "Coverage", "Relevance", "Clarity", "Structure"],
        "Classification": ["Page", "System", "Classification_Count", "Overall", "Classification", "Faithfulness", "Coverage", "Relevance", "Clarity", "Structure"],
        "Sentiment": ["Page", "System", "Sentiment_Count", "Sentiment_Language", "Sentiment_Score_Validity", "Sentiment_Summary_Relevance"],
    }

    if REPORTING_MODE == "Executive":
        return executive_cols.get(attribute_type, [])
    if REPORTING_MODE == "Balanced":
        return balanced_cols.get(attribute_type, [])
    return []


def render_quality_table(title, attribute_type, widget_key, df, outlier_label=None, outlier_group_cols=None):
    if df is None or df.empty:
        return

    header_col, btn_col = st.columns([25, 0.8], vertical_alignment="center", gap="small")
    with header_col:
        st.markdown(title)
    with btn_col:
        show_metric_info(attribute_type, widget_key)

    if REPORTING_MODE == "Detailed":
        st.dataframe(df, use_container_width=True)
    else:
        curated = _existing_columns(df, _curated_columns(attribute_type))
        display_df = df[curated] if curated else df
        st.dataframe(display_df, use_container_width=True)
        with st.expander("Show all detailed metrics"):
            st.dataframe(df, use_container_width=True)

    if SHOW_DIAGNOSTICS and outlier_label and outlier_group_cols:
        render_outlier_report(df, outlier_label, outlier_group_cols)

def pretty_preview(data):
    return json.dumps(data, indent=2)[:1000]  # Limit preview to 1000 chars

def get_top_level_keys(data):
    if isinstance(data, dict):
        return list(data.keys())
    return []

TEMP_DIR = "temp"
WORKSPACE_DIR = os.path.dirname(__file__)
TEMP_PATH = os.path.join(WORKSPACE_DIR, TEMP_DIR)
os.makedirs(TEMP_PATH, exist_ok=True)

def save_uploaded_file(uploaded_file, filename):
    file_path = os.path.join(TEMP_PATH, filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# --- Summary Quality Evaluation Helpers (automatic heuristic reviewer) ---
def _tokenize(text):
    if not text:
        return []
    return [t.lower() for t in re.findall(r"\w+", str(text))]


def _score_0_to_1_from_ratio(ratio, target):
    if target <= 0:
        return 0.0
    penalty = min(1.0, abs(ratio - target) / target)
    return max(0.0, 1.0 - penalty)


def _words_and_sentences(text):
    tokens = _tokenize(text)
    raw = str(text or "")
    # Very simple sentence split
    parts = re.split(r"[.!?]+", raw)
    sentences = [p.strip() for p in parts if p.strip()]
    return tokens, sentences


def evaluate_summary_heuristic(context, summary):
    """Return 1–5 scores for the six dimensions using simple heuristics."""

    ctx_tokens, _ = _words_and_sentences(context)
    sum_tokens, sum_sentences = _words_and_sentences(summary)

    if not ctx_tokens or not sum_tokens:
        # Default neutral scores if we cannot evaluate
        return {
            "Faithfulness": 3.0,
            "Coverage": 3.0,
            "Relevance": 3.0,
            "Conciseness": 3.0,
            "Clarity": 3.0,
            "Structure": 3.0,
        }

    ctx_set = set(ctx_tokens)
    sum_set = set(sum_tokens)

    # Coverage: fraction of summary vocab present in context
    coverage_raw = len(ctx_set & sum_set) / max(1, len(sum_set))

    # Faithfulness: penalize unique summary tokens not in context
    hallucination_rate = len(sum_set - ctx_set) / max(1, len(sum_set))
    faithfulness_raw = max(0.0, 1.0 - hallucination_rate)

    # Relevance: here approximated as coverage as well
    relevance_raw = coverage_raw

    # Conciseness: prefer summary ~20% of context length
    len_ctx = len(ctx_tokens)
    len_sum = len(sum_tokens)
    length_ratio = len_sum / max(1, len_ctx)
    conciseness_raw = _score_0_to_1_from_ratio(length_ratio, target=0.2)

    # Clarity: shorter average sentence length is better up to a point
    if sum_sentences:
        avg_len = len_sum / max(1, len(sum_sentences))
        # Ideal average sentence length around 12–25 words
        clarity_raw = _score_0_to_1_from_ratio(avg_len, target=18)
    else:
        clarity_raw = 0.5

    # Structure: reward having multiple sentences but not excessively many
    n_sent = max(1, len(sum_sentences))
    if n_sent == 1:
        structure_raw = 0.5
    else:
        # Ideal around 3–6 sentences
        structure_raw = _score_0_to_1_from_ratio(n_sent, target=4)

    def to_1_5(x):
        return round(1.0 + 4.0 * max(0.0, min(1.0, x)), 2)

    return {
        "Faithfulness": to_1_5(faithfulness_raw),
        "Coverage": to_1_5(coverage_raw),
        "Relevance": to_1_5(relevance_raw),
        "Conciseness": to_1_5(conciseness_raw),
        "Clarity": to_1_5(clarity_raw),
        "Structure": to_1_5(structure_raw),
    }


BASE_DIMS = ["Faithfulness", "Coverage", "Relevance", "Conciseness", "Clarity", "Structure"]
SUMMARY_EXTRA_COLS = [
    "ROUGE-1",
    "ROUGE-L",
    "BERTScore",
    "Context_Precision",
    "Context_Recall",
    "Compression_Ratio",
    "TFIDF_Overlap",
    "Answer_Groundedness",
    "Hallucination_Rate",
    "Consistency_Score",
]
FAQ_EXTRA_COLS = ["Question_Relevance", "Answer_Correctness", "Redundancy_Score"]
KEYWORD_EXTRA_COLS = ["Precision@K", "Recall@K", "TFIDF_Overlap"]
SENTIMENT_EXTRA_COLS = ["Sentiment_Language", "Sentiment_Score_Validity", "Sentiment_Summary_Relevance"]
ENTITY_EXTRA_COLS = ["NER_F1", "Entity_Linking_Accuracy"]
CLASSIFICATION_EXTRA_COLS = ["Accuracy", "Precision", "Recall", "F1", "Confusion_Matrix"]


def _safe_div(numerator, denominator):
    if not denominator:
        return 0.0
    return numerator / denominator


def _median_or_zero(values, ndigits=2):
    numeric_vals = []
    for v in values or []:
        try:
            numeric_vals.append(float(v))
        except Exception:
            continue
    if not numeric_vals:
        return 0.0
    return round(float(median(numeric_vals)), ndigits)


def _quartiles(sorted_vals):
    n = len(sorted_vals)
    if n < 2:
        return None, None
    mid = n // 2
    if n % 2 == 0:
        lower = sorted_vals[:mid]
        upper = sorted_vals[mid:]
    else:
        lower = sorted_vals[:mid]
        upper = sorted_vals[mid + 1 :]
    if not lower or not upper:
        return None, None
    return median(lower), median(upper)


def detect_iqr_outliers(values, k=1.5):
    numeric_vals = []
    for v in values or []:
        try:
            numeric_vals.append(float(v))
        except Exception:
            continue

    if len(numeric_vals) < 4:
        return {
            "median": _median_or_zero(numeric_vals, ndigits=4),
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower": None,
            "upper": None,
            "outlier_indices": [],
            "outlier_values": [],
        }

    sorted_vals = sorted(numeric_vals)
    q1, q3 = _quartiles(sorted_vals)
    if q1 is None or q3 is None:
        return {
            "median": _median_or_zero(numeric_vals, ndigits=4),
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower": None,
            "upper": None,
            "outlier_indices": [],
            "outlier_values": [],
        }

    iqr = float(q3) - float(q1)
    lower = float(q1) - k * iqr
    upper = float(q3) + k * iqr
    outlier_indices = [i for i, x in enumerate(numeric_vals) if x < lower or x > upper]
    outlier_values = [numeric_vals[i] for i in outlier_indices]

    return {
        "median": round(float(median(numeric_vals)), 4),
        "q1": round(float(q1), 4),
        "q3": round(float(q3), 4),
        "iqr": round(float(iqr), 4),
        "lower": round(float(lower), 4),
        "upper": round(float(upper), 4),
        "outlier_indices": outlier_indices,
        "outlier_values": [round(x, 4) for x in outlier_values],
    }


def build_outlier_report(df, group_cols, metric_cols, min_points=4):
    if df is None or df.empty:
        return pd.DataFrame()

    rows = []
    grouped = df.groupby(group_cols, dropna=False)
    for group_key, group_df in grouped:
        for metric in metric_cols:
            if metric not in group_df.columns:
                continue
            vals = pd.to_numeric(group_df[metric], errors="coerce").dropna().tolist()
            if len(vals) < min_points:
                continue

            stats = detect_iqr_outliers(vals)
            if not stats["outlier_values"]:
                continue

            key_values = group_key if isinstance(group_key, tuple) else (group_key,)
            group_meta = {col: key_values[idx] for idx, col in enumerate(group_cols)}

            rows.append(
                {
                    **group_meta,
                    "Metric": metric,
                    "Sample_Size": len(vals),
                    "Median": stats["median"],
                    "Q1": stats["q1"],
                    "Q3": stats["q3"],
                    "IQR": stats["iqr"],
                    "Lower_Bound": stats["lower"],
                    "Upper_Bound": stats["upper"],
                    "Outlier_Count": len(stats["outlier_values"]),
                    "Outlier_Values": ", ".join(str(v) for v in stats["outlier_values"]),
                }
            )

    return pd.DataFrame(rows)


def render_outlier_report(df, section_label, group_cols):
    metric_cols = BASE_DIMS + ["Overall"]
    outlier_df = build_outlier_report(df, group_cols=group_cols, metric_cols=metric_cols, min_points=4)
    st.markdown(f"##### {section_label} Outlier Report (IQR)")
    if outlier_df.empty:
        st.caption("No outliers detected for numeric quality metrics.")
    else:
        st.dataframe(outlier_df, use_container_width=True)


def _group_quality_scores(df, group_cols, score_col, section_name):
    if df is None or df.empty:
        return pd.DataFrame()
    if score_col not in df.columns:
        return pd.DataFrame()

    tmp = df.copy()
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
    tmp = tmp.dropna(subset=[score_col])
    if tmp.empty:
        return pd.DataFrame()

    grouped = (
        tmp.groupby(group_cols, dropna=False)[score_col]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "Average_Score", "median": "Median_Score", "count": "Sample_Size"})
    )
    grouped["Average_Score"] = grouped["Average_Score"].round(3)
    grouped["Median_Score"] = grouped["Median_Score"].round(3)
    grouped.insert(0, "Section", section_name)
    return grouped


def build_quality_analysis_tables(summary_df=None, faq_df=None, keyword_df=None, entity_df=None, classification_df=None, sentiment_df=None):
    tables = {}

    component_scores = pd.DataFrame()
    component_winners = pd.DataFrame()
    summary_dimension_profile = pd.DataFrame()
    if summary_df is not None and not summary_df.empty and {"System", "Component", "Overall"}.issubset(summary_df.columns):
        component_scores = _group_quality_scores(
            summary_df,
            group_cols=["System", "Component"],
            score_col="Overall",
            section_name="Summary Component",
        )
        if not component_scores.empty:
            comp_rank = component_scores.sort_values(["Component", "Average_Score"], ascending=[True, False])
            component_winners = comp_rank.groupby("Component", as_index=False).first()[["Component", "System", "Average_Score", "Median_Score"]]
            component_winners = component_winners.rename(
                columns={
                    "System": "Best_Setup",
                    "Average_Score": "Best_Average_Score",
                    "Median_Score": "Best_Median_Score",
                }
            )

        usable_dims = [d for d in BASE_DIMS if d in summary_df.columns]
        if usable_dims:
            dim_profile = summary_df.copy()
            for d in usable_dims:
                dim_profile[d] = pd.to_numeric(dim_profile[d], errors="coerce")
            summary_dimension_profile = dim_profile.groupby("System", dropna=False)[usable_dims].mean().round(3).reset_index()

    tables["component_scores"] = component_scores
    tables["component_winners"] = component_winners
    tables["summary_dimension_profile"] = summary_dimension_profile

    section_frames = [
        ("Summary", summary_df, "Overall"),
        ("FAQ", faq_df, "Overall"),
        ("Keyword", keyword_df, "Overall"),
        ("Entity", entity_df, "Overall"),
        ("Classification", classification_df, "Overall"),
    ]

    section_rows = []
    for section_name, frame, score_col in section_frames:
        section_score = _group_quality_scores(
            frame,
            group_cols=["System"],
            score_col=score_col,
            section_name=section_name,
        )
        if not section_score.empty:
            section_rows.append(section_score)

    if sentiment_df is not None and not sentiment_df.empty and "System" in sentiment_df.columns:
        sentiment_cols = [c for c in ["Sentiment_Score_Validity", "Sentiment_Summary_Relevance"] if c in sentiment_df.columns]
        if sentiment_cols:
            sent_tmp = sentiment_df.copy()
            for c in sentiment_cols:
                sent_tmp[c] = pd.to_numeric(sent_tmp[c], errors="coerce")
            sent_tmp["Sentiment_Quality"] = sent_tmp[sentiment_cols].mean(axis=1)
            sentiment_score = _group_quality_scores(
                sent_tmp,
                group_cols=["System"],
                score_col="Sentiment_Quality",
                section_name="Sentiment",
            )
            if not sentiment_score.empty:
                section_rows.append(sentiment_score)

    section_scorecard = pd.concat(section_rows, ignore_index=True) if section_rows else pd.DataFrame()
    tables["section_scorecard"] = section_scorecard

    section_winners = pd.DataFrame()
    setup_ranking = pd.DataFrame()
    setup_strengths = pd.DataFrame()
    if not section_scorecard.empty:
        winner_rows = []
        for section_name, grp in section_scorecard.groupby("Section", dropna=False):
            ordered = grp.sort_values("Average_Score", ascending=False).reset_index(drop=True)
            if ordered.empty:
                continue
            top = ordered.iloc[0]
            runner_up = ordered.iloc[1] if len(ordered) > 1 else None
            winner_rows.append(
                {
                    "Section": section_name,
                    "Best_Setup": top["System"],
                    "Best_Average_Score": round(float(top["Average_Score"]), 3),
                    "Runner_Up": runner_up["System"] if runner_up is not None else "N/A",
                    "Runner_Up_Score": round(float(runner_up["Average_Score"]), 3) if runner_up is not None else None,
                    "Gap": round(float(top["Average_Score"] - runner_up["Average_Score"]), 3)
                    if runner_up is not None
                    else None,
                }
            )
        section_winners = pd.DataFrame(winner_rows)

        setup_ranking = (
            section_scorecard.groupby("System", dropna=False)["Average_Score"]
            .mean()
            .reset_index(name="Overall_Average_Across_Sections")
            .sort_values("Overall_Average_Across_Sections", ascending=False)
        )
        setup_ranking["Overall_Average_Across_Sections"] = setup_ranking["Overall_Average_Across_Sections"].round(3)
        setup_ranking.insert(0, "Rank", range(1, len(setup_ranking) + 1))

        strength_rows = []
        for sys_name, grp in section_scorecard.groupby("System", dropna=False):
            ordered = grp.sort_values("Average_Score", ascending=False).reset_index(drop=True)
            if ordered.empty:
                continue
            strongest = ordered.iloc[0]
            weakest = ordered.iloc[-1]
            strength_rows.append(
                {
                    "System": sys_name,
                    "Strongest_Section": strongest["Section"],
                    "Strongest_Section_Score": round(float(strongest["Average_Score"]), 3),
                    "Needs_Improvement_Section": weakest["Section"],
                    "Needs_Improvement_Score": round(float(weakest["Average_Score"]), 3),
                }
            )
        setup_strengths = pd.DataFrame(strength_rows)

    tables["section_winners"] = section_winners
    tables["setup_ranking"] = setup_ranking
    tables["setup_strengths"] = setup_strengths
    return tables


def render_quality_analysis_section(
    summary_df=None,
    faq_df=None,
    keyword_df=None,
    entity_df=None,
    classification_df=None,
    sentiment_df=None,
    title="Quality Summary and Comparative Analysis",
    show_comparative_sections=True,
):
    analysis_tables = build_quality_analysis_tables(
        summary_df=summary_df,
        faq_df=faq_df,
        keyword_df=keyword_df,
        entity_df=entity_df,
        classification_df=classification_df,
        sentiment_df=sentiment_df,
    )

    has_content = any((df is not None and not df.empty) for df in analysis_tables.values())
    if not has_content:
        return analysis_tables

    st.subheader(title)

    if REPORTING_MODE == "Executive":
        st.caption("Executive view: highlights best setup by artifact quality and overall ranking.")

    if REPORTING_MODE in ["Balanced", "Detailed"]:
        component_scores = analysis_tables.get("component_scores")
        if component_scores is not None and not component_scores.empty:
            st.markdown("#### Final Average Score by Summary Component")
            st.dataframe(component_scores, use_container_width=True)

    if show_comparative_sections:
        component_winners = analysis_tables.get("component_winners")
        if component_winners is not None and not component_winners.empty:
            st.markdown("#### Best Setup per Summary Component")
            st.dataframe(component_winners, use_container_width=True)

    section_scorecard = analysis_tables.get("section_scorecard")
    if section_scorecard is not None and not section_scorecard.empty and REPORTING_MODE != "Executive":
        st.markdown("#### Section-wise Scorecard by Setup")
        st.dataframe(section_scorecard, use_container_width=True)

    if show_comparative_sections:
        section_winners = analysis_tables.get("section_winners")
        if section_winners is not None and not section_winners.empty:
            st.markdown("#### Which Setup is Better per Artifact Quality")
            st.dataframe(section_winners, use_container_width=True)

    if show_comparative_sections:
        setup_ranking = analysis_tables.get("setup_ranking")
        if setup_ranking is not None and not setup_ranking.empty:
            st.markdown("#### Overall Setup Ranking Across Sections")
            st.dataframe(setup_ranking, use_container_width=True)

            top_setup = setup_ranking.iloc[0]
            st.markdown(
                f"**Top overall setup:** {top_setup['System']} with average score {top_setup['Overall_Average_Across_Sections']}."
            )

    if show_comparative_sections:
        setup_strengths = analysis_tables.get("setup_strengths")
        if setup_strengths is not None and not setup_strengths.empty and REPORTING_MODE != "Executive":
            st.markdown("#### Setup Strengths and Improvement Areas")
            st.dataframe(setup_strengths, use_container_width=True)

    summary_dimension_profile = analysis_tables.get("summary_dimension_profile")
    if summary_dimension_profile is not None and not summary_dimension_profile.empty and REPORTING_MODE == "Detailed":
        st.markdown("#### Summary Dimension Profile by Setup")
        st.dataframe(summary_dimension_profile, use_container_width=True)

    return analysis_tables


def _jaccard_similarity(tokens_a, tokens_b):
    a = set(tokens_a)
    b = set(tokens_b)
    if not a and not b:
        return 0.0
    return _safe_div(len(a & b), len(a | b))


def _lcs_length(tokens_a, tokens_b, limit=400):
    a = tokens_a[:limit]
    b = tokens_b[:limit]
    if not a or not b:
        return 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            current = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = current
    return dp[-1]


def _tfidf_overlap(text_a, text_b):
    if not text_a or not text_b:
        return 0.0
    if SKLEARN_AVAILABLE:
        try:
            vec = TfidfVectorizer(stop_words="english")
            mat = vec.fit_transform([str(text_a), str(text_b)])
            row_a = mat[0]
            row_b = mat[1]
            dot = float(row_a.multiply(row_b).sum())
            norm_a = float(sqrt(row_a.multiply(row_a).sum()))
            norm_b = float(sqrt(row_b.multiply(row_b).sum()))
            return round(_safe_div(dot, norm_a * norm_b), 4)
        except Exception:
            pass
    return round(_jaccard_similarity(_tokenize(text_a), _tokenize(text_b)), 4)


def _bertscore_f1(text_a, text_b):
    if not BERTSCORE_AVAILABLE or not text_a or not text_b:
        return "N/A"
    try:
        _, _, f1 = bert_score([str(text_b)], [str(text_a)], lang="en", verbose=False)
        return round(float(f1.mean().item()), 4)
    except Exception:
        return "N/A"


# ---- Groundedness & Consistency helpers ----

GROUNDEDNESS_THRESHOLD = 0.6


def _sentence_split(text):
    """Split text into non-trivial sentences using punctuation boundaries."""
    raw = str(text or "").strip()
    if not raw:
        return []
    parts = re.split(r'(?<=[.!?])\s+', raw)
    return [p.strip() for p in parts if len(p.strip()) > 5]


def _build_context_units(context, chunk_size_tokens=80):
    """Build context units for groundedness matching.

    We evaluate each generated sentence against the best matching context
    sentence/chunk rather than the full context blob to avoid length bias in
    cosine similarity.
    """
    raw = str(context or "").strip()
    if not raw:
        return []

    sent_units = _sentence_split(raw)

    # Add fixed-size token chunks to better handle long or poorly punctuated
    # context blocks.
    tokens = _tokenize(raw)
    chunk_units = []
    if tokens:
        step = max(1, chunk_size_tokens)
        for i in range(0, len(tokens), step):
            chunk = " ".join(tokens[i : i + step]).strip()
            if chunk:
                chunk_units.append(chunk)

    units = sent_units + chunk_units
    # De-duplicate while preserving order.
    deduped = list(dict.fromkeys(units))
    return deduped


def _max_similarity_to_context(sentence, context_units):
    """Return max TF-IDF similarity of sentence against context units."""
    if not sentence or not context_units:
        return 0.0
    sims = [_tfidf_overlap(sentence, unit) for unit in context_units]
    return max(sims) if sims else 0.0


def evaluate_groundedness_metrics(context, generated_text, threshold=GROUNDEDNESS_THRESHOLD):
    """Return Answer_Groundedness and Hallucination_Rate at sentence level.

    A sentence is 'grounded' when its TF-IDF cosine similarity to the
    reference context is >= threshold (default 0.6).
    Hallucination_Rate = 1 - Answer_Groundedness.
    """
    sentences = _sentence_split(generated_text)
    context_units = _build_context_units(context)
    if not sentences or not context_units:
        return {"Answer_Groundedness": 0.0, "Hallucination_Rate": 0.0}
    grounded = sum(1 for s in sentences if _max_similarity_to_context(s, context_units) >= threshold)
    total = len(sentences)
    groundedness = grounded / total
    return {
        "Answer_Groundedness": round(groundedness, 4),
        "Hallucination_Rate": round(1.0 - groundedness, 4),
    }


def compute_consistency_score(texts):
    """Return mean pairwise TF-IDF cosine similarity across a list of texts.

    Measures how stable/consistent outputs are when the same content is
    generated by multiple system variants or across multiple runs.
    """
    valid = [str(t) for t in (texts or []) if t and str(t).strip()]
    if len(valid) < 2:
        return 0.0
    sims = [
        _tfidf_overlap(valid[i], valid[j])
        for i in range(len(valid))
        for j in range(i + 1, len(valid))
    ]
    return round(sum(sims) / len(sims), 4) if sims else 0.0


def evaluate_summary_additional_metrics(context, summary):
    ctx_tokens = _tokenize(context)
    sum_tokens = _tokenize(summary)
    if not ctx_tokens or not sum_tokens:
        return {
            "ROUGE-1": 0.0,
            "ROUGE-L": 0.0,
            "BERTScore": "N/A",
            "Context_Precision": 0.0,
            "Context_Recall": 0.0,
            "Compression_Ratio": 0.0,
            "TFIDF_Overlap": 0.0,
            "Answer_Groundedness": 0.0,
            "Hallucination_Rate": 0.0,
            "Consistency_Score": 0.0,
        }

    ctx_counter = Counter(ctx_tokens)
    sum_counter = Counter(sum_tokens)
    overlap = sum(min(sum_counter[t], ctx_counter[t]) for t in sum_counter)
    context_precision = _safe_div(overlap, len(sum_tokens))
    context_recall = _safe_div(overlap, len(ctx_tokens))
    rouge_1 = _safe_div(overlap, len(ctx_tokens))
    rouge_l = _safe_div(_lcs_length(sum_tokens, ctx_tokens), len(ctx_tokens))
    compression_ratio = _safe_div(len(sum_tokens), len(ctx_tokens))

    return {
        "ROUGE-1": round(rouge_1, 4),
        "ROUGE-L": round(rouge_l, 4),
        "BERTScore": _bertscore_f1(context, summary),
        "Context_Precision": round(context_precision, 4),
        "Context_Recall": round(context_recall, 4),
        "Compression_Ratio": round(compression_ratio, 4),
        "TFIDF_Overlap": _tfidf_overlap(context, summary),
        **evaluate_groundedness_metrics(context, summary),
        "Consistency_Score": 0.0,  # populated by _add_consistency_scores post-processing
    }


def _add_consistency_scores(df, group_cols):
    """Post-process a summary DataFrame to fill the Consistency_Score column.

    Requires a '_summary_text' helper column in *df*.  Groups rows by
    *group_cols*, computes mean pairwise TF-IDF similarity of all texts in
    each group (i.e., across different systems for the same page/component),
    writes that score into Consistency_Score, and drops '_summary_text'.
    """
    if df is None or df.empty or "_summary_text" not in df.columns:
        return df

    valid_cols = [c for c in group_cols if c in df.columns]
    if not valid_cols:
        valid_cols = [df.columns[0]]

    idx_to_score = {}
    for _keys, grp in df.groupby(valid_cols, dropna=False):
        score = compute_consistency_score(grp["_summary_text"].tolist())
        for idx in grp.index:
            idx_to_score[idx] = score

    df = df.copy()
    df["Consistency_Score"] = df.index.map(idx_to_score).fillna(0.0)
    df = df.drop(columns=["_summary_text"])
    return df


def _parse_faq_text(faq_text):
    raw = str(faq_text or "")
    m = re.match(r"\s*Q:\s*(.*?)\s*A:\s*(.*)\s*$", raw, flags=re.DOTALL | re.IGNORECASE)
    if not m:
        return raw, ""
    return m.group(1).strip(), m.group(2).strip()


def evaluate_faq_additional_metrics(faq_texts, context):
    if not faq_texts or not context or not str(context).strip():
        return {
            "Question_Relevance": 0.0,
            "Answer_Correctness": 0.0,
            "Redundancy_Score": 0.0,
        }

    q_scores = []
    a_scores = []
    full_texts = []
    for item in faq_texts:
        q, a = _parse_faq_text(item)
        if q:
            q_scores.append(_tfidf_overlap(context, q))
        if a:
            a_scores.append(_tfidf_overlap(context, a))
        full_texts.append(f"{q} {a}".strip())

    sim_scores = []
    limit = min(len(full_texts), 30)
    for i in range(limit):
        for j in range(i + 1, limit):
            sim_scores.append(_jaccard_similarity(_tokenize(full_texts[i]), _tokenize(full_texts[j])))

    redundancy = _median_or_zero(sim_scores, ndigits=4) if sim_scores else 0.0
    return {
        "Question_Relevance": _median_or_zero(q_scores, ndigits=4) if q_scores else 0.0,
        "Answer_Correctness": _median_or_zero(a_scores, ndigits=4) if a_scores else 0.0,
        "Redundancy_Score": round(redundancy, 4),
    }


def evaluate_keyword_additional_metrics(keywords, context, top_k=10):
    clean_keywords = [str(x).strip().lower() for x in (keywords or []) if str(x).strip()]
    if not clean_keywords or not context or not str(context).strip():
        return {
            "Precision@K": 0.0,
            "Recall@K": 0.0,
            "TFIDF_Overlap": 0.0,
        }

    top_terms = clean_keywords[: max(1, int(top_k))]
    ctx_tokens = _tokenize(context)
    ctx_set = set(ctx_tokens)
    hits = sum(1 for term in top_terms if term in ctx_set)
    precision_at_k = _safe_div(hits, len(top_terms))

    ref_counter = Counter(t for t in ctx_tokens if len(t) > 2)
    ref_terms = {term for term, _ in ref_counter.most_common(max(1, int(top_k)))}
    recall_at_k = _safe_div(len(set(top_terms) & ref_terms), len(ref_terms))

    return {
        "Precision@K": round(precision_at_k, 4),
        "Recall@K": round(recall_at_k, 4),
        "TFIDF_Overlap": _tfidf_overlap(context, " ".join(top_terms)),
    }


def _normalize_sentiment_entries(sentiment_data):
    """Return normalized sentiment entries for both old and new schemas.

    Old schema:
      {"sentiment": "...", "summary": "...", "score": 0.85}

    New schema:
      {"content": [{"sentiment": "...", "summary": "...", "score": 0.85}, ...]}
    """
    if not isinstance(sentiment_data, dict):
        return []

    normalized = []
    content_items = sentiment_data.get("content")
    if isinstance(content_items, list):
        for item in content_items:
            if not isinstance(item, dict):
                continue
            normalized.append(
                {
                    "sentiment": item.get("sentiment"),
                    "summary": item.get("summary"),
                    "score": item.get("score"),
                }
            )

    if normalized:
        return normalized

    # Fallback to legacy flat structure.
    if any(k in sentiment_data for k in ["summary", "score", "sentiment"]):
        return [
            {
                "sentiment": sentiment_data.get("sentiment"),
                "summary": sentiment_data.get("summary"),
                "score": sentiment_data.get("score"),
            }
        ]

    return []


def evaluate_sentiment_metrics(sentiment_data, context):
    """
    Evaluate sentiment quality metrics.
    
    sentiment_data: dict with keys 'sentiment', 'summary', 'score'
    context: reference context text
    
    Returns dict with sentiment metrics and language check.
    """
    entries = _normalize_sentiment_entries(sentiment_data)
    if not entries:
        return {
            "Sentiment_Language": "Ignored (No sentiment data)",
            "Sentiment_Score_Validity": 0.0,
            "Sentiment_Summary_Relevance": 0.0,
        }

    metric_rows = []
    for entry in entries:
        sentiment_summary = entry.get("summary", "")
        sentiment_score = entry.get("score")

        if not sentiment_summary or not str(sentiment_summary).strip():
            metric_rows.append(
                {
                    "Sentiment_Language": "Ignored (Empty summary)",
                    "Sentiment_Score_Validity": 0.0,
                    "Sentiment_Summary_Relevance": 0.0,
                }
            )
            continue

        # Check if sentiment summary is in English
        sentiment_lang_stats = get_language_stats(str(sentiment_summary))
        primary_lang = None
        if sentiment_lang_stats:
            primary_lang = max(sentiment_lang_stats.items(), key=lambda x: x[1])[0]

        # Mark as ignored if not English
        if primary_lang and primary_lang != "en":
            lang_name = get_language_name(primary_lang)
            metric_rows.append(
                {
                    "Sentiment_Language": f"Ignored ({lang_name})",
                    "Sentiment_Score_Validity": 0.0,
                    "Sentiment_Summary_Relevance": 0.0,
                }
            )
            continue

        # Validate sentiment score (should be 0-1)
        score_validity = 0.0
        if isinstance(sentiment_score, (int, float)):
            if 0.0 <= sentiment_score <= 1.0:
                score_validity = 1.0
            else:
                score_validity = 0.5  # Partially valid if outside range

        # Calculate sentiment summary relevance to context
        summary_relevance = 0.0
        if context and str(context).strip():
            summary_relevance = _tfidf_overlap(context, str(sentiment_summary))

        metric_rows.append(
            {
                "Sentiment_Language": "English",
                "Sentiment_Score_Validity": round(score_validity, 4),
                "Sentiment_Summary_Relevance": round(summary_relevance, 4),
            }
        )

    if not metric_rows:
        return {
            "Sentiment_Language": "Ignored (No sentiment data)",
            "Sentiment_Score_Validity": 0.0,
            "Sentiment_Summary_Relevance": 0.0,
        }

    lang_statuses = [m.get("Sentiment_Language") for m in metric_rows]
    english_count = sum(1 for s in lang_statuses if s == "English")
    ignored_count = len(lang_statuses) - english_count

    if ignored_count > 0 and english_count > 0:
        language_status = f"Partially Ignored ({ignored_count} non-English, {english_count} English)"
    elif ignored_count == len(lang_statuses):
        language_status = lang_statuses[0] if lang_statuses else "Ignored"
    else:
        language_status = "English"

    score_validities = [m.get("Sentiment_Score_Validity", 0.0) for m in metric_rows]
    relevances = [m.get("Sentiment_Summary_Relevance", 0.0) for m in metric_rows]

    return {
        "Sentiment_Language": language_status,
        "Sentiment_Score_Validity": _median_or_zero(score_validities, ndigits=4) if score_validities else 0.0,
        "Sentiment_Summary_Relevance": _median_or_zero(relevances, ndigits=4) if relevances else 0.0,
    }


def entity_label_metrics():
    return {
        "NER_F1": "N/A",
        "Entity_Linking_Accuracy": "N/A",
    }


def classification_label_metrics():
    return {
        "Accuracy": "N/A",
        "Precision": "N/A",
        "Recall": "N/A",
        "F1": "N/A",
        "Confusion_Matrix": "N/A",
    }


if nav == "Comparison":
    # --- Single-pair Comparison Logic (dependent selection) ---
    st.subheader("Comparison")

    st.write("### BoozAllen Files")
    boozallen_files = st.file_uploader(
        "Upload BoozAllen JSON files",
        type=["json"],
        key="boozallen_files",
        accept_multiple_files=True,
    )

    selected_ba_file = None
    selected_ba_name = None
    selected_dell_path = None

    # Immediately show which Dell framework files match the uploaded BoozAllen files
    if boozallen_files:
        workspace_dir_for_match = os.path.dirname(__file__)
        dell_folder_for_match = os.path.join(workspace_dir_for_match, "Dell_Setup_CRD_files")
        for f in boozallen_files:
            candidate = os.path.join(dell_folder_for_match, f.name)
            if os.path.exists(candidate):
                st.write(f"Matched Dell file from framework folder: {f.name}")
            else:
                st.write(
                    f"No matching Dell file found in framework folder for: {f.name}. "
                    "Expected location: Dell_Setup_CRD_files/" + f.name
                )

        # Document-level language stats for all uploaded documents (BoozAllen + Dell)
        all_lang_dfs = []
        for f in boozallen_files:
            # BoozAllen document
            try:
                f.seek(0)
                ba_data_all = json.load(f)
                ba_text_all = extract_all_text(ba_data_all)
                ba_stats_all = get_language_stats(ba_text_all)
                ba_df = language_stats_to_df(ba_stats_all)
                if not ba_df.empty:
                    ba_df.insert(0, "System", "BoozAllen")
                    ba_df.insert(0, "Document", f.name)
                    all_lang_dfs.append(ba_df)
            except Exception:
                pass

            # Matching Dell document, if available
            candidate = os.path.join(dell_folder_for_match, f.name)
            if os.path.exists(candidate):
                try:
                    with open(candidate, "r", encoding="utf-8") as dfp:
                        dell_data_all = json.load(dfp)
                    dell_text_all = extract_all_text(dell_data_all)
                    dell_stats_all = get_language_stats(dell_text_all)
                    dell_df = language_stats_to_df(dell_stats_all)
                    if not dell_df.empty:
                        dell_df.insert(0, "System", "Dell")
                        dell_df.insert(0, "Document", f.name)
                        all_lang_dfs.append(dell_df)
                except Exception:
                    pass

        if all_lang_dfs:
            combined_lang_df = pd.concat(all_lang_dfs, ignore_index=True)
            st.subheader("Language Statistics for All Uploaded Documents (Document Level)")
            st.dataframe(combined_lang_df, use_container_width=True)

    # If BoozAllen files are uploaded, allow user to select one and
    # automatically match the Dell file from the Dell_Setup_CRD_files folder.
    if boozallen_files:
        ba_filenames = [f.name for f in boozallen_files]
        st.write("### Select BoozAllen file for comparison")
        selected_ba_name = st.selectbox("Select BoozAllen file", ba_filenames, key="ba_select")
        selected_ba_file = next((f for f in boozallen_files if f.name == selected_ba_name), None)

        dell_folder = os.path.join(WORKSPACE_DIR, "Dell_Setup_CRD_files")
        candidate_path = os.path.join(dell_folder, selected_ba_name)
        if os.path.exists(candidate_path):
            selected_dell_path = candidate_path
            st.success(f"Matched Dell file from framework folder: {selected_ba_name}")
        else:
            selected_dell_path = None
            st.warning(
                f"No matching Dell file found in Dell_Setup_CRD_files for: {selected_ba_name}. "
                "Please ensure a Dell JSON with the same filename exists in that folder."
            )

    if selected_ba_file and selected_dell_path:
        show_only_diff = st.checkbox("Show only differences", value=False)

        # Helper to compute artifact-level counts based on inner items per type
        def compute_artifact_stats(data):
            return _compute_artifact_stats_generic(data)

        # Load and process the selected pair
        fname = selected_ba_file.name
        try:
            selected_ba_file.seek(0)
        except Exception:
            pass
        ba_data = json.load(selected_ba_file)

        # Load Dell data from the auto-matched file path
        with open(selected_dell_path, "r", encoding="utf-8") as f:
            dell_data = json.load(f)

        def get_nested_keys(data, prefix=""):
            keys = []
            if isinstance(data, dict):
                for k, v in data.items():
                    full_key = f"{prefix}.{k}" if prefix else k
                    if isinstance(v, dict):
                        keys.extend(get_nested_keys(v, full_key))
                    else:
                        keys.append(full_key)
            return keys

        ba_keys = set(get_nested_keys(ba_data))
        dell_keys = set(get_nested_keys(dell_data))
        matching_keys = sorted(list(ba_keys & dell_keys))

        def get_nested_value(data, key):
            keys = key.split('.')
            val = data
            for k in keys:
                if isinstance(val, dict):
                    val = val.get(k, None)
                else:
                    return None
            return val

        rows = []
        for key in matching_keys:
            if key == "topics" or key.startswith("topics."):
                continue
            val_ba = get_nested_value(ba_data, key)
            val_dell = get_nested_value(dell_data, key)
            if show_only_diff:
                if val_ba != val_dell:
                    rows.append({"Metric": key, f"BoozAllen ({fname})": val_ba, f"Dell ({fname})": val_dell})
            else:
                rows.append({"Metric": key, f"BoozAllen ({fname})": val_ba, f"Dell ({fname})": val_dell})

        # Derived metric: artifact generation duration (mm:ss) using gen_ai trigger/completion
        artifact_start_key = "meta_data.gen_ai.triggered_at"
        artifact_end_key = "meta_data.gen_ai.completed_at"
        if artifact_start_key in matching_keys and artifact_end_key in matching_keys:
            from datetime import datetime

            def parse_time(val):
                try:
                    return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                except Exception:
                    return None

            ba_start = get_nested_value(ba_data, artifact_start_key)
            ba_end = get_nested_value(ba_data, artifact_end_key)
            dell_start = get_nested_value(dell_data, artifact_start_key)
            dell_end = get_nested_value(dell_data, artifact_end_key)

            def fmt_mmss(start, end):
                t0 = parse_time(start)
                t1 = parse_time(end)
                if not t0 or not t1:
                    return None
                total_sec = max(0, int((t1 - t0).total_seconds()))
                mm, ss = divmod(total_sec, 60)
                return f"{mm:02d}:{ss:02d}"

            ba_dur = fmt_mmss(ba_start, ba_end)
            dell_dur = fmt_mmss(dell_start, dell_end)
            if ba_dur is not None or dell_dur is not None:
                rows.append({
                    "Metric": "derived.artifact_generation_time_mm_ss",
                    f"BoozAllen ({fname})": ba_dur,
                    f"Dell ({fname})": dell_dur,
                })

        # Derived metrics: counts of FAQs, summaries, classifications, keywords, sentiments, entities
        ba_stats = compute_artifact_stats(ba_data)
        dell_stats = compute_artifact_stats(dell_data)
        # --- Language detection and statistics (document level) ---
        ba_text = extract_all_text(ba_data)
        dell_text = extract_all_text(dell_data)
        ba_lang_stats = get_language_stats(ba_text)
        dell_lang_stats = get_language_stats(dell_text)
        for metric_key in [
            "derived.total_faqs",
            "derived.total_summaries",
            "derived.total_classifications",
            "derived.total_keywords",
            "derived.total_sentiments",
            "derived.total_entities",
        ]:
            rows.append({
                "Metric": metric_key,
                f"BoozAllen ({fname})": ba_stats.get(metric_key, 0),
                f"Dell ({fname})": dell_stats.get(metric_key, 0),
            })

        # Sum of all derived artifact counts for consistency check
        ba_total_breakdown = sum(ba_stats.values())
        dell_total_breakdown = sum(dell_stats.values())
        rows.append({
            "Metric": "derived.total_artifacts_from_breakdown",
            f"BoozAllen ({fname})": ba_total_breakdown,
            f"Dell ({fname})": dell_total_breakdown,
        })


        # Build report DataFrame for this pair
        columns = ["Metric", f"BoozAllen ({fname})", f"Dell ({fname})"]
        report_df = pd.DataFrame(rows, columns=columns)

        # --- Display language statistics clearly (tabular) ---
        st.subheader("Language Statistics (Document Level)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**BoozAllen ({fname})**")
            st.dataframe(language_stats_to_df(ba_lang_stats), use_container_width=True)
        with col2:
            st.markdown(f"**Dell ({fname})**")
            st.dataframe(language_stats_to_df(dell_lang_stats), use_container_width=True)

        # Highlight key metrics in the report
        highlight_metrics = {
            # Performance / volume
            "derived.artifact_generation_time_mm_ss",
            "meta_data.genai_total_artifacts_generated",
            "meta_data.total_atoms",
            "derived.total_faqs",
            "derived.total_summaries",
            "derived.total_classifications",
            "derived.total_keywords",
            "derived.total_sentiments",
            "derived.total_entities",
            "derived.total_artifacts_from_breakdown",
            # Model / provider choices
            "meta_data.gen_ai.model",
            "meta_data.gen_ai.model_provider",
            "meta_data.gen_ai.llm_id",
            "meta_data.atomization.model_name",
            "meta_data.atomization.model_provider",
            # Prompt / configuration and status
            "meta_data.atomization.prompt_version",
            "meta_data.atomization_status",
            "meta_data.genai_progress_status",
        }

        def highlight_row(row):
            if row["Metric"] in highlight_metrics:
                return ["background-color: #fff3cd; font-weight: bold;" for _ in row]
            return ["" for _ in row]

        styler = report_df.style.apply(highlight_row, axis=1).set_properties(
            subset=["Metric"], **{"width": "300px"}
        )

        if REPORTING_MODE == "Detailed":
            st.dataframe(styler, use_container_width=True)
        else:
            compact_report_df = report_df[report_df["Metric"].isin(highlight_metrics)].copy()
            if compact_report_df.empty:
                compact_report_df = report_df.head(25)
            st.markdown("#### Key Metric Snapshot")
            st.dataframe(compact_report_df, use_container_width=True)
            with st.expander("Show full metric comparison table"):
                st.dataframe(styler, use_container_width=True)

        # Data quality checks: compare reported vs. derived artifact totals
        st.subheader("Data Quality Checks")
        ba_reported = next(
            (r[columns[1]] for r in rows if r["Metric"] == "meta_data.genai_total_artifacts_generated"),
            None,
        )
        ba_derived = next(
            (r[columns[1]] for r in rows if r["Metric"] == "derived.total_artifacts_from_breakdown"),
            None,
        )
        dell_reported = next(
            (r[columns[2]] for r in rows if r["Metric"] == "meta_data.genai_total_artifacts_generated"),
            None,
        )
        dell_derived = next(
            (r[columns[2]] for r in rows if r["Metric"] == "derived.total_artifacts_from_breakdown"),
            None,
        )

        if isinstance(ba_reported, (int, float)) and isinstance(ba_derived, (int, float)):
            if ba_reported != ba_derived:
                st.warning(
                    f"BoozAllen ({fname}): genai_total_artifacts_generated={ba_reported} differs from "
                    f"derived.total_artifacts_from_breakdown={ba_derived}."
                )
            else:
                st.success(
                    f"BoozAllen ({fname}): Reported artifact total matches derived breakdown ({ba_reported})."
                )

        if isinstance(dell_reported, (int, float)) and isinstance(dell_derived, (int, float)):
            if dell_reported != dell_derived:
                st.warning(
                    f"Dell ({fname}): genai_total_artifacts_generated={dell_reported} differs from "
                    f"derived.total_artifacts_from_breakdown={dell_derived}."
                )
            else:
                st.success(
                    f"Dell ({fname}): Reported artifact total matches derived breakdown ({dell_reported})."
                )
    else:
        st.info(
            "Please upload at least one BoozAllen file so a matching Dell file can be selected automatically from Dell_Setup_CRD_files."
        )


elif nav == "Quality Evaluation":
    st.write("Upload BoozAllen JSON files. Select a BoozAllen file to evaluate; the matching Dell file will be loaded automatically from Dell_Setup_CRD_files.")
    ba_quality_files = st.file_uploader(
        "BoozAllen JSON files for quality evaluation",
        type=["json"],
        key="quality_ba_multi",
        accept_multiple_files=True,
    )

    if ba_quality_files:
        ba_q_filenames = [f.name for f in ba_quality_files]
        selected_ba_q_name = st.selectbox(
            "Select BoozAllen file for quality evaluation",
            ba_q_filenames,
            key="ba_q_select",
        )
        selected_ba_q_file = next((f for f in ba_quality_files if f.name == selected_ba_q_name), None)

        dell_folder_q = os.path.join(WORKSPACE_DIR, "Dell_Setup_CRD_files")
        selected_dell_q_path = os.path.join(dell_folder_q, selected_ba_q_name)
        if os.path.exists(selected_dell_q_path):
            st.success(f"Matched Dell file from framework folder: {selected_ba_q_name}")
        else:
            selected_dell_q_path = None
            st.warning(
                f"No matching Dell file found in Dell_Setup_CRD_files for: {selected_ba_q_name}. "
                "Please ensure a Dell JSON with the same filename exists in that folder."
            )
    else:
        selected_ba_q_file = None
        selected_dell_q_path = None

    def extract_summary(topic):
        return extract_primary_summary_from_topic(topic)

    def extract_summary_components(topic):
        return extract_summary_components_from_topic(topic)

    def extract_faqs(topic):
        return extract_faq_texts_from_topic(topic)

    def extract_keywords(topic):
        return extract_keywords_from_topic(topic)

    def extract_entities(topic):
        return extract_entities_from_topic(topic)

    def extract_classifications(topic):
        return extract_classifications_from_topic(topic)

    def extract_sentiments(topic):
        """Extract all sentiment data from a topic."""
        sentiments = []
        if not isinstance(topic, dict):
            return sentiments
        artifacts = topic.get("generated_artifacts", [])
        if not isinstance(artifacts, list):
            return sentiments
        for block in artifacts:
            if not isinstance(block, dict):
                continue
            sentiment_data = block.get("sentiment")
            if isinstance(sentiment_data, dict):
                normalized_entries = _normalize_sentiment_entries(sentiment_data)
                if normalized_entries:
                    sentiments.extend(normalized_entries)
        return sentiments

    def classify_summary_auto(scores):
        vals = list(scores.values())
        overall = _median_or_zero(vals, ndigits=2) if vals else 0.0
        min_val = min(vals) if vals else 0.0
        if overall >= 4.5 and min_val >= 4:
            label = "Excellent"
        elif overall >= 4.0 and min_val >= 3:
            label = "Acceptable"
        elif overall < 3.0 or min_val <= 2:
            label = "Needs review"
        else:
            label = "Mixed"
        return round(overall, 2), label

    if selected_ba_q_file and selected_dell_q_path:
        try:
            selected_ba_q_file.seek(0)
            ba_q_data = json.load(selected_ba_q_file)
            with open(selected_dell_q_path, "r", encoding="utf-8") as f:
                dell_q_data = json.load(f)
        except Exception as e:
            st.error(f"Failed to read one of the JSON files: {e}")
        else:
            ba_topics = ba_q_data.get("topics", []) if isinstance(ba_q_data, dict) else []
            dell_topics = dell_q_data.get("topics", []) if isinstance(dell_q_data, dict) else []

            if not ba_topics:
                st.warning("No topics found in the BoozAllen JSON.")
            elif not dell_topics:
                st.warning("No topics found in the Dell JSON.")
            else:
                dims = BASE_DIMS

                # Map topics by page number (from title) to align BoozAllen and Dell topics
                def get_page_number_from_title(topic):
                    if not isinstance(topic, dict):
                        return None
                    title = topic.get("title") or ""
                    m = re.search(r"(\d+)", str(title))
                    if not m:
                        return None
                    try:
                        return int(m.group(1))
                    except Exception:
                        return None

                ba_by_page = {}
                for t in ba_topics:
                    pn = get_page_number_from_title(t)
                    if pn is not None:
                        ba_by_page[pn] = t

                dell_by_page = {}
                for t in dell_topics:
                    pn = get_page_number_from_title(t)
                    if pn is not None:
                        dell_by_page[pn] = t

                pages = sorted(set(ba_by_page.keys()) & set(dell_by_page.keys()))

                summary_rows = []
                faq_rows = []
                keyword_rows = []
                entity_rows = []
                classification_rows = []
                sentiment_rows = []

                def score_artifact_list(text_list, context_text, empty_label):
                    if not text_list or not context_text or not str(context_text).strip():
                        return {d: 0.0 for d in dims}, 0.0, empty_label, 0
                    acc = {d: [] for d in dims}
                    for txt in text_list:
                        scores = evaluate_summary_heuristic(context_text, txt)
                        for d in dims:
                            acc[d].append(scores.get(d, 0.0))
                    avg_scores = {}
                    for d in dims:
                        vals = acc[d]
                        avg_scores[d] = _median_or_zero(vals, ndigits=2) if vals else 0.0
                    overall, label = classify_summary_auto(avg_scores)
                    return avg_scores, overall, label, len(text_list)

                for pn in pages:
                    ba_topic = ba_by_page.get(pn, {})
                    dell_topic = dell_by_page.get(pn, {})

                    ba_context = str(ba_topic.get("context") or "") if isinstance(ba_topic, dict) else ""
                    dell_context = str(dell_topic.get("context") or "") if isinstance(dell_topic, dict) else ""

                    # Use title from BoozAllen if available, else Dell, else page number
                    title = (
                        (ba_topic.get("title") if isinstance(ba_topic, dict) else None)
                        or (dell_topic.get("title") if isinstance(dell_topic, dict) else None)
                        or f"Page {pn}"
                    )

                    ba_summary_components = extract_summary_components(ba_topic)
                    dell_summary_components = extract_summary_components(dell_topic)

                    # --- Summary quality (evaluate each component individually) ---
                    # BoozAllen components
                    for comp_name, comp_text in ba_summary_components:
                        if comp_text and ba_context.strip():
                            ba_sum_scores = evaluate_summary_heuristic(ba_context, comp_text)
                            ba_sum_overall, ba_sum_label = classify_summary_auto(ba_sum_scores)
                        else:
                            ba_sum_scores = {d: 0.0 for d in dims}
                            ba_sum_overall, ba_sum_label = 0.0, "No Context or Summary"

                        summary_rows.append({
                            "Page": title,
                            "System": "BoozAllen",
                            "Component": comp_name,
                            **ba_sum_scores,
                            "Overall": ba_sum_overall,
                            "Classification": ba_sum_label,
                            **evaluate_summary_additional_metrics(ba_context, comp_text),
                            "_summary_text": comp_text,
                        })

                    # Dell components
                    for comp_name, comp_text in dell_summary_components:
                        if comp_text and dell_context.strip():
                            dell_sum_scores = evaluate_summary_heuristic(dell_context, comp_text)
                            dell_sum_overall, dell_sum_label = classify_summary_auto(dell_sum_scores)
                        else:
                            dell_sum_scores = {d: 0.0 for d in dims}
                            dell_sum_overall, dell_sum_label = 0.0, "No Context or Summary"

                        summary_rows.append({
                            "Page": title,
                            "System": "Dell",
                            "Component": comp_name,
                            **dell_sum_scores,
                            "Overall": dell_sum_overall,
                            "Classification": dell_sum_label,
                            **evaluate_summary_additional_metrics(dell_context, comp_text),
                            "_summary_text": comp_text,
                        })

                    # --- FAQ quality (per topic/page, averaged over FAQs) ---
                    ba_faq_texts = extract_faqs(ba_topic)
                    dell_faq_texts = extract_faqs(dell_topic)

                    ba_faq_scores, ba_faq_overall, ba_faq_label, ba_faq_count = score_artifact_list(
                        ba_faq_texts, ba_context, "No FAQs or Context"
                    )
                    dell_faq_scores, dell_faq_overall, dell_faq_label, dell_faq_count = score_artifact_list(
                        dell_faq_texts, dell_context, "No FAQs or Context"
                    )

                    faq_rows.append({
                        "Page": title,
                        "System": "BoozAllen",
                        "FAQ_Count": ba_faq_count,
                        **ba_faq_scores,
                        "Overall": ba_faq_overall,
                        "Classification": ba_faq_label,
                        **evaluate_faq_additional_metrics(ba_faq_texts, ba_context),
                    })
                    faq_rows.append({
                        "Page": title,
                        "System": "Dell",
                        "FAQ_Count": dell_faq_count,
                        **dell_faq_scores,
                        "Overall": dell_faq_overall,
                        "Classification": dell_faq_label,
                        **evaluate_faq_additional_metrics(dell_faq_texts, dell_context),
                    })

                    # --- Keywords quality (per topic/page, averaged over keywords) ---
                    ba_kw_texts = extract_keywords(ba_topic)
                    dell_kw_texts = extract_keywords(dell_topic)
                    ba_kw_scores, ba_kw_overall, ba_kw_label, ba_kw_count = score_artifact_list(
                        ba_kw_texts, ba_context, "No Keywords or Context"
                    )
                    dell_kw_scores, dell_kw_overall, dell_kw_label, dell_kw_count = score_artifact_list(
                        dell_kw_texts, dell_context, "No Keywords or Context"
                    )

                    keyword_rows.append({
                        "Page": title,
                        "System": "BoozAllen",
                        "Keyword_Count": ba_kw_count,
                        **ba_kw_scores,
                        "Overall": ba_kw_overall,
                        "Classification": ba_kw_label,
                        **evaluate_keyword_additional_metrics(ba_kw_texts, ba_context, top_k=keyword_top_k),
                    })
                    keyword_rows.append({
                        "Page": title,
                        "System": "Dell",
                        "Keyword_Count": dell_kw_count,
                        **dell_kw_scores,
                        "Overall": dell_kw_overall,
                        "Classification": dell_kw_label,
                        **evaluate_keyword_additional_metrics(dell_kw_texts, dell_context, top_k=keyword_top_k),
                    })

                    # --- Entities quality (per topic/page, averaged over entities) ---
                    ba_ent_texts = extract_entities(ba_topic)
                    dell_ent_texts = extract_entities(dell_topic)
                    ba_ent_scores, ba_ent_overall, ba_ent_label, ba_ent_count = score_artifact_list(
                        ba_ent_texts, ba_context, "No Entities or Context"
                    )
                    dell_ent_scores, dell_ent_overall, dell_ent_label, dell_ent_count = score_artifact_list(
                        dell_ent_texts, dell_context, "No Entities or Context"
                    )

                    entity_rows.append({
                        "Page": title,
                        "System": "BoozAllen",
                        "Entity_Count": ba_ent_count,
                        **ba_ent_scores,
                        "Overall": ba_ent_overall,
                        "Classification": ba_ent_label,
                        **entity_label_metrics(),
                    })
                    entity_rows.append({
                        "Page": title,
                        "System": "Dell",
                        "Entity_Count": dell_ent_count,
                        **dell_ent_scores,
                        "Overall": dell_ent_overall,
                        "Classification": dell_ent_label,
                        **entity_label_metrics(),
                    })

                    # --- Classifications quality (per topic/page, averaged over classifications) ---
                    ba_cls_texts = extract_classifications(ba_topic)
                    dell_cls_texts = extract_classifications(dell_topic)
                    ba_cls_scores, ba_cls_overall, ba_cls_label, ba_cls_count = score_artifact_list(
                        ba_cls_texts, ba_context, "No Classifications or Context"
                    )
                    dell_cls_scores, dell_cls_overall, dell_cls_label, dell_cls_count = score_artifact_list(
                        dell_cls_texts, dell_context, "No Classifications or Context"
                    )

                    classification_rows.append({
                        "Page": title,
                        "System": "BoozAllen",
                        "Classification_Count": ba_cls_count,
                        **ba_cls_scores,
                        "Overall": ba_cls_overall,
                        "Classification": ba_cls_label,
                        **classification_label_metrics(),
                    })
                    classification_rows.append({
                        "Page": title,
                        "System": "Dell",
                        "Classification_Count": dell_cls_count,
                        **dell_cls_scores,
                        "Overall": dell_cls_overall,
                        "Classification": dell_cls_label,
                        **classification_label_metrics(),
                    })

                    # --- Sentiment quality (per topic/page) ---
                    ba_sentiment_list = extract_sentiments(ba_topic)
                    dell_sentiment_list = extract_sentiments(dell_topic)

                    # Evaluate sentiment for BoozAllen
                    ba_sentiment_metrics_list = []
                    for sentiment in ba_sentiment_list:
                        sentiment_metrics = evaluate_sentiment_metrics(sentiment, ba_context)
                        ba_sentiment_metrics_list.append(sentiment_metrics)

                    # Average sentiment metrics for BoozAllen
                    ba_sentiment_metrics = {}
                    if ba_sentiment_metrics_list:
                        # For Sentiment_Language, check if all are English
                        lang_statuses = [m.get("Sentiment_Language") for m in ba_sentiment_metrics_list]
                        english_count = sum(1 for s in lang_statuses if s == "English")
                        ignored_count = len(lang_statuses) - english_count
                        
                        if ignored_count > 0 and english_count > 0:
                            ba_sentiment_metrics["Sentiment_Language"] = f"Partially Ignored ({ignored_count} non-English, {english_count} English)"
                        elif ignored_count == len(lang_statuses):
                            ba_sentiment_metrics["Sentiment_Language"] = lang_statuses[0] if lang_statuses else "Ignored"
                        else:
                            ba_sentiment_metrics["Sentiment_Language"] = "English"
                        
                        # Median numeric metrics
                        score_validities = [m.get("Sentiment_Score_Validity", 0.0) for m in ba_sentiment_metrics_list]
                        relevances = [m.get("Sentiment_Summary_Relevance", 0.0) for m in ba_sentiment_metrics_list]
                        
                        ba_sentiment_metrics["Sentiment_Score_Validity"] = _median_or_zero(score_validities, ndigits=4) if score_validities else 0.0
                        ba_sentiment_metrics["Sentiment_Summary_Relevance"] = _median_or_zero(relevances, ndigits=4) if relevances else 0.0
                    else:
                        ba_sentiment_metrics = {
                            "Sentiment_Language": "No Sentiment Data",
                            "Sentiment_Score_Validity": 0.0,
                            "Sentiment_Summary_Relevance": 0.0,
                        }

                    # Evaluate sentiment for Dell
                    dell_sentiment_metrics_list = []
                    for sentiment in dell_sentiment_list:
                        sentiment_metrics = evaluate_sentiment_metrics(sentiment, dell_context)
                        dell_sentiment_metrics_list.append(sentiment_metrics)

                    # Average sentiment metrics for Dell
                    dell_sentiment_metrics = {}
                    if dell_sentiment_metrics_list:
                        # For Sentiment_Language, check if all are English
                        lang_statuses = [m.get("Sentiment_Language") for m in dell_sentiment_metrics_list]
                        english_count = sum(1 for s in lang_statuses if s == "English")
                        ignored_count = len(lang_statuses) - english_count
                        
                        if ignored_count > 0 and english_count > 0:
                            dell_sentiment_metrics["Sentiment_Language"] = f"Partially Ignored ({ignored_count} non-English, {english_count} English)"
                        elif ignored_count == len(lang_statuses):
                            dell_sentiment_metrics["Sentiment_Language"] = lang_statuses[0] if lang_statuses else "Ignored"
                        else:
                            dell_sentiment_metrics["Sentiment_Language"] = "English"
                        
                        # Median numeric metrics
                        score_validities = [m.get("Sentiment_Score_Validity", 0.0) for m in dell_sentiment_metrics_list]
                        relevances = [m.get("Sentiment_Summary_Relevance", 0.0) for m in dell_sentiment_metrics_list]
                        
                        dell_sentiment_metrics["Sentiment_Score_Validity"] = _median_or_zero(score_validities, ndigits=4) if score_validities else 0.0
                        dell_sentiment_metrics["Sentiment_Summary_Relevance"] = _median_or_zero(relevances, ndigits=4) if relevances else 0.0
                    else:
                        dell_sentiment_metrics = {
                            "Sentiment_Language": "No Sentiment Data",
                            "Sentiment_Score_Validity": 0.0,
                            "Sentiment_Summary_Relevance": 0.0,
                        }

                    sentiment_rows.append({
                        "Page": title,
                        "System": "BoozAllen",
                        "Sentiment_Count": len(ba_sentiment_list),
                        **ba_sentiment_metrics,
                    })
                    sentiment_rows.append({
                        "Page": title,
                        "System": "Dell",
                        "Sentiment_Count": len(dell_sentiment_list),
                        **dell_sentiment_metrics,
                    })

                if summary_rows:
                    summary_df = _add_consistency_scores(
                        pd.DataFrame(summary_rows), ["Page", "Component"]
                    )
                    render_quality_table(
                        "#### Per-topic Summary Quality (automatic, 1–5 scale)",
                        "Summary",
                        "quality_summary",
                        summary_df,
                        outlier_label="Summary",
                        outlier_group_cols=["System", "Component"],
                    )
                else:
                    st.warning("No evaluable topics with both context and summaries were found.")

                if faq_rows:
                    faq_df = pd.DataFrame(faq_rows)
                    render_quality_table(
                        "#### Per-topic FAQ Quality (automatic, 1–5 scale, median per FAQ)",
                        "FAQ",
                        "quality_faq",
                        faq_df,
                        outlier_label="FAQ",
                        outlier_group_cols=["System"],
                    )

                if keyword_rows:
                    kw_df = pd.DataFrame(keyword_rows)
                    render_quality_table(
                        "#### Per-topic Keyword Quality (automatic, 1–5 scale, median per keyword)",
                        "Keyword",
                        "quality_keyword",
                        kw_df,
                        outlier_label="Keyword",
                        outlier_group_cols=["System"],
                    )

                if entity_rows:
                    ent_df = pd.DataFrame(entity_rows)
                    render_quality_table(
                        "#### Per-topic Entity Quality (automatic, 1–5 scale, median per entity)",
                        "Entity",
                        "quality_entity",
                        ent_df,
                        outlier_label="Entity",
                        outlier_group_cols=["System"],
                    )

                if classification_rows:
                    cls_df = pd.DataFrame(classification_rows)
                    render_quality_table(
                        "#### Per-topic Classification Quality (automatic, 1–5 scale, median per classification)",
                        "Classification",
                        "quality_classification",
                        cls_df,
                        outlier_label="Classification",
                        outlier_group_cols=["System"],
                    )

                if sentiment_rows:
                    sentiment_df = pd.DataFrame(sentiment_rows)
                    render_quality_table(
                        "#### Per-topic Sentiment Quality (language check and relevance metrics)",
                        "Sentiment",
                        "quality_sentiment",
                        sentiment_df,
                    )

                analysis_tables_quality = render_quality_analysis_section(
                    summary_df=summary_df if summary_rows else None,
                    faq_df=faq_df if faq_rows else None,
                    keyword_df=kw_df if keyword_rows else None,
                    entity_df=ent_df if entity_rows else None,
                    classification_df=cls_df if classification_rows else None,
                    sentiment_df=sentiment_df if sentiment_rows else None,
                    title="Quality Summary and Comparative Analysis",
                )

elif nav == "Single File Evaluation":
    st.subheader("Single File Evaluation")
    st.write(
        "Upload one JSON file for immediate evaluation, or bulk upload multiple JSON files "
        "to generate a consolidated downloadable metrics report."
    )

    single_eval_file = st.file_uploader(
        "Upload a single JSON file",
        type=["json"],
        key="single_eval_file",
        accept_multiple_files=False,
    )

    bulk_eval_files = st.file_uploader(
        "Bulk upload JSON files (for report generation)",
        type=["json"],
        key="single_bulk_eval_files",
        accept_multiple_files=True,
    )

    def extract_summary_components_single(topic):
        return extract_summary_components_from_topic(topic)

    def extract_faqs_single(topic):
        return extract_faq_texts_from_topic(topic)

    def extract_keywords_single(topic):
        return extract_keywords_from_topic(topic)

    def extract_entities_single(topic):
        return extract_entities_from_topic(topic)

    def extract_classifications_single(topic):
        return extract_classifications_from_topic(topic)

    def extract_sentiments_single(topic):
        sentiments = []
        if not isinstance(topic, dict):
            return sentiments
        artifacts = topic.get("generated_artifacts", [])
        if not isinstance(artifacts, list):
            return sentiments
        for block in artifacts:
            if not isinstance(block, dict):
                continue
            sentiment_data = block.get("sentiment")
            if isinstance(sentiment_data, dict):
                normalized_entries = _normalize_sentiment_entries(sentiment_data)
                if normalized_entries:
                    sentiments.extend(normalized_entries)
        return sentiments

    def classify_summary_auto_single(scores):
        vals = list(scores.values())
        overall = _median_or_zero(vals, ndigits=2) if vals else 0.0
        min_val = min(vals) if vals else 0.0
        if overall >= 4.5 and min_val >= 4:
            label = "Excellent"
        elif overall >= 4.0 and min_val >= 3:
            label = "Acceptable"
        elif overall < 3.0 or min_val <= 2:
            label = "Needs review"
        else:
            label = "Mixed"
        return round(overall, 2), label

    def evaluate_document_single_mode(doc_data, system_label="Uploaded"):
        topics = doc_data.get("topics", []) if isinstance(doc_data, dict) else []
        if not topics:
            return {
                "error": "No topics found in the uploaded JSON.",
                "summary_df": None,
                "faq_df": None,
                "kw_df": None,
                "ent_df": None,
                "cls_df": None,
                "sentiment_df": None,
                "analysis_tables": None,
            }

        dims = BASE_DIMS

        summary_rows = []
        faq_rows = []
        keyword_rows = []
        entity_rows = []
        classification_rows = []
        sentiment_rows = []

        def score_artifact_list_single(text_list, context_text, empty_label):
            if not text_list or not context_text or not str(context_text).strip():
                return {d: 0.0 for d in dims}, 0.0, empty_label, 0
            acc = {d: [] for d in dims}
            for txt in text_list:
                scores = evaluate_summary_heuristic(context_text, txt)
                for d in dims:
                    acc[d].append(scores.get(d, 0.0))
            avg_scores = {}
            for d in dims:
                vals = acc[d]
                avg_scores[d] = _median_or_zero(vals, ndigits=2) if vals else 0.0
            overall, label = classify_summary_auto_single(avg_scores)
            return avg_scores, overall, label, len(text_list)

        for idx, topic in enumerate(topics, start=1):
            if not isinstance(topic, dict):
                continue

            context = str(topic.get("context") or "")
            title = topic.get("title") or f"Topic {idx}"

            summary_components = extract_summary_components_single(topic)

            for comp_name, comp_text in summary_components:
                if comp_text and context.strip():
                    sum_scores = evaluate_summary_heuristic(context, comp_text)
                    sum_overall, sum_label = classify_summary_auto_single(sum_scores)
                else:
                    sum_scores = {d: 0.0 for d in dims}
                    sum_overall, sum_label = 0.0, "No Context or Summary"

                summary_rows.append({
                    "Page": title,
                    "System": system_label,
                    "Component": comp_name,
                    **sum_scores,
                    "Overall": sum_overall,
                    "Classification": sum_label,
                    **evaluate_summary_additional_metrics(context, comp_text),
                    "_summary_text": comp_text,
                })

            faq_texts = extract_faqs_single(topic)
            faq_scores, faq_overall, faq_label, faq_count = score_artifact_list_single(
                faq_texts, context, "No FAQs or Context"
            )
            faq_rows.append({
                "Page": title,
                "System": system_label,
                "FAQ_Count": faq_count,
                **faq_scores,
                "Overall": faq_overall,
                "Classification": faq_label,
                **evaluate_faq_additional_metrics(faq_texts, context),
            })

            kw_texts = extract_keywords_single(topic)
            kw_scores, kw_overall, kw_label, kw_count = score_artifact_list_single(
                kw_texts, context, "No Keywords or Context"
            )
            keyword_rows.append({
                "Page": title,
                "System": system_label,
                "Keyword_Count": kw_count,
                **kw_scores,
                "Overall": kw_overall,
                "Classification": kw_label,
                **evaluate_keyword_additional_metrics(kw_texts, context, top_k=keyword_top_k),
            })

            ent_texts = extract_entities_single(topic)
            ent_scores, ent_overall, ent_label, ent_count = score_artifact_list_single(
                ent_texts, context, "No Entities or Context"
            )
            entity_rows.append({
                "Page": title,
                "System": system_label,
                "Entity_Count": ent_count,
                **ent_scores,
                "Overall": ent_overall,
                "Classification": ent_label,
                **entity_label_metrics(),
            })

            cls_texts = extract_classifications_single(topic)
            cls_scores, cls_overall, cls_label, cls_count = score_artifact_list_single(
                cls_texts, context, "No Classifications or Context"
            )
            classification_rows.append({
                "Page": title,
                "System": system_label,
                "Classification_Count": cls_count,
                **cls_scores,
                "Overall": cls_overall,
                "Classification": cls_label,
                **classification_label_metrics(),
            })

            sentiment_list = extract_sentiments_single(topic)
            sentiment_metrics_list = []
            for sentiment in sentiment_list:
                sentiment_metrics_list.append(evaluate_sentiment_metrics(sentiment, context))

            if sentiment_metrics_list:
                lang_statuses = [m.get("Sentiment_Language") for m in sentiment_metrics_list]
                english_count = sum(1 for s in lang_statuses if s == "English")
                ignored_count = len(lang_statuses) - english_count

                if ignored_count > 0 and english_count > 0:
                    sent_lang = f"Partially Ignored ({ignored_count} non-English, {english_count} English)"
                elif ignored_count == len(lang_statuses):
                    sent_lang = lang_statuses[0] if lang_statuses else "Ignored"
                else:
                    sent_lang = "English"

                sent_score_validity = _median_or_zero(
                    [m.get("Sentiment_Score_Validity", 0.0) for m in sentiment_metrics_list],
                    ndigits=4,
                )
                sent_relevance = _median_or_zero(
                    [m.get("Sentiment_Summary_Relevance", 0.0) for m in sentiment_metrics_list],
                    ndigits=4,
                )
            else:
                sent_lang = "No Sentiment Data"
                sent_score_validity = 0.0
                sent_relevance = 0.0

            sentiment_rows.append({
                "Page": title,
                "System": system_label,
                "Sentiment_Count": len(sentiment_list),
                "Sentiment_Language": sent_lang,
                "Sentiment_Score_Validity": sent_score_validity,
                "Sentiment_Summary_Relevance": sent_relevance,
            })

        summary_df = _add_consistency_scores(pd.DataFrame(summary_rows), ["Page", "Component"]) if summary_rows else None
        faq_df = pd.DataFrame(faq_rows) if faq_rows else None
        kw_df = pd.DataFrame(keyword_rows) if keyword_rows else None
        ent_df = pd.DataFrame(entity_rows) if entity_rows else None
        cls_df = pd.DataFrame(classification_rows) if classification_rows else None
        sentiment_df = pd.DataFrame(sentiment_rows) if sentiment_rows else None

        analysis_tables = build_quality_analysis_tables(
            summary_df=summary_df,
            faq_df=faq_df,
            keyword_df=kw_df,
            entity_df=ent_df,
            classification_df=cls_df,
            sentiment_df=sentiment_df,
        )

        return {
            "error": None,
            "summary_df": summary_df,
            "faq_df": faq_df,
            "kw_df": kw_df,
            "ent_df": ent_df,
            "cls_df": cls_df,
            "sentiment_df": sentiment_df,
            "analysis_tables": analysis_tables,
        }

    if single_eval_file is not None:
        try:
            single_eval_file.seek(0)
            single_data = json.load(single_eval_file)
        except Exception as e:
            st.error(f"Failed to read JSON file: {e}")
        else:
            eval_result = evaluate_document_single_mode(single_data, system_label="Uploaded")
            if eval_result["error"]:
                st.warning(eval_result["error"])
            else:
                summary_df = eval_result["summary_df"]
                faq_df = eval_result["faq_df"]
                kw_df = eval_result["kw_df"]
                ent_df = eval_result["ent_df"]
                cls_df = eval_result["cls_df"]
                sentiment_df = eval_result["sentiment_df"]

                if summary_df is not None and not summary_df.empty:
                    render_quality_table(
                        "#### Per-topic Summary Quality (automatic, 1-5 scale)",
                        "Summary",
                        "single_quality_summary",
                        summary_df,
                        outlier_label="Summary",
                        outlier_group_cols=["Component"],
                    )
                else:
                    st.warning("No evaluable summaries were found.")

                if faq_df is not None and not faq_df.empty:
                    render_quality_table(
                        "#### Per-topic FAQ Quality (automatic, 1-5 scale, median per FAQ)",
                        "FAQ",
                        "single_quality_faq",
                        faq_df,
                        outlier_label="FAQ",
                        outlier_group_cols=["System"],
                    )

                if kw_df is not None and not kw_df.empty:
                    render_quality_table(
                        "#### Per-topic Keyword Quality (automatic, 1-5 scale, median per keyword)",
                        "Keyword",
                        "single_quality_keyword",
                        kw_df,
                        outlier_label="Keyword",
                        outlier_group_cols=["System"],
                    )

                if ent_df is not None and not ent_df.empty:
                    render_quality_table(
                        "#### Per-topic Entity Quality (automatic, 1-5 scale, median per entity)",
                        "Entity",
                        "single_quality_entity",
                        ent_df,
                        outlier_label="Entity",
                        outlier_group_cols=["System"],
                    )

                if cls_df is not None and not cls_df.empty:
                    render_quality_table(
                        "#### Per-topic Classification Quality (automatic, 1-5 scale, median per classification)",
                        "Classification",
                        "single_quality_classification",
                        cls_df,
                        outlier_label="Classification",
                        outlier_group_cols=["System"],
                    )

                if sentiment_df is not None and not sentiment_df.empty:
                    render_quality_table(
                        "#### Per-topic Sentiment Quality (language check and relevance metrics)",
                        "Sentiment",
                        "single_quality_sentiment",
                        sentiment_df,
                    )

                render_quality_analysis_section(
                    summary_df=summary_df,
                    faq_df=faq_df,
                    keyword_df=kw_df,
                    entity_df=ent_df,
                    classification_df=cls_df,
                    sentiment_df=sentiment_df,
                    title="Quality Summary Analysis",
                    show_comparative_sections=False,
                )

    if bulk_eval_files:
        st.caption(f"{len(bulk_eval_files)} file(s) ready for bulk evaluation.")

    if st.button("Bulk Evaluate and Generate Report", key="single_bulk_evaluate_btn"):
        if not bulk_eval_files:
            st.error("Please upload at least one JSON file for bulk evaluation.")
        else:
            processing_rows = []

            summary_all = []
            faq_all = []
            keyword_all = []
            entity_all = []
            classification_all = []
            sentiment_all = []

            analysis_component_scores_all = []
            analysis_section_scorecard_all = []
            analysis_setup_ranking_all = []

            success_count = 0

            for uploaded in bulk_eval_files:
                try:
                    uploaded.seek(0)
                    data = json.load(uploaded)
                except Exception as e:
                    processing_rows.append(
                        {
                            "File": uploaded.name,
                            "Status": "Failed",
                            "Topics": 0,
                            "Error": str(e),
                        }
                    )
                    continue

                eval_result = evaluate_document_single_mode(data, system_label="Uploaded")
                topics_count = len(data.get("topics", [])) if isinstance(data, dict) and isinstance(data.get("topics", []), list) else 0

                if eval_result["error"]:
                    processing_rows.append(
                        {
                            "File": uploaded.name,
                            "Status": "Failed",
                            "Topics": topics_count,
                            "Error": eval_result["error"],
                        }
                    )
                    continue

                success_count += 1
                processing_rows.append(
                    {
                        "File": uploaded.name,
                        "Status": "Success",
                        "Topics": topics_count,
                        "Error": "",
                    }
                )

                def _append_with_file(container, df):
                    if df is not None and not df.empty:
                        tmp = df.copy()
                        tmp.insert(0, "File", uploaded.name)
                        container.append(tmp)

                _append_with_file(summary_all, eval_result["summary_df"])
                _append_with_file(faq_all, eval_result["faq_df"])
                _append_with_file(keyword_all, eval_result["kw_df"])
                _append_with_file(entity_all, eval_result["ent_df"])
                _append_with_file(classification_all, eval_result["cls_df"])
                _append_with_file(sentiment_all, eval_result["sentiment_df"])

                analysis_tables = eval_result.get("analysis_tables") or {}
                _append_with_file(analysis_component_scores_all, analysis_tables.get("component_scores"))
                _append_with_file(analysis_section_scorecard_all, analysis_tables.get("section_scorecard"))
                _append_with_file(analysis_setup_ranking_all, analysis_tables.get("setup_ranking"))

            processing_df = pd.DataFrame(processing_rows, columns=["File", "Status", "Topics", "Error"])
            st.subheader("Bulk Evaluation Summary")
            st.dataframe(processing_df, use_container_width=True)

            if success_count == 0:
                st.error("No files were successfully evaluated. Fix the listed errors and try again.")
            else:
                st.success(f"Bulk evaluation completed. Successfully evaluated {success_count} file(s).")

                sheets = {"Processing_Summary": processing_df}

                def _concat_or_empty(frames):
                    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

                summary_df_all = _concat_or_empty(summary_all)
                faq_df_all = _concat_or_empty(faq_all)
                keyword_df_all = _concat_or_empty(keyword_all)
                entity_df_all = _concat_or_empty(entity_all)
                classification_df_all = _concat_or_empty(classification_all)
                sentiment_df_all = _concat_or_empty(sentiment_all)

                analysis_component_scores_df_all = _concat_or_empty(analysis_component_scores_all)
                analysis_section_scorecard_df_all = _concat_or_empty(analysis_section_scorecard_all)
                analysis_setup_ranking_df_all = _concat_or_empty(analysis_setup_ranking_all)

                if not summary_df_all.empty:
                    sheets["Summary_Quality_All"] = summary_df_all
                if not faq_df_all.empty:
                    sheets["FAQ_Quality_All"] = faq_df_all
                if not keyword_df_all.empty:
                    sheets["Keyword_Quality_All"] = keyword_df_all
                if not entity_df_all.empty:
                    sheets["Entity_Quality_All"] = entity_df_all
                if not classification_df_all.empty:
                    sheets["Classification_Quality_All"] = classification_df_all
                if not sentiment_df_all.empty:
                    sheets["Sentiment_Quality_All"] = sentiment_df_all

                if not analysis_component_scores_df_all.empty:
                    sheets["Analysis_ComponentScores"] = analysis_component_scores_df_all
                if not analysis_section_scorecard_df_all.empty:
                    sheets["Analysis_SectionScorecard"] = analysis_section_scorecard_df_all
                if not analysis_setup_ranking_df_all.empty:
                    sheets["Analysis_SetupRanking"] = analysis_setup_ranking_df_all

                try:
                    buffer = BytesIO()
                    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                        for sheet_name, df in sheets.items():
                            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                    buffer.seek(0)

                    st.download_button(
                        label="Download bulk evaluation report (.xlsx)",
                        data=buffer.getvalue(),
                        file_name="single_tab_bulk_evaluation_report.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
                except Exception as e:
                    st.error(f"Failed to generate bulk report: {e}")

elif nav == "Dell Configuration Evaluation":
    st.subheader("Dell Configuration Evaluation")

    st.write("Upload one or more ground truth Markdown files and one or more CRD JSON files. Then select a pair to evaluate.")

    # Ground truth Markdown uploader – now supports multiple files
    gt_md_files = st.file_uploader(
        "Ground Truth File(s) (.md)",
        type=["md"],
        key="dell_gt_md_multi",
        accept_multiple_files=True,
    )

    gt_context = None
    gt_md_name = None
    if gt_md_files:
        md_names = [f.name for f in gt_md_files]
        gt_md_name = st.selectbox(
            "Select Ground Truth file",
            md_names,
            key="dell_gt_md_select",
        )
        selected_gt_file = next((f for f in gt_md_files if f.name == gt_md_name), None)
        if selected_gt_file is not None:
            try:
                selected_gt_file.seek(0)
                gt_bytes = selected_gt_file.read()
                gt_context = gt_bytes.decode("utf-8", errors="ignore")
                st.success(f"Ground truth context loaded from: {gt_md_name}")
            except Exception as e:
                st.error(f"Failed to read ground truth file: {e}")

    # CRD JSON uploader – now supports multiple files (1x1 configs)
    crd_files = st.file_uploader(
        "CRD File(s) (.json) from a Dell_Configurations subfolder (e.g. test_set_1x1_at_crd_com or S2_BoxOfChocolates_1x1_at_crd_com)",
        type=["json"],
        key="dell_crd_json_multi",
        accept_multiple_files=True,
    )

    set_paths = {}
    crd_name = None
    selected_crd_file = None
    if crd_files:
        crd_names = [f.name for f in crd_files]
        crd_name = st.selectbox(
            "Select CRD file (treated as 1x1 variant)",
            crd_names,
            key="dell_crd_select",
        )
        selected_crd_file = next((f for f in crd_files if f.name == crd_name), None)

    # Root folder for Dell configuration CRDs
    dell_config_root = os.path.join(WORKSPACE_DIR, "Dell_Configurations")

    if selected_crd_file is not None:
        st.success(f"CRD file selected (treated as 1x1 variant): {crd_name}")

        # Look for the uploaded file name under Dell_Configurations to
        # infer the parent folder (e.g. test_set_1x1_at_crd_com or
        # S2_BoxOfChocolates_1x1_at_crd_com). Then search for the other
        # variants in sibling folders with the same prefix
        # (e.g. test_set_* or S2_BoxOfChocolates_*).

        parent_folder_name = None
        parent_candidate_path = None
        if os.path.isdir(dell_config_root):
            for entry in os.scandir(dell_config_root):
                if not entry.is_dir():
                    continue
                candidate = os.path.join(entry.path, crd_name)
                if os.path.exists(candidate):
                    parent_folder_name = os.path.basename(entry.path)
                    parent_candidate_path = candidate
                    break

        set_paths = {}

        # Map dimension to internal label names used throughout the UI
        dim_to_label = {
            "1x1": "test_set_1x1",
            "2x2": "test_set_2x2",
            "4x4": "test_set_4x4",
        }

        if parent_folder_name:
            # Capture the base prefix before the dimension token so that
            # we can find matching folders such as test_set_* or
            # S2_BoxOfChocolates_*.
            m = re.search(r"^(.*?)(1x1|2x2|4x4)", parent_folder_name)
            if m:
                base_prefix, first_dim = m.group(1), m.group(2)

                # Always record the on-disk path for the discovered folder
                if first_dim in dim_to_label and parent_candidate_path:
                    set_paths[dim_to_label[first_dim]] = parent_candidate_path

                # Now scan for sibling folders with the same prefix and
                # other dimension tokens (2x2 / 4x4).
                for entry in os.scandir(dell_config_root):
                    if not entry.is_dir():
                        continue
                    folder_name = entry.name
                    m2 = re.search(r"^(.*?)(1x1|2x2|4x4)", folder_name)
                    if not m2:
                        continue
                    prefix, dim = m2.group(1), m2.group(2)
                    if prefix != base_prefix:
                        continue

                    label = dim_to_label.get(dim)
                    if not label:
                        continue

                    candidate = os.path.join(entry.path, crd_name)
                    if os.path.exists(candidate):
                        set_paths[label] = candidate
        else:
            # Fallback to the original fixed test_set_* folders if we
            # cannot infer the parent folder automatically.
            fixed_folders = [
                ("test_set_1x1", "test_set_1x1_at_crd_com"),
                ("test_set_2x2", "test_set_2x2_at_crd_com"),
                ("test_set_4x4", "test_set_4x4_at_crd_com"),
            ]

            if os.path.isdir(dell_config_root):
                for label, folder in fixed_folders:
                    candidate = os.path.join(dell_config_root, folder, crd_name)
                    if os.path.exists(candidate):
                        set_paths[label] = candidate

        if set_paths:
            st.write("Matching CRD files found in Dell_Configurations:")
            for label, path in set_paths.items():
                rel = os.path.relpath(path, WORKSPACE_DIR)
                folder_name = os.path.basename(os.path.dirname(path))
                # Show the parent folder name (e.g. S2_BoxOfChocolates_1x1_at_crd_com)
                # rather than the internal key (test_set_1x1, etc.).
                st.write(f"- {folder_name}: {rel}")
        else:
            st.warning(
                "No matching CRD files with this name were found under Dell_Configurations/test_set_* folders."
            )

        # --- Evaluation: compare and quality-score all three CRD variants (single pair) ---
        if st.button("Evaluate", key="dell_config_evaluate"):
            if not gt_context or not str(gt_context or "").strip():
                st.error("Please upload and select a ground truth (.md) file before evaluating.")
            else:
                # Use the uploaded file as test_set_1x1 and
                # pick corresponding files from test_set_2x2_at_crd_com and test_set_4x4_at_crd_com.
                path_2x2 = set_paths.get("test_set_2x2")
                path_4x4 = set_paths.get("test_set_4x4")

                if not path_2x2 or not path_4x4:
                    st.error(
                        "Could not locate matching CRD files in test_set_2x2_at_crd_com and test_set_4x4_at_crd_com. "
                        "Please ensure both folders contain a file with the same name as the uploaded CRD."
                    )
                else:
                    # Derive human-readable column / system names from the
                    # actual parent folders where the CRDs live. This keeps
                    # the tables aligned with folders like
                    # S2_BoxOfChocolates_1x1_at_crd_com, etc.
                    def _folder_display_name(path, default_label):
                        try:
                            return os.path.basename(os.path.dirname(path))
                        except Exception:
                            return default_label

                    display_1x1 = None
                    # Prefer the discovered parent folder name when
                    # available; fall back to the internal label.
                    if parent_folder_name:
                        display_1x1 = parent_folder_name
                    elif "test_set_1x1" in set_paths:
                        display_1x1 = _folder_display_name(set_paths["test_set_1x1"], "test_set_1x1")
                    else:
                        display_1x1 = "test_set_1x1"

                    display_2x2 = _folder_display_name(path_2x2, "test_set_2x2")
                    display_4x4 = _folder_display_name(path_4x4, "test_set_4x4")

                    try:
                        # Uploaded file -> test_set_1x1
                        selected_crd_file.seek(0)
                        data_1x1 = json.load(selected_crd_file)

                        # On-disk files -> test_set_2x2 and test_set_4x4
                        with open(path_2x2, "r", encoding="utf-8") as f:
                            data_2x2 = json.load(f)
                        with open(path_4x4, "r", encoding="utf-8") as f:
                            data_4x4 = json.load(f)
                    except Exception as e:
                        st.error(f"Failed to read one of the CRD JSON files: {e}")
                    else:
                        st.write("Using the following CRD files for evaluation:")
                        if "test_set_1x1" in set_paths:
                            st.write(
                                f"- {display_1x1}: {os.path.relpath(set_paths['test_set_1x1'], WORKSPACE_DIR)} (uploaded)"
                            )
                        else:
                            st.write(f"- {display_1x1}: (uploaded file) {crd_name}")
                        st.write(f"- {display_2x2}: {os.path.relpath(path_2x2, WORKSPACE_DIR)}")
                        st.write(f"- {display_4x4}: {os.path.relpath(path_4x4, WORKSPACE_DIR)}")

                        # --- 1) Metric comparison across three systems ---

                        # Alias for compatibility with existing comparison helpers
                        ba_data = data_1x1
                        dell_setup_data = data_2x2
                        config_data = data_4x4

                        def compute_artifact_stats_cfg(data):
                            return _compute_artifact_stats_generic(data)

                        def get_nested_keys_cfg(data, prefix=""):
                            keys = []
                            if isinstance(data, dict):
                                for k, v in data.items():
                                    full_key = f"{prefix}.{k}" if prefix else k
                                    if isinstance(v, dict):
                                        keys.extend(get_nested_keys_cfg(v, full_key))
                                    else:
                                        keys.append(full_key)
                            return keys

                        def get_nested_value_cfg(data, key):
                            keys = key.split(".")
                            val = data
                            for k in keys:
                                if isinstance(val, dict):
                                    val = val.get(k, None)
                                else:
                                    return None
                            return val

                        ba_keys = set(get_nested_keys_cfg(ba_data))
                        dell_setup_keys = set(get_nested_keys_cfg(dell_setup_data))
                        config_keys = set(get_nested_keys_cfg(config_data))
                        matching_keys = sorted(list(ba_keys & dell_setup_keys & config_keys))

                        rows = []
                        for key in matching_keys:
                            if key == "topics" or key.startswith("topics."):
                                continue
                            val_ba = get_nested_value_cfg(ba_data, key)
                            val_dell_setup = get_nested_value_cfg(dell_setup_data, key)
                            val_config = get_nested_value_cfg(config_data, key)
                            rows.append(
                                {
                                    "Metric": key,
                                    display_1x1: val_ba,
                                    display_2x2: val_dell_setup,
                                    display_4x4: val_config,
                                }
                            )

                        # Derived metric: artifact generation duration (mm:ss)
                        artifact_start_key = "meta_data.gen_ai.triggered_at"
                        artifact_end_key = "meta_data.gen_ai.completed_at"
                        if artifact_start_key in matching_keys and artifact_end_key in matching_keys:
                            from datetime import datetime

                            def parse_time(val):
                                try:
                                    return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                                except Exception:
                                    return None

                            def fmt_mmss(start, end):
                                t0 = parse_time(start)
                                t1 = parse_time(end)
                                if not t0 or not t1:
                                    return None
                                total_sec = max(0, int((t1 - t0).total_seconds()))
                                mm, ss = divmod(total_sec, 60)
                                return f"{mm:02d}:{ss:02d}"

                            ba_dur = fmt_mmss(
                                get_nested_value_cfg(ba_data, artifact_start_key),
                                get_nested_value_cfg(ba_data, artifact_end_key),
                            )
                            dell_setup_dur = fmt_mmss(
                                get_nested_value_cfg(dell_setup_data, artifact_start_key),
                                get_nested_value_cfg(dell_setup_data, artifact_end_key),
                            )
                            config_dur = fmt_mmss(
                                get_nested_value_cfg(config_data, artifact_start_key),
                                get_nested_value_cfg(config_data, artifact_end_key),
                            )
                            if any([ba_dur, dell_setup_dur, config_dur]):
                                rows.append(
                                    {
                                        "Metric": "derived.artifact_generation_time_mm_ss",
                                        display_1x1: ba_dur,
                                        display_2x2: dell_setup_dur,
                                        display_4x4: config_dur,
                                    }
                                )

                        ba_stats = compute_artifact_stats_cfg(ba_data)
                        dell_stats = compute_artifact_stats_cfg(dell_setup_data)
                        config_stats = compute_artifact_stats_cfg(config_data)
                        for metric_key in [
                            "derived.total_faqs",
                            "derived.total_summaries",
                            "derived.total_classifications",
                            "derived.total_keywords",
                            "derived.total_sentiments",
                            "derived.total_entities",
                        ]:
                            rows.append(
                                {
                                    "Metric": metric_key,
                                    display_1x1: ba_stats.get(metric_key, 0),
                                    display_2x2: dell_stats.get(metric_key, 0),
                                    display_4x4: config_stats.get(metric_key, 0),
                                }
                            )

                        ba_total_breakdown = sum(ba_stats.values())
                        dell_total_breakdown = sum(dell_stats.values())
                        config_total_breakdown = sum(config_stats.values())
                        rows.append(
                            {
                                "Metric": "derived.total_artifacts_from_breakdown",
                                display_1x1: ba_total_breakdown,
                                display_2x2: dell_total_breakdown,
                                display_4x4: config_total_breakdown,
                            }
                        )


                        columns = [
                            "Metric",
                            display_1x1,
                            display_2x2,
                            display_4x4,
                        ]
                        report_df = pd.DataFrame(rows, columns=columns)

                        # --- Language detection and statistics (document level) ---
                        ba_text = extract_all_text(ba_data)
                        dell_text = extract_all_text(dell_setup_data)
                        config_text = extract_all_text(config_data)
                        ba_lang_stats = get_language_stats(ba_text)
                        dell_lang_stats = get_language_stats(dell_text)
                        config_lang_stats = get_language_stats(config_text)

                        st.subheader("Language Statistics (Document Level)")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.markdown(f"**{display_1x1}**")
                            st.dataframe(language_stats_to_df(ba_lang_stats), use_container_width=True)
                        with col2:
                            st.markdown(f"**{display_2x2}**")
                            st.dataframe(language_stats_to_df(dell_lang_stats), use_container_width=True)
                        with col3:
                            st.markdown(f"**{display_4x4}**")
                            st.dataframe(language_stats_to_df(config_lang_stats), use_container_width=True)

                        highlight_metrics = {
                            "derived.artifact_generation_time_mm_ss",
                            "meta_data.genai_total_artifacts_generated",
                            "meta_data.total_atoms",
                            "derived.total_faqs",
                            "derived.total_summaries",
                            "derived.total_classifications",
                            "derived.total_keywords",
                            "derived.total_sentiments",
                            "derived.total_entities",
                            "derived.total_artifacts_from_breakdown",
                            "meta_data.gen_ai.model",
                            "meta_data.gen_ai.model_provider",
                            "meta_data.gen_ai.llm_id",
                            "meta_data.atomization.model_name",
                            "meta_data.atomization.model_provider",
                            "meta_data.atomization.prompt_version",
                            "meta_data.atomization_status",
                            "meta_data.genai_progress_status",
                        }

                        def highlight_row_cfg(row):
                            if row["Metric"] in highlight_metrics:
                                return ["background-color: #fff3cd; font-weight: bold;" for _ in row]
                            return ["" for _ in row]

                        st.subheader("CRD Metric Comparison (BoozAllen vs Dell Setup vs Dell Config)")
                        styler = report_df.style.apply(highlight_row_cfg, axis=1).set_properties(
                            subset=["Metric"], **{"width": "300px"}
                        )
                        if REPORTING_MODE == "Detailed":
                            st.dataframe(styler, use_container_width=True)
                        else:
                            compact_report_df = report_df[report_df["Metric"].isin(highlight_metrics)].copy()
                            if compact_report_df.empty:
                                compact_report_df = report_df.head(25)
                            st.markdown("#### Key Metric Snapshot")
                            st.dataframe(compact_report_df, use_container_width=True)
                            with st.expander("Show full metric comparison table"):
                                st.dataframe(styler, use_container_width=True)

                        # --- 2) Aggregate quality evaluation vs ground truth context ---

                        def extract_summary_cfg(topic):
                            return extract_primary_summary_from_topic(topic)

                        def extract_summary_components_cfg(data):
                            """Extract all summary components from all topics.
                            Returns list of (component_name, component_text) tuples."""
                            components = []
                            topics = data.get("topics", []) if isinstance(data, dict) else []
                            for topic in topics:
                                components.extend(extract_summary_components_from_topic(topic))
                            return components

                        def extract_faqs_cfg(topic):
                            return extract_faq_texts_from_topic(topic)

                        def extract_keywords_cfg(topic):
                            return extract_keywords_from_topic(topic)

                        def extract_entities_cfg(topic):
                            return extract_entities_from_topic(topic)

                        def extract_classifications_cfg(topic):
                            return extract_classifications_from_topic(topic)

                        def classify_summary_auto_cfg(scores):
                            vals = list(scores.values())
                            overall = _median_or_zero(vals, ndigits=2) if vals else 0.0
                            min_val = min(vals) if vals else 0.0
                            if overall >= 4.5 and min_val >= 4:
                                label = "Excellent"
                            elif overall >= 4.0 and min_val >= 3:
                                label = "Acceptable"
                            elif overall < 3.0 or min_val <= 2:
                                label = "Needs review"
                            else:
                                label = "Mixed"
                            return round(overall, 2), label

                        dims = BASE_DIMS

                        def score_artifact_list_cfg(text_list, context_text, empty_label):
                            if not text_list or not context_text or not str(context_text).strip():
                                return {d: 0.0 for d in dims}, 0.0, empty_label, 0
                            acc = {d: [] for d in dims}
                            for txt in text_list:
                                scores = evaluate_summary_heuristic(context_text, txt)
                                for d in dims:
                                    acc[d].append(scores.get(d, 0.0))
                            avg_scores = {}
                            for d in dims:
                                vals = acc[d]
                                avg_scores[d] = _median_or_zero(vals, ndigits=2) if vals else 0.0
                            overall, label = classify_summary_auto_cfg(avg_scores)
                            return avg_scores, overall, label, len(text_list)

                        def gather_artifacts_cfg(data):
                            topics = data.get("topics", []) if isinstance(data, dict) else []
                            summaries = []
                            faqs = []
                            keywords = []
                            entities = []
                            classifications = []
                            for t in topics:
                                s = extract_summary_cfg(t)
                                if s:
                                    summaries.append(s)
                                faqs.extend(extract_faqs_cfg(t))
                                keywords.extend(extract_keywords_cfg(t))
                                entities.extend(extract_entities_cfg(t))
                                classifications.extend(extract_classifications_cfg(t))
                            return summaries, faqs, keywords, entities, classifications

                        systems = [
                            (display_1x1, ba_data),
                            (display_2x2, dell_setup_data),
                            (display_4x4, config_data),
                        ]

                        summary_rows = []
                        faq_rows = []
                        keyword_rows = []
                        entity_rows = []
                        classification_rows = []

                        for sys_name, data_obj in systems:
                            summaries, faqs_list, kw_list, ent_list, cls_list = gather_artifacts_cfg(data_obj)
                            
                            # Evaluate individual summary components
                            summary_components = extract_summary_components_cfg(data_obj)
                            for comp_name, comp_text in summary_components:
                                if comp_text and gt_context.strip():
                                    sum_scores = evaluate_summary_heuristic(gt_context, comp_text)
                                    sum_overall, sum_label = classify_summary_auto(sum_scores)
                                else:
                                    sum_scores = {d: 0.0 for d in dims}
                                    sum_overall, sum_label = 0.0, "No Context or Summary"
                                
                                summary_rows.append(
                                    {
                                        "System": sys_name,
                                        "Component": comp_name,
                                        **sum_scores,
                                        "Overall": sum_overall,
                                        "Classification": sum_label,
                                        **evaluate_summary_additional_metrics(gt_context, comp_text),
                                        "_summary_text": comp_text,
                                    }
                                )

                            faq_scores, faq_overall, faq_label, faq_count = score_artifact_list_cfg(
                                faqs_list, gt_context, "No FAQs or Context"
                            )
                            faq_rows.append(
                                {
                                    "System": sys_name,
                                    "FAQ_Count": faq_count,
                                    **faq_scores,
                                    "Overall": faq_overall,
                                    "Classification": faq_label,
                                    **evaluate_faq_additional_metrics(faqs_list, gt_context),
                                }
                            )

                            kw_scores, kw_overall, kw_label, kw_count = score_artifact_list_cfg(
                                kw_list, gt_context, "No Keywords or Context"
                            )
                            keyword_rows.append(
                                {
                                    "System": sys_name,
                                    "Keyword_Count": kw_count,
                                    **kw_scores,
                                    "Overall": kw_overall,
                                    "Classification": kw_label,
                                    **evaluate_keyword_additional_metrics(kw_list, gt_context, top_k=keyword_top_k),
                                }
                            )

                            ent_scores, ent_overall, ent_label, ent_count = score_artifact_list_cfg(
                                ent_list, gt_context, "No Entities or Context"
                            )
                            entity_rows.append(
                                {
                                    "System": sys_name,
                                    "Entity_Count": ent_count,
                                    **ent_scores,
                                    "Overall": ent_overall,
                                    "Classification": ent_label,
                                    **entity_label_metrics(),
                                }
                            )

                            cls_scores, cls_overall, cls_label, cls_count = score_artifact_list_cfg(
                                cls_list, gt_context, "No Classifications or Context"
                            )
                            classification_rows.append(
                                {
                                    "System": sys_name,
                                    "Classification_Count": cls_count,
                                    **cls_scores,
                                    "Overall": cls_overall,
                                    "Classification": cls_label,
                                    **classification_label_metrics(),
                                }
                            )

                        if summary_rows:
                            summary_df = _add_consistency_scores(
                                pd.DataFrame(summary_rows), ["Component"]
                            )
                            render_quality_table(
                                "#### Overall Summary Quality vs Ground Truth (automatic, 1–5 scale)",
                                "Summary",
                                "no_gt_summary",
                                summary_df,
                                outlier_label="Summary",
                                outlier_group_cols=["System", "Component"],
                            )

                        if faq_rows:
                            faq_df = pd.DataFrame(faq_rows)
                            render_quality_table(
                                "#### Overall FAQ Quality vs Ground Truth (automatic, 1–5 scale)",
                                "FAQ",
                                "no_gt_faq",
                                faq_df,
                                outlier_label="FAQ",
                                outlier_group_cols=["System"],
                            )

                        if keyword_rows:
                            kw_df = pd.DataFrame(keyword_rows)
                            render_quality_table(
                                "#### Overall Keyword Quality vs Ground Truth (automatic, 1–5 scale)",
                                "Keyword",
                                "no_gt_keyword",
                                kw_df,
                                outlier_label="Keyword",
                                outlier_group_cols=["System"],
                            )

                        if entity_rows:
                            ent_df = pd.DataFrame(entity_rows)
                            render_quality_table(
                                "#### Overall Entity Quality vs Ground Truth (automatic, 1–5 scale)",
                                "Entity",
                                "no_gt_entity",
                                ent_df,
                                outlier_label="Entity",
                                outlier_group_cols=["System"],
                            )

                        if classification_rows:
                            cls_df = pd.DataFrame(classification_rows)
                            render_quality_table(
                                "#### Overall Classification Quality vs Ground Truth (automatic, 1–5 scale)",
                                "Classification",
                                "no_gt_classification",
                                cls_df,
                                outlier_label="Classification",
                                outlier_group_cols=["System"],
                            )

                        analysis_tables_cfg = render_quality_analysis_section(
                            summary_df=summary_df if summary_rows else None,
                            faq_df=faq_df if faq_rows else None,
                            keyword_df=kw_df if keyword_rows else None,
                            entity_df=ent_df if entity_rows else None,
                            classification_df=cls_df if classification_rows else None,
                            sentiment_df=None,
                            title="Configuration Quality Summary and Comparative Analysis",
                        )

                        # --- Download full evaluation report as Excel ---
                        try:
                            excel_sheets = {"CRD_Metrics": report_df}
                            if summary_rows:
                                excel_sheets["Summary_Quality"] = summary_df
                            if faq_rows:
                                excel_sheets["FAQ_Quality"] = faq_df
                            if keyword_rows:
                                excel_sheets["Keyword_Quality"] = kw_df
                            if entity_rows:
                                excel_sheets["Entity_Quality"] = ent_df
                            if classification_rows:
                                excel_sheets["Classification_Quality"] = cls_df
                            if analysis_tables_cfg.get("component_scores") is not None and not analysis_tables_cfg["component_scores"].empty:
                                excel_sheets["Analysis_ComponentScores"] = analysis_tables_cfg["component_scores"]
                            if analysis_tables_cfg.get("component_winners") is not None and not analysis_tables_cfg["component_winners"].empty:
                                excel_sheets["Analysis_ComponentBest"] = analysis_tables_cfg["component_winners"]
                            if analysis_tables_cfg.get("section_scorecard") is not None and not analysis_tables_cfg["section_scorecard"].empty:
                                excel_sheets["Analysis_SectionScores"] = analysis_tables_cfg["section_scorecard"]
                            if analysis_tables_cfg.get("section_winners") is not None and not analysis_tables_cfg["section_winners"].empty:
                                excel_sheets["Analysis_SectionBest"] = analysis_tables_cfg["section_winners"]
                            if analysis_tables_cfg.get("setup_ranking") is not None and not analysis_tables_cfg["setup_ranking"].empty:
                                excel_sheets["Analysis_SetupRanking"] = analysis_tables_cfg["setup_ranking"]
                            if analysis_tables_cfg.get("setup_strengths") is not None and not analysis_tables_cfg["setup_strengths"].empty:
                                excel_sheets["Analysis_SetupStrengths"] = analysis_tables_cfg["setup_strengths"]
                            if analysis_tables_cfg.get("summary_dimension_profile") is not None and not analysis_tables_cfg["summary_dimension_profile"].empty:
                                excel_sheets["Analysis_DimProfile"] = analysis_tables_cfg["summary_dimension_profile"]

                            buffer = BytesIO()
                            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                                for sheet_name, df in excel_sheets.items():
                                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                            buffer.seek(0)

                            st.download_button(
                                label="Download report (.xlsx)",
                                data=buffer.getvalue(),
                                file_name=f"{crd_name}_dell_configuration_evaluation.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
                        except Exception as e:
                            st.error(f"Failed to generate Excel report: {e}")

        # --- Bulk evaluation for all uploaded pairs ---
        if st.button("Bulk Evaluate All", key="dell_config_bulk_evaluate"):
            if not gt_md_files or not crd_files:
                st.error("Please upload at least one ground truth (.md) file and one CRD (.json) file before running bulk evaluation.")
            else:
                bulk_metric_rows = []
                bulk_summary_rows = []
                bulk_faq_rows = []
                bulk_keyword_rows = []
                bulk_entity_rows = []
                bulk_classification_rows = []

                # Helper functions for metrics-only comparison
                def _compute_artifact_stats_bulk(data):
                    return _compute_artifact_stats_generic(data)

                def _get_nested_keys_bulk(data, prefix=""):
                    keys = []
                    if isinstance(data, dict):
                        for k, v in data.items():
                            full_key = f"{prefix}.{k}" if prefix else k
                            if isinstance(v, dict):
                                keys.extend(_get_nested_keys_bulk(v, full_key))
                            else:
                                keys.append(full_key)
                    return keys

                def _get_nested_value_bulk(data, key):
                    parts = key.split(".")
                    val = data
                    for k in parts:
                        if isinstance(val, dict):
                            val = val.get(k, None)
                        else:
                            return None
                    return val

                from datetime import datetime

                def _parse_time_bulk(val):
                    try:
                        return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                    except Exception:
                        return None

                def _fmt_mmss_bulk(start, end):
                    t0 = _parse_time_bulk(start)
                    t1 = _parse_time_bulk(end)
                    if not t0 or not t1:
                        return None
                    total_sec = max(0, int((t1 - t0).total_seconds()))
                    mm, ss = divmod(total_sec, 60)
                    return f"{mm:02d}:{ss:02d}"

                # Helper functions for quality evaluation vs ground truth

                def _extract_summary_bulk(topic):
                    return extract_primary_summary_from_topic(topic)

                def _extract_faqs_bulk(topic):
                    return extract_faq_texts_from_topic(topic)

                def _extract_keywords_bulk(topic):
                    return extract_keywords_from_topic(topic)

                def _extract_entities_bulk(topic):
                    return extract_entities_from_topic(topic)

                def _extract_classifications_bulk(topic):
                    return extract_classifications_from_topic(topic)

                def _classify_summary_auto_bulk(scores):
                    vals = list(scores.values())
                    overall = _median_or_zero(vals, ndigits=2) if vals else 0.0
                    min_val = min(vals) if vals else 0.0
                    if overall >= 4.5 and min_val >= 4:
                        label = "Excellent"
                    elif overall >= 4.0 and min_val >= 3:
                        label = "Acceptable"
                    elif overall < 3.0 or min_val <= 2:
                        label = "Needs review"
                    else:
                        label = "Mixed"
                    return round(overall, 2), label

                bulk_dims = BASE_DIMS

                def _score_artifact_list_bulk(text_list, context_text, empty_label):
                    if not text_list or not context_text or not str(context_text).strip():
                        return {d: 0.0 for d in bulk_dims}, 0.0, empty_label, 0
                    acc = {d: [] for d in bulk_dims}
                    for txt in text_list:
                        scores = evaluate_summary_heuristic(context_text, txt)
                        for d in bulk_dims:
                            acc[d].append(scores.get(d, 0.0))
                    avg_scores = {}
                    for d in bulk_dims:
                        vals = acc[d]
                        avg_scores[d] = _median_or_zero(vals, ndigits=2) if vals else 0.0
                    overall, label = _classify_summary_auto_bulk(avg_scores)
                    return avg_scores, overall, label, len(text_list)

                def _gather_artifacts_bulk(data):
                    topics = data.get("topics", []) if isinstance(data, dict) else []
                    summaries = []
                    faqs = []
                    keywords = []
                    entities = []
                    classifications = []
                    for t in topics:
                        s = _extract_summary_bulk(t)
                        if s:
                            summaries.append(s)
                        faqs.extend(_extract_faqs_bulk(t))
                        keywords.extend(_extract_keywords_bulk(t))
                        entities.extend(_extract_entities_bulk(t))
                        classifications.extend(_extract_classifications_bulk(t))
                    return summaries, faqs, keywords, entities, classifications

                for md_file in gt_md_files:
                    md_base = os.path.splitext(md_file.name)[0]
                    md_base_lower = md_base.lower()

                    # Find corresponding CRD in uploaded 1x1 files by name substring (case-insensitive)
                    matching_crds = [
                        cf for cf in crd_files if md_base_lower in cf.name.lower()
                    ]
                    if not matching_crds:
                        st.warning(f"No CRD (.json) uploaded that matches ground truth '{md_file.name}'. Skipping.")
                        continue

                    crd_1x1 = matching_crds[0]
                    crd_name_bulk = crd_1x1.name

                    # Find 2x2 and 4x4 versions on disk
                    path_2x2_bulk = os.path.join(dell_config_root, "test_set_2x2_at_crd_com", crd_name_bulk)
                    path_4x4_bulk = os.path.join(dell_config_root, "test_set_4x4_at_crd_com", crd_name_bulk)

                    if not os.path.exists(path_2x2_bulk) or not os.path.exists(path_4x4_bulk):
                        st.warning(
                            f"Missing 2x2 or 4x4 CRD for '{crd_name_bulk}' under Dell_Configurations. Skipping this document."
                        )
                        continue

                    try:
                        # Read ground truth context for this document
                        md_file.seek(0)
                        md_text = md_file.read().decode("utf-8", errors="ignore")

                        crd_1x1.seek(0)
                        data_1x1_bulk = json.load(crd_1x1)
                        with open(path_2x2_bulk, "r", encoding="utf-8") as f:
                            data_2x2_bulk = json.load(f)
                        with open(path_4x4_bulk, "r", encoding="utf-8") as f:
                            data_4x4_bulk = json.load(f)
                    except Exception as e:
                        st.warning(f"Failed to read CRD JSON for '{crd_name_bulk}': {e}. Skipping.")
                        continue

                    # Metric comparison for this document
                    keys_1x1 = set(_get_nested_keys_bulk(data_1x1_bulk))
                    keys_2x2 = set(_get_nested_keys_bulk(data_2x2_bulk))
                    keys_4x4 = set(_get_nested_keys_bulk(data_4x4_bulk))
                    matching_keys_bulk = sorted(list(keys_1x1 & keys_2x2 & keys_4x4))

                    for key in matching_keys_bulk:
                        if key == "topics" or key.startswith("topics."):
                            continue
                        val_1x1 = _get_nested_value_bulk(data_1x1_bulk, key)
                        val_2x2 = _get_nested_value_bulk(data_2x2_bulk, key)
                        val_4x4 = _get_nested_value_bulk(data_4x4_bulk, key)
                        bulk_metric_rows.append(
                            {
                                "Document": md_file.name,
                                "CRD_Name": crd_name_bulk,
                                "Metric": key,
                                "test_set_1x1": val_1x1,
                                "test_set_2x2": val_2x2,
                                "test_set_4x4": val_4x4,
                            }
                        )

                    # Derived generation time
                    artifact_start_key = "meta_data.gen_ai.triggered_at"
                    artifact_end_key = "meta_data.gen_ai.completed_at"
                    if artifact_start_key in matching_keys_bulk and artifact_end_key in matching_keys_bulk:
                        dur_1x1 = _fmt_mmss_bulk(
                            _get_nested_value_bulk(data_1x1_bulk, artifact_start_key),
                            _get_nested_value_bulk(data_1x1_bulk, artifact_end_key),
                        )
                        dur_2x2 = _fmt_mmss_bulk(
                            _get_nested_value_bulk(data_2x2_bulk, artifact_start_key),
                            _get_nested_value_bulk(data_2x2_bulk, artifact_end_key),
                        )
                        dur_4x4 = _fmt_mmss_bulk(
                            _get_nested_value_bulk(data_4x4_bulk, artifact_start_key),
                            _get_nested_value_bulk(data_4x4_bulk, artifact_end_key),
                        )
                        if any([dur_1x1, dur_2x2, dur_4x4]):
                            bulk_metric_rows.append(
                                {
                                    "Document": md_file.name,
                                    "CRD_Name": crd_name_bulk,
                                    "Metric": "derived.artifact_generation_time_mm_ss",
                                    "test_set_1x1": dur_1x1,
                                    "test_set_2x2": dur_2x2,
                                    "test_set_4x4": dur_4x4,
                                }
                            )

                    # Derived artifact counts
                    stats_1x1 = _compute_artifact_stats_bulk(data_1x1_bulk)
                    stats_2x2 = _compute_artifact_stats_bulk(data_2x2_bulk)
                    stats_4x4 = _compute_artifact_stats_bulk(data_4x4_bulk)
                    for metric_key in [
                        "derived.total_faqs",
                        "derived.total_summaries",
                        "derived.total_classifications",
                        "derived.total_keywords",
                        "derived.total_sentiments",
                        "derived.total_entities",
                    ]:
                        bulk_metric_rows.append(
                            {
                                "Document": md_file.name,
                                "CRD_Name": crd_name_bulk,
                                "Metric": metric_key,
                                "test_set_1x1": stats_1x1.get(metric_key, 0),
                                "test_set_2x2": stats_2x2.get(metric_key, 0),
                                "test_set_4x4": stats_4x4.get(metric_key, 0),
                            }
                        )

                    total_1x1 = sum(stats_1x1.values())
                    total_2x2 = sum(stats_2x2.values())
                    total_4x4 = sum(stats_4x4.values())
                    bulk_metric_rows.append(
                        {
                            "Document": md_file.name,
                            "CRD_Name": crd_name_bulk,
                            "Metric": "derived.total_artifacts_from_breakdown",
                            "test_set_1x1": total_1x1,
                            "test_set_2x2": total_2x2,
                            "test_set_4x4": total_4x4,
                        }
                    )

                    # --- Quality evaluation vs ground truth for this document ---

                    systems_bulk = [
                        ("test_set_1x1", data_1x1_bulk),
                        ("test_set_2x2", data_2x2_bulk),
                        ("test_set_4x4", data_4x4_bulk),
                    ]

                    for sys_name, data_obj in systems_bulk:
                        summaries, faqs_list, kw_list, ent_list, cls_list = _gather_artifacts_bulk(data_obj)

                        sum_scores, sum_overall, sum_label, _ = _score_artifact_list_bulk(
                            summaries, md_text, "No Summaries or Context"
                        )
                        bulk_summary_rows.append(
                            {
                                "Document": md_file.name,
                                "CRD_Name": crd_name_bulk,
                                "System": sys_name,
                                **sum_scores,
                                "Overall": sum_overall,
                                "Classification": sum_label,
                                **evaluate_summary_additional_metrics(md_text, "\n".join(summaries)),
                                "_summary_text": "\n".join(summaries),
                            }
                        )

                        faq_scores, faq_overall, faq_label, faq_count = _score_artifact_list_bulk(
                            faqs_list, md_text, "No FAQs or Context"
                        )
                        bulk_faq_rows.append(
                            {
                                "Document": md_file.name,
                                "CRD_Name": crd_name_bulk,
                                "System": sys_name,
                                "FAQ_Count": faq_count,
                                **faq_scores,
                                "Overall": faq_overall,
                                "Classification": faq_label,
                                **evaluate_faq_additional_metrics(faqs_list, md_text),
                            }
                        )

                        kw_scores, kw_overall, kw_label, kw_count = _score_artifact_list_bulk(
                            kw_list, md_text, "No Keywords or Context"
                        )
                        bulk_keyword_rows.append(
                            {
                                "Document": md_file.name,
                                "CRD_Name": crd_name_bulk,
                                "System": sys_name,
                                "Keyword_Count": kw_count,
                                **kw_scores,
                                "Overall": kw_overall,
                                "Classification": kw_label,
                                **evaluate_keyword_additional_metrics(kw_list, md_text, top_k=keyword_top_k),
                            }
                        )

                        ent_scores, ent_overall, ent_label, ent_count = _score_artifact_list_bulk(
                            ent_list, md_text, "No Entities or Context"
                        )
                        bulk_entity_rows.append(
                            {
                                "Document": md_file.name,
                                "CRD_Name": crd_name_bulk,
                                "System": sys_name,
                                "Entity_Count": ent_count,
                                **ent_scores,
                                "Overall": ent_overall,
                                "Classification": ent_label,
                                **entity_label_metrics(),
                            }
                        )

                        cls_scores, cls_overall, cls_label, cls_count = _score_artifact_list_bulk(
                            cls_list, md_text, "No Classifications or Context"
                        )
                        bulk_classification_rows.append(
                            {
                                "Document": md_file.name,
                                "CRD_Name": crd_name_bulk,
                                "System": sys_name,
                                "Classification_Count": cls_count,
                                **cls_scores,
                                "Overall": cls_overall,
                                "Classification": cls_label,
                                **classification_label_metrics(),
                            }
                        )

                if not bulk_metric_rows:
                    st.warning("No documents could be evaluated in bulk. Please check file name mappings and folder contents.")
                else:
                    bulk_df = pd.DataFrame(
                        bulk_metric_rows,
                        columns=[
                            "Document",
                            "CRD_Name",
                            "Metric",
                            "test_set_1x1",
                            "test_set_2x2",
                            "test_set_4x4",
                        ],
                    )

                    st.subheader("Bulk CRD Metric Comparison")
                    if REPORTING_MODE == "Detailed":
                        st.dataframe(bulk_df, use_container_width=True)
                    else:
                        compact_bulk = bulk_df.copy()
                        if "Metric" in compact_bulk.columns:
                            keep_bulk = [
                                "derived.artifact_generation_time_mm_ss",
                                "meta_data.genai_total_artifacts_generated",
                                "derived.total_faqs",
                                "derived.total_summaries",
                                "derived.total_classifications",
                                "derived.total_keywords",
                                "derived.total_sentiments",
                                "derived.total_entities",
                            ]
                            compact_bulk = compact_bulk[compact_bulk["Metric"].isin(keep_bulk)]
                        if compact_bulk.empty:
                            compact_bulk = bulk_df.head(30)
                        st.markdown("#### Key Bulk Metric Snapshot")
                        st.dataframe(compact_bulk, use_container_width=True)
                        with st.expander("Show full bulk metric comparison table"):
                            st.dataframe(bulk_df, use_container_width=True)

                    # --- Language stats for bulk evaluation ---
                    # Show language stats for each document and system
                    st.subheader("Language Statistics (Bulk, Document Level)")
                    for md_file in gt_md_files:
                        md_base = os.path.splitext(md_file.name)[0]
                        crd_1x1 = next((cf for cf in crd_files if md_base.lower() in cf.name.lower()), None)
                        if not crd_1x1:
                            continue
                        crd_name_bulk = crd_1x1.name
                        path_2x2_bulk = os.path.join(dell_config_root, "test_set_2x2_at_crd_com", crd_name_bulk)
                        path_4x4_bulk = os.path.join(dell_config_root, "test_set_4x4_at_crd_com", crd_name_bulk)
                        if not os.path.exists(path_2x2_bulk) or not os.path.exists(path_4x4_bulk):
                            continue
                        crd_1x1.seek(0)
                        data_1x1_bulk = json.load(crd_1x1)
                        with open(path_2x2_bulk, "r", encoding="utf-8") as f:
                            data_2x2_bulk = json.load(f)
                        with open(path_4x4_bulk, "r", encoding="utf-8") as f:
                            data_4x4_bulk = json.load(f)
                        text_1x1 = extract_all_text(data_1x1_bulk)
                        text_2x2 = extract_all_text(data_2x2_bulk)
                        text_4x4 = extract_all_text(data_4x4_bulk)
                        stats_1x1 = get_language_stats(text_1x1)
                        stats_2x2 = get_language_stats(text_2x2)
                        stats_4x4 = get_language_stats(text_4x4)

                        # Build a single table for this document across all systems
                        rows = []
                        for system_name, stats in [
                            ("test_set_1x1", stats_1x1),
                            ("test_set_2x2", stats_2x2),
                            ("test_set_4x4", stats_4x4),
                        ]:
                            for code, pct in sorted(stats.items(), key=lambda x: -x[1]):
                                rows.append(
                                    {
                                        "System": system_name,
                                        "Language": get_language_name(code),
                                        "Code": code.upper(),
                                        "Percentage": pct,
                                    }
                                )

                        lang_df = (
                            pd.DataFrame(rows, columns=["System", "Language", "Code", "Percentage"])
                            if rows
                            else pd.DataFrame(columns=["System", "Language", "Code", "Percentage"])
                        )

                        st.markdown(f"**{md_file.name}**")
                        st.dataframe(lang_df, use_container_width=True)

                    # Build quality DataFrames if any rows were collected
                    bulk_summary_df = None
                    bulk_faq_df = None
                    bulk_keyword_df = None
                    bulk_entity_df = None
                    bulk_cls_df = None

                    if bulk_summary_rows:
                        bulk_summary_df = _add_consistency_scores(
                            pd.DataFrame(bulk_summary_rows), ["Document", "CRD_Name"]
                        )
                        render_quality_table(
                            "#### Bulk Summary Quality vs Ground Truth (automatic, 1–5 scale)",
                            "Summary",
                            "bulk_summary",
                            bulk_summary_df,
                            outlier_label="Bulk Summary",
                            outlier_group_cols=["Document", "System"],
                        )

                    if bulk_faq_rows:
                        bulk_faq_df = pd.DataFrame(bulk_faq_rows)
                        render_quality_table(
                            "#### Bulk FAQ Quality vs Ground Truth (automatic, 1–5 scale)",
                            "FAQ",
                            "bulk_faq",
                            bulk_faq_df,
                            outlier_label="Bulk FAQ",
                            outlier_group_cols=["Document", "System"],
                        )

                    if bulk_keyword_rows:
                        bulk_keyword_df = pd.DataFrame(bulk_keyword_rows)
                        render_quality_table(
                            "#### Bulk Keyword Quality vs Ground Truth (automatic, 1–5 scale)",
                            "Keyword",
                            "bulk_keyword",
                            bulk_keyword_df,
                            outlier_label="Bulk Keyword",
                            outlier_group_cols=["Document", "System"],
                        )

                    if bulk_entity_rows:
                        bulk_entity_df = pd.DataFrame(bulk_entity_rows)
                        render_quality_table(
                            "#### Bulk Entity Quality vs Ground Truth (automatic, 1–5 scale)",
                            "Entity",
                            "bulk_entity",
                            bulk_entity_df,
                            outlier_label="Bulk Entity",
                            outlier_group_cols=["Document", "System"],
                        )

                    if bulk_classification_rows:
                        bulk_cls_df = pd.DataFrame(bulk_classification_rows)
                        render_quality_table(
                            "#### Bulk Classification Quality vs Ground Truth (automatic, 1–5 scale)",
                            "Classification",
                            "bulk_classification",
                            bulk_cls_df,
                            outlier_label="Bulk Classification",
                            outlier_group_cols=["Document", "System"],
                        )

                    analysis_tables_bulk = render_quality_analysis_section(
                        summary_df=bulk_summary_df,
                        faq_df=bulk_faq_df,
                        keyword_df=bulk_keyword_df,
                        entity_df=bulk_entity_df,
                        classification_df=bulk_cls_df,
                        sentiment_df=None,
                        title="Bulk Quality Summary and Comparative Analysis",
                    )

                    try:
                        bulk_buffer = BytesIO()
                        with pd.ExcelWriter(bulk_buffer, engine="xlsxwriter") as writer:
                            bulk_df.to_excel(writer, sheet_name="CRD_Metrics_Bulk", index=False)
                            if bulk_summary_df is not None:
                                bulk_summary_df.to_excel(writer, sheet_name="Summary_Quality_Bulk", index=False)
                            if bulk_faq_df is not None:
                                bulk_faq_df.to_excel(writer, sheet_name="FAQ_Quality_Bulk", index=False)
                            if bulk_keyword_df is not None:
                                bulk_keyword_df.to_excel(writer, sheet_name="Keyword_Quality_Bulk", index=False)
                            if bulk_entity_df is not None:
                                bulk_entity_df.to_excel(writer, sheet_name="Entity_Quality_Bulk", index=False)
                            if bulk_cls_df is not None:
                                bulk_cls_df.to_excel(writer, sheet_name="Classification_Quality_Bulk", index=False)
                            if analysis_tables_bulk.get("component_scores") is not None and not analysis_tables_bulk["component_scores"].empty:
                                analysis_tables_bulk["component_scores"].to_excel(writer, sheet_name="Analysis_ComponentScores", index=False)
                            if analysis_tables_bulk.get("component_winners") is not None and not analysis_tables_bulk["component_winners"].empty:
                                analysis_tables_bulk["component_winners"].to_excel(writer, sheet_name="Analysis_ComponentBest", index=False)
                            if analysis_tables_bulk.get("section_scorecard") is not None and not analysis_tables_bulk["section_scorecard"].empty:
                                analysis_tables_bulk["section_scorecard"].to_excel(writer, sheet_name="Analysis_SectionScores", index=False)
                            if analysis_tables_bulk.get("section_winners") is not None and not analysis_tables_bulk["section_winners"].empty:
                                analysis_tables_bulk["section_winners"].to_excel(writer, sheet_name="Analysis_SectionBest", index=False)
                            if analysis_tables_bulk.get("setup_ranking") is not None and not analysis_tables_bulk["setup_ranking"].empty:
                                analysis_tables_bulk["setup_ranking"].to_excel(writer, sheet_name="Analysis_SetupRanking", index=False)
                        bulk_buffer.seek(0)

                        st.download_button(
                            label="Download bulk report (.xlsx)",
                            data=bulk_buffer.getvalue(),
                            file_name="dell_configuration_bulk_evaluation.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    except Exception as e:
                        st.error(f"Failed to generate bulk Excel report: {e}")

elif nav == "Dell_Config_no_GT":
    st.subheader("Dell Configuration Evaluation (No Ground Truth)")

    st.write(
        "Upload one or more CRD JSON files from a Dell_Configurations subfolder. "
        "The evaluation in this tab uses each topic's own 'context' field as the reference "
        "for scoring summaries, FAQs, keywords, entities, and classifications."
    )

    # CRD JSON uploader – multiple files, 1x1 variant selection
    crd_files_ng = st.file_uploader(
        "CRD File(s) (.json) from a Dell_Configurations subfolder (e.g. test_set_1x1_at_crd_com or S2_BoxOfChocolates_1x1_at_crd_com)",
        type=["json"],
        key="dell_crd_json_multi_nogt",
        accept_multiple_files=True,
    )

    set_paths_ng = {}
    crd_name_ng = None
    selected_crd_file_ng = None
    if crd_files_ng:
        crd_names_ng = [f.name for f in crd_files_ng]
        crd_name_ng = st.selectbox(
            "Select CRD file (treated as 1x1 variant)",
            crd_names_ng,
            key="dell_crd_select_nogt",
        )
        selected_crd_file_ng = next((f for f in crd_files_ng if f.name == crd_name_ng), None)

    dell_config_root = os.path.join(WORKSPACE_DIR, "Dell_Configurations")

    if selected_crd_file_ng is not None:
        st.success(f"CRD file selected (treated as 1x1 variant): {crd_name_ng}")

        # Discover the parent folder and sibling 2x2 / 4x4 folders
        parent_folder_name_ng = None
        parent_candidate_path_ng = None
        if os.path.isdir(dell_config_root):
            for entry in os.scandir(dell_config_root):
                if not entry.is_dir():
                    continue
                candidate = os.path.join(entry.path, crd_name_ng)
                if os.path.exists(candidate):
                    parent_folder_name_ng = os.path.basename(entry.path)
                    parent_candidate_path_ng = candidate
                    break

        set_paths_ng = {}

        dim_to_label_ng = {
            "1x1": "test_set_1x1",
            "2x2": "test_set_2x2",
            "4x4": "test_set_4x4",
        }

        if parent_folder_name_ng:
            m = re.search(r"^(.*?)(1x1|2x2|4x4)", parent_folder_name_ng)
            if m:
                base_prefix_ng, first_dim_ng = m.group(1), m.group(2)

                if first_dim_ng in dim_to_label_ng and parent_candidate_path_ng:
                    set_paths_ng[dim_to_label_ng[first_dim_ng]] = parent_candidate_path_ng

                for entry in os.scandir(dell_config_root):
                    if not entry.is_dir():
                        continue
                    folder_name_ng = entry.name
                    m2 = re.search(r"^(.*?)(1x1|2x2|4x4)", folder_name_ng)
                    if not m2:
                        continue
                    prefix_ng, dim_ng = m2.group(1), m2.group(2)
                    if prefix_ng != base_prefix_ng:
                        continue

                    label_ng = dim_to_label_ng.get(dim_ng)
                    if not label_ng:
                        continue

                    candidate = os.path.join(entry.path, crd_name_ng)
                    if os.path.exists(candidate):
                        set_paths_ng[label_ng] = candidate
        else:
            fixed_folders_ng = [
                ("test_set_1x1", "test_set_1x1_at_crd_com"),
                ("test_set_2x2", "test_set_2x2_at_crd_com"),
                ("test_set_4x4", "test_set_4x4_at_crd_com"),
            ]

            if os.path.isdir(dell_config_root):
                for label_ng, folder_ng in fixed_folders_ng:
                    candidate = os.path.join(dell_config_root, folder_ng, crd_name_ng)
                    if os.path.exists(candidate):
                        set_paths_ng[label_ng] = candidate

        if set_paths_ng:
            st.write("Matching CRD files found in Dell_Configurations:")
            for _label_ng, path in set_paths_ng.items():
                rel = os.path.relpath(path, WORKSPACE_DIR)
                folder_name = os.path.basename(os.path.dirname(path))
                st.write(f"- {folder_name}: {rel}")
        else:
            st.warning(
                "No matching CRD files with this name were found under Dell_Configurations/test_set_* folders."
            )

        if st.button("Evaluate (No GT)", key="dell_config_evaluate_nogt"):
            path_1x1_ng = set_paths_ng.get("test_set_1x1")
            path_2x2_ng = set_paths_ng.get("test_set_2x2")
            path_4x4_ng = set_paths_ng.get("test_set_4x4")

            if not path_2x2_ng or not path_4x4_ng:
                st.error(
                    "Could not locate matching CRD files in 2x2 and 4x4 folders. "
                    "Please ensure both folders contain a file with the same name as the uploaded CRD."
                )
            else:
                def _folder_display_name_ng(path, default_label):
                    try:
                        return os.path.basename(os.path.dirname(path))
                    except Exception:
                        return default_label

                if parent_folder_name_ng:
                    display_1x1_ng = parent_folder_name_ng
                elif path_1x1_ng:
                    display_1x1_ng = _folder_display_name_ng(path_1x1_ng, "test_set_1x1")
                else:
                    display_1x1_ng = "test_set_1x1"

                display_2x2_ng = _folder_display_name_ng(path_2x2_ng, "test_set_2x2")
                display_4x4_ng = _folder_display_name_ng(path_4x4_ng, "test_set_4x4")

                try:
                    selected_crd_file_ng.seek(0)
                    data_1x1_ng = json.load(selected_crd_file_ng)
                    with open(path_2x2_ng, "r", encoding="utf-8") as f:
                        data_2x2_ng = json.load(f)
                    with open(path_4x4_ng, "r", encoding="utf-8") as f:
                        data_4x4_ng = json.load(f)
                except Exception as e:
                    st.error(f"Failed to read one of the CRD JSON files: {e}")
                else:
                    st.write("Using the following CRD files for evaluation (no ground truth):")
                    if path_1x1_ng:
                        st.write(
                            f"- {display_1x1_ng}: {os.path.relpath(path_1x1_ng, WORKSPACE_DIR)} (uploaded)"
                        )
                    else:
                        st.write(f"- {display_1x1_ng}: (uploaded file) {crd_name_ng}")
                    st.write(f"- {display_2x2_ng}: {os.path.relpath(path_2x2_ng, WORKSPACE_DIR)}")
                    st.write(f"- {display_4x4_ng}: {os.path.relpath(path_4x4_ng, WORKSPACE_DIR)}")

                    # --- Metric comparison (same as GT tab) ---
                    def _compute_artifact_stats_nogt(data):
                        return _compute_artifact_stats_generic(data)

                    def _get_nested_keys_nogt(data, prefix=""):
                        keys = []
                        if isinstance(data, dict):
                            for k, v in data.items():
                                full_key = f"{prefix}.{k}" if prefix else k
                                if isinstance(v, dict):
                                    keys.extend(_get_nested_keys_nogt(v, full_key))
                                else:
                                    keys.append(full_key)
                        return keys

                    def _get_nested_value_nogt(data, key):
                        parts = key.split(".")
                        val = data
                        for k in parts:
                            if isinstance(val, dict):
                                val = val.get(k, None)
                            else:
                                return None
                        return val

                    keys_1x1_ng = set(_get_nested_keys_nogt(data_1x1_ng))
                    keys_2x2_ng = set(_get_nested_keys_nogt(data_2x2_ng))
                    keys_4x4_ng = set(_get_nested_keys_nogt(data_4x4_ng))
                    matching_keys_ng = sorted(list(keys_1x1_ng & keys_2x2_ng & keys_4x4_ng))

                    rows_ng = []
                    for key in matching_keys_ng:
                        if key == "topics" or key.startswith("topics."):
                            continue
                        val_1x1 = _get_nested_value_nogt(data_1x1_ng, key)
                        val_2x2 = _get_nested_value_nogt(data_2x2_ng, key)
                        val_4x4 = _get_nested_value_nogt(data_4x4_ng, key)
                        rows_ng.append(
                            {
                                "Metric": key,
                                display_1x1_ng: val_1x1,
                                display_2x2_ng: val_2x2,
                                display_4x4_ng: val_4x4,
                            }
                        )

                    artifact_start_key = "meta_data.gen_ai.triggered_at"
                    artifact_end_key = "meta_data.gen_ai.completed_at"
                    if artifact_start_key in matching_keys_ng and artifact_end_key in matching_keys_ng:
                        from datetime import datetime

                        def _parse_time_nogt(val):
                            try:
                                return datetime.fromisoformat(str(val).replace("Z", "+00:00"))
                            except Exception:
                                return None

                        def _fmt_mmss_nogt(start, end):
                            t0 = _parse_time_nogt(start)
                            t1 = _parse_time_nogt(end)
                            if not t0 or not t1:
                                return None
                            total_sec = max(0, int((t1 - t0).total_seconds()))
                            mm, ss = divmod(total_sec, 60)
                            return f"{mm:02d}:{ss:02d}"

                        dur_1x1 = _fmt_mmss_nogt(
                            _get_nested_value_nogt(data_1x1_ng, artifact_start_key),
                            _get_nested_value_nogt(data_1x1_ng, artifact_end_key),
                        )
                        dur_2x2 = _fmt_mmss_nogt(
                            _get_nested_value_nogt(data_2x2_ng, artifact_start_key),
                            _get_nested_value_nogt(data_2x2_ng, artifact_end_key),
                        )
                        dur_4x4 = _fmt_mmss_nogt(
                            _get_nested_value_nogt(data_4x4_ng, artifact_start_key),
                            _get_nested_value_nogt(data_4x4_ng, artifact_end_key),
                        )
                        if any([dur_1x1, dur_2x2, dur_4x4]):
                            rows_ng.append(
                                {
                                    "Metric": "derived.artifact_generation_time_mm_ss",
                                    display_1x1_ng: dur_1x1,
                                    display_2x2_ng: dur_2x2,
                                    display_4x4_ng: dur_4x4,
                                }
                            )

                    stats_1x1_ng = _compute_artifact_stats_nogt(data_1x1_ng)
                    stats_2x2_ng = _compute_artifact_stats_nogt(data_2x2_ng)
                    stats_4x4_ng = _compute_artifact_stats_nogt(data_4x4_ng)
                    for metric_key in [
                        "derived.total_faqs",
                        "derived.total_summaries",
                        "derived.total_classifications",
                        "derived.total_keywords",
                        "derived.total_sentiments",
                        "derived.total_entities",
                    ]:
                        rows_ng.append(
                            {
                                "Metric": metric_key,
                                display_1x1_ng: stats_1x1_ng.get(metric_key, 0),
                                display_2x2_ng: stats_2x2_ng.get(metric_key, 0),
                                display_4x4_ng: stats_4x4_ng.get(metric_key, 0),
                            }
                        )

                    total_1x1_ng = sum(stats_1x1_ng.values())
                    total_2x2_ng = sum(stats_2x2_ng.values())
                    total_4x4_ng = sum(stats_4x4_ng.values())
                    rows_ng.append(
                        {
                            "Metric": "derived.total_artifacts_from_breakdown",
                            display_1x1_ng: total_1x1_ng,
                            display_2x2_ng: total_2x2_ng,
                            display_4x4_ng: total_4x4_ng,
                        }
                    )

                    columns_ng = [
                        "Metric",
                        display_1x1_ng,
                        display_2x2_ng,
                        display_4x4_ng,
                    ]
                    report_df_ng = pd.DataFrame(rows_ng, columns=columns_ng)

                    # Highlight key metrics similar to the GT-based tab
                    highlight_metrics_ng = {
                        "derived.artifact_generation_time_mm_ss",
                        "meta_data.genai_total_artifacts_generated",
                        "meta_data.total_atoms",
                        "derived.total_faqs",
                        "derived.total_summaries",
                        "derived.total_classifications",
                        "derived.total_keywords",
                        "derived.total_sentiments",
                        "derived.total_entities",
                        "derived.total_artifacts_from_breakdown",
                        "meta_data.gen_ai.model",
                        "meta_data.gen_ai.model_provider",
                        "meta_data.gen_ai.llm_id",
                        "meta_data.atomization.model_name",
                        "meta_data.atomization.model_provider",
                        "meta_data.atomization.prompt_version",
                        "meta_data.atomization_status",
                        "meta_data.genai_progress_status",
                    }

                    def _highlight_row_nogt(row):
                        if row["Metric"] in highlight_metrics_ng:
                            return ["background-color: #fff3cd; font-weight: bold;" for _ in row]
                        return ["" for _ in row]

                    st.subheader("CRD Metric Comparison (No Ground Truth)")
                    styler_ng = report_df_ng.style.apply(_highlight_row_nogt, axis=1).set_properties(
                        subset=["Metric"], **{"width": "300px"}
                    )
                    if REPORTING_MODE == "Detailed":
                        st.dataframe(styler_ng, use_container_width=True)
                    else:
                        compact_report_df_ng = report_df_ng[report_df_ng["Metric"].isin(highlight_metrics_ng)].copy()
                        if compact_report_df_ng.empty:
                            compact_report_df_ng = report_df_ng.head(25)
                        st.markdown("#### Key Metric Snapshot")
                        st.dataframe(compact_report_df_ng, use_container_width=True)
                        with st.expander("Show full metric comparison table"):
                            st.dataframe(styler_ng, use_container_width=True)

                    # --- Language detection and statistics (document level) ---
                    text_1x1_ng = extract_all_text(data_1x1_ng)
                    text_2x2_ng = extract_all_text(data_2x2_ng)
                    text_4x4_ng = extract_all_text(data_4x4_ng)
                    lang_1x1_ng = get_language_stats(text_1x1_ng)
                    lang_2x2_ng = get_language_stats(text_2x2_ng)
                    lang_4x4_ng = get_language_stats(text_4x4_ng)

                    # Build both per-system tables for the UI and a combined
                    # table for Excel export.
                    st.subheader("Language Statistics (Document Level)")
                    c1_ng, c2_ng, c3_ng = st.columns(3)
                    with c1_ng:
                        st.markdown(f"**{display_1x1_ng}**")
                        df_1x1_ng = language_stats_to_df(lang_1x1_ng)
                        st.dataframe(df_1x1_ng, use_container_width=True)
                    with c2_ng:
                        st.markdown(f"**{display_2x2_ng}**")
                        df_2x2_ng = language_stats_to_df(lang_2x2_ng)
                        st.dataframe(df_2x2_ng, use_container_width=True)
                    with c3_ng:
                        st.markdown(f"**{display_4x4_ng}**")
                        df_4x4_ng = language_stats_to_df(lang_4x4_ng)
                        st.dataframe(df_4x4_ng, use_container_width=True)

                    lang_rows_ng = []
                    for system_name, df_lang in [
                        (display_1x1_ng, df_1x1_ng),
                        (display_2x2_ng, df_2x2_ng),
                        (display_4x4_ng, df_4x4_ng),
                    ]:
                        if df_lang is None or df_lang.empty:
                            continue
                        for _, row_lang in df_lang.iterrows():
                            lang_rows_ng.append(
                                {
                                    "System": system_name,
                                    "Language": row_lang.get("Language"),
                                    "Code": row_lang.get("Code"),
                                    "Percentage": row_lang.get("Percentage"),
                                }
                            )

                    lang_df_ng = (
                        pd.DataFrame(lang_rows_ng, columns=["System", "Language", "Code", "Percentage"])
                        if lang_rows_ng
                        else pd.DataFrame(columns=["System", "Language", "Code", "Percentage"])
                    )

                    # --- Quality evaluation using per-topic 'context' ---

                    def _gather_artifacts_with_context(data):
                        topics = data.get("topics", []) if isinstance(data, dict) else []
                        summaries = []
                        faqs = []
                        keywords = []
                        entities = []
                        classifications = []
                        for t in topics:
                            if not isinstance(t, dict):
                                continue
                            ctx = t.get("context")
                            summary_text = extract_primary_summary_from_topic(t)
                            if summary_text:
                                summaries.append((ctx, summary_text))

                            for faq_text in extract_faq_texts_from_topic(t):
                                if faq_text:
                                    faqs.append((ctx, faq_text))

                            for word in extract_keywords_from_topic(t):
                                if word:
                                    keywords.append((ctx, str(word)))

                            for label in extract_entities_from_topic(t):
                                if label:
                                    entities.append((ctx, str(label)))

                            for name in extract_classifications_from_topic(t):
                                if name:
                                    classifications.append((ctx, str(name)))
                        return summaries, faqs, keywords, entities, classifications

                    def _gather_summary_components_with_context(data):
                        """Extract individual summary components with their context.
                        Returns list of (ctx, component_name, component_text) tuples."""
                        components = []
                        topics = data.get("topics", []) if isinstance(data, dict) else []
                        for t in topics:
                            if not isinstance(t, dict):
                                continue
                            ctx = t.get("context")
                            
                            for comp_name, comp_text in extract_summary_components_from_topic(t):
                                if comp_text and isinstance(comp_text, str) and comp_text.strip() and ctx:
                                    components.append((ctx, comp_name, comp_text.strip()))
                        return components

                    dims_ng = BASE_DIMS

                    def _score_items_with_topic_context(items):
                        if not items:
                            return {d: 0.0 for d in dims_ng}, 0.0, "No artifacts or context", 0
                        acc = {d: [] for d in dims_ng}
                        count = 0
                        for ctx, text in items:
                            if not ctx or not str(ctx).strip() or not text or not str(text).strip():
                                continue
                            scores = evaluate_summary_heuristic(ctx, text)
                            for d in dims_ng:
                                acc[d].append(scores.get(d, 0.0))
                            count += 1
                        if not count:
                            return {d: 0.0 for d in dims_ng}, 0.0, "No artifacts with usable context", 0
                        avg_scores = {}
                        for d in dims_ng:
                            vals = acc[d]
                            avg_scores[d] = _median_or_zero(vals, ndigits=2) if vals else 0.0
                        overall = _median_or_zero(list(avg_scores.values()), ndigits=2)
                        label = "Needs review" if overall < 3.0 else ("Excellent" if overall >= 4.5 else "Acceptable")
                        return avg_scores, overall, label, count

                    systems_ng = [
                        (display_1x1_ng, data_1x1_ng),
                        (display_2x2_ng, data_2x2_ng),
                        (display_4x4_ng, data_4x4_ng),
                    ]

                    summary_rows_ng = []
                    faq_rows_ng = []
                    keyword_rows_ng = []
                    entity_rows_ng = []
                    classification_rows_ng = []

                    for sys_name, data_obj in systems_ng:
                        summaries_i, faqs_i, kw_i, ent_i, cls_i = _gather_artifacts_with_context(data_obj)

                        # Evaluate individual summary components
                        summary_components_i = _gather_summary_components_with_context(data_obj)
                        for ctx, comp_name, comp_text in summary_components_i:
                            if ctx and str(ctx).strip() and comp_text and str(comp_text).strip():
                                sum_scores = evaluate_summary_heuristic(ctx, comp_text)
                                sum_overall, sum_label = classify_summary_auto(sum_scores)
                            else:
                                sum_scores = {d: 0.0 for d in dims_ng}
                                sum_overall, sum_label = 0.0, "No Context or Summary"
                            
                            summary_rows_ng.append(
                                {
                                    "System": sys_name,
                                    "Component": comp_name,
                                    **sum_scores,
                                    "Overall": sum_overall,
                                    "Classification": sum_label,
                                    **evaluate_summary_additional_metrics(str(ctx), comp_text),
                                    "_summary_text": comp_text,
                                }
                            )

                        faq_scores, faq_overall, faq_label, faq_count = _score_items_with_topic_context(faqs_i)
                        faq_rows_ng.append(
                            {
                                "System": sys_name,
                                "FAQ_Count": faq_count,
                                **faq_scores,
                                "Overall": faq_overall,
                                "Classification": faq_label,
                                **evaluate_faq_additional_metrics([t for _, t in faqs_i], "\n".join([str(c) for c, _ in faqs_i])),
                            }
                        )

                        kw_scores, kw_overall, kw_label, kw_count = _score_items_with_topic_context(kw_i)
                        keyword_rows_ng.append(
                            {
                                "System": sys_name,
                                "Keyword_Count": kw_count,
                                **kw_scores,
                                "Overall": kw_overall,
                                "Classification": kw_label,
                                **evaluate_keyword_additional_metrics(
                                    [t for _, t in kw_i],
                                    "\n".join([str(c) for c, _ in kw_i]),
                                    top_k=keyword_top_k,
                                ),
                            }
                        )

                        ent_scores, ent_overall, ent_label, ent_count = _score_items_with_topic_context(ent_i)
                        entity_rows_ng.append(
                            {
                                "System": sys_name,
                                "Entity_Count": ent_count,
                                **ent_scores,
                                "Overall": ent_overall,
                                "Classification": ent_label,
                                **entity_label_metrics(),
                            }
                        )

                        cls_scores, cls_overall, cls_label, cls_count = _score_items_with_topic_context(cls_i)
                        classification_rows_ng.append(
                            {
                                "System": sys_name,
                                "Classification_Count": cls_count,
                                **cls_scores,
                                "Overall": cls_overall,
                                "Classification": cls_label,
                                **classification_label_metrics(),
                            }
                        )

                    if summary_rows_ng:
                        summary_df_ng = _add_consistency_scores(
                            pd.DataFrame(summary_rows_ng), ["Component"]
                        )
                        render_quality_table(
                            "#### Overall Summary Quality vs Topic Context (automatic, 1–5 scale)",
                            "Summary",
                            "nogt_summary",
                            summary_df_ng,
                            outlier_label="No-GT Summary",
                            outlier_group_cols=["System", "Component"],
                        )

                    if faq_rows_ng:
                        faq_df_ng = pd.DataFrame(faq_rows_ng)
                        render_quality_table(
                            "#### Overall FAQ Quality vs Topic Context (automatic, 1–5 scale)",
                            "FAQ",
                            "nogt_faq",
                            faq_df_ng,
                            outlier_label="No-GT FAQ",
                            outlier_group_cols=["System"],
                        )

                    if keyword_rows_ng:
                        kw_df_ng = pd.DataFrame(keyword_rows_ng)
                        render_quality_table(
                            "#### Overall Keyword Quality vs Topic Context (automatic, 1–5 scale)",
                            "Keyword",
                            "nogt_keyword",
                            kw_df_ng,
                            outlier_label="No-GT Keyword",
                            outlier_group_cols=["System"],
                        )

                    if entity_rows_ng:
                        ent_df_ng = pd.DataFrame(entity_rows_ng)
                        render_quality_table(
                            "#### Overall Entity Quality vs Topic Context (automatic, 1–5 scale)",
                            "Entity",
                            "nogt_entity",
                            ent_df_ng,
                            outlier_label="No-GT Entity",
                            outlier_group_cols=["System"],
                        )

                    if classification_rows_ng:
                        cls_df_ng = pd.DataFrame(classification_rows_ng)
                        render_quality_table(
                            "#### Overall Classification Quality vs Topic Context (automatic, 1–5 scale)",
                            "Classification",
                            "nogt_classification",
                            cls_df_ng,
                            outlier_label="No-GT Classification",
                            outlier_group_cols=["System"],
                        )

                    analysis_tables_ng = render_quality_analysis_section(
                        summary_df=summary_df_ng if summary_rows_ng else None,
                        faq_df=faq_df_ng if faq_rows_ng else None,
                        keyword_df=kw_df_ng if keyword_rows_ng else None,
                        entity_df=ent_df_ng if entity_rows_ng else None,
                        classification_df=cls_df_ng if classification_rows_ng else None,
                        sentiment_df=None,
                        title="No-GT Quality Summary and Comparative Analysis",
                    )

                    # --- Download full no-GT evaluation report as Excel ---
                    try:
                        excel_sheets_ng = {"CRD_Metrics_No_GT": report_df_ng}
                        # Add combined language statistics if available
                        if not lang_df_ng.empty:
                            excel_sheets_ng["Language_Stats_No_GT"] = lang_df_ng
                        if summary_rows_ng:
                            excel_sheets_ng["Summary_Quality_No_GT"] = summary_df_ng
                        if faq_rows_ng:
                            excel_sheets_ng["FAQ_Quality_No_GT"] = faq_df_ng
                        if keyword_rows_ng:
                            excel_sheets_ng["Keyword_Quality_No_GT"] = kw_df_ng
                        if entity_rows_ng:
                            excel_sheets_ng["Entity_Quality_No_GT"] = ent_df_ng
                        if classification_rows_ng:
                            excel_sheets_ng["Classification_Quality_No_GT"] = cls_df_ng
                        if analysis_tables_ng.get("component_scores") is not None and not analysis_tables_ng["component_scores"].empty:
                            excel_sheets_ng["Analysis_ComponentScores"] = analysis_tables_ng["component_scores"]
                        if analysis_tables_ng.get("component_winners") is not None and not analysis_tables_ng["component_winners"].empty:
                            excel_sheets_ng["Analysis_ComponentBest"] = analysis_tables_ng["component_winners"]
                        if analysis_tables_ng.get("section_scorecard") is not None and not analysis_tables_ng["section_scorecard"].empty:
                            excel_sheets_ng["Analysis_SectionScores"] = analysis_tables_ng["section_scorecard"]
                        if analysis_tables_ng.get("section_winners") is not None and not analysis_tables_ng["section_winners"].empty:
                            excel_sheets_ng["Analysis_SectionBest"] = analysis_tables_ng["section_winners"]
                        if analysis_tables_ng.get("setup_ranking") is not None and not analysis_tables_ng["setup_ranking"].empty:
                            excel_sheets_ng["Analysis_SetupRanking"] = analysis_tables_ng["setup_ranking"]

                        buffer_ng = BytesIO()
                        with pd.ExcelWriter(buffer_ng, engine="xlsxwriter") as writer:
                            for sheet_name, df in excel_sheets_ng.items():
                                df.to_excel(writer, sheet_name=sheet_name, index=False)
                        buffer_ng.seek(0)

                        st.download_button(
                            label="Download no-GT report (.xlsx)",
                            data=buffer_ng.getvalue(),
                            file_name=f"{crd_name_ng}_dell_configuration_evaluation_no_gt.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        )
                    except Exception as e:
                        st.error(f"Failed to generate no-GT Excel report: {e}")


