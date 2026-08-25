import streamlit as st
import json
import os
from datetime import datetime
from io import BytesIO
from collections import Counter
import nltk
from llm_router import LLMRouter
from pdf_processor import extract_text_from_pdf, extract_text_from_pdf_with_llm
from evaluation_metrics import evaluate_output

# Download required NLTK data for text analysis
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Load model information
MODELS_INFO_FILE = 'models_info.json'
def load_models_info():
    """Load model information including pricing and descriptions"""
    if os.path.exists(MODELS_INFO_FILE):
        with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

# Page configuration
st.set_page_config(
    page_title="Prompt Evaluation Framework",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data directory for storing prompts and configs
DATA_DIR = 'data'
PROMPTS_FILE = os.path.join(DATA_DIR, 'prompts.json')
API_KEYS_FILE = os.path.join(DATA_DIR, 'api_keys.json')
MODEL_CONFIGS_FILE = os.path.join(DATA_DIR, 'model_configs.json')
EVALUATIONS_FILE = os.path.join(DATA_DIR, 'evaluations.json')
EVALUATIONS_DIR = os.path.join(DATA_DIR, 'evaluations')  # Directory for individual evaluation files

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(EVALUATIONS_DIR, exist_ok=True)

def load_prompts():
    """Load saved prompts from file"""
    if os.path.exists(PROMPTS_FILE):
        with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_prompts(prompts):
    """Save prompts to file"""
    with open(PROMPTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(prompts, f, indent=2, ensure_ascii=False)

def load_api_keys():
    """Load API keys from file"""
    if os.path.exists(API_KEYS_FILE):
        with open(API_KEYS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "routellm": "",
        "openai": "",
        "anthropic": ""
    }

def save_api_keys(api_keys):
    """Save API keys to file"""
    with open(API_KEYS_FILE, 'w', encoding='utf-8') as f:
        json.dump(api_keys, f, indent=2, ensure_ascii=False)

def load_model_configs():
    """Load model configurations from file"""
    if os.path.exists(MODEL_CONFIGS_FILE):
        with open(MODEL_CONFIGS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_model_configs(configs):
    """Save model configurations to file"""
    with open(MODEL_CONFIGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(configs, f, indent=2, ensure_ascii=False)

def get_next_id(items):
    """Get the next available ID"""
    if not items:
        return 1
    return max(item.get('id', 0) for item in items) + 1

def load_evaluations():
    """Load all evaluation results"""
    if os.path.exists(EVALUATIONS_FILE):
        with open(EVALUATIONS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_evaluation(evaluation_data):
    """Save a single evaluation result with full metadata"""
    evaluations = load_evaluations()
    
    # Create evaluation entry
    evaluation_entry = {
        'id': get_next_id(evaluations),
        'timestamp': evaluation_data['timestamp'],
        'prompt': evaluation_data['prompt'],
        'prompt_name': evaluation_data.get('prompt_name', 'Unnamed Prompt'),
        'model_configs': evaluation_data['model_configs'],
        'results': evaluation_data['results'],
        'metadata': evaluation_data.get('metadata', {})
    }
    
    # Save individual evaluation file
    eval_filename = f"evaluation_{evaluation_entry['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    eval_filepath = os.path.join(EVALUATIONS_DIR, eval_filename)
    with open(eval_filepath, 'w', encoding='utf-8') as f:
        json.dump(evaluation_entry, f, indent=2, ensure_ascii=False)
    
    # Add file path to entry
    evaluation_entry['file_path'] = eval_filepath
    
    # Add to evaluations list
    evaluations.append(evaluation_entry)
    
    # Save evaluations index
    with open(EVALUATIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump(evaluations, f, indent=2, ensure_ascii=False)
    
    return evaluation_entry

def get_evaluation_by_id(eval_id):
    """Get a specific evaluation by ID"""
    evaluations = load_evaluations()
    return next((e for e in evaluations if e['id'] == eval_id), None)

def delete_evaluation(eval_id):
    """Delete an evaluation"""
    evaluations = load_evaluations()
    evaluation = next((e for e in evaluations if e['id'] == eval_id), None)
    
    if evaluation:
        # Delete individual file if exists
        if 'file_path' in evaluation and os.path.exists(evaluation['file_path']):
            try:
                os.remove(evaluation['file_path'])
            except:
                pass
        
        # Remove from list
        evaluations = [e for e in evaluations if e['id'] != eval_id]
        
        # Save updated list
        with open(EVALUATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(evaluations, f, indent=2, ensure_ascii=False)
        
        return True
    return False

def generate_score_summary(reference_text: str, response_text: str, metrics: dict) -> str:
    """
    Generate a contextual summary of evaluation scores based on the actual texts.
    
    Args:
        reference_text: The reference text
        response_text: The response text being evaluated
        metrics: Dictionary containing BLEU and ROUGE scores
    
    Returns:
        A formatted markdown string with contextual analysis
    """
    if metrics.get('error'):
        return f"⚠️ **Error:** {metrics.get('error')}"
    
    bleu = metrics.get('bleu', {})
    rouge = metrics.get('rouge', {})
    
    # Tokenize texts for analysis
    try:
        ref_tokens = nltk.word_tokenize(reference_text.lower())
        resp_tokens = nltk.word_tokenize(response_text.lower())
        
        ref_words = set(ref_tokens)
        resp_words = set(resp_tokens)
        
        # Calculate overlaps
        common_words = ref_words.intersection(resp_words)
        missing_words = ref_words - resp_words
        extra_words = resp_words - ref_words
        
        # Get word frequencies
        ref_word_freq = Counter(ref_tokens)
        resp_word_freq = Counter(resp_tokens)
        
        # Find common bigrams and trigrams
        ref_bigrams = set(nltk.bigrams(ref_tokens))
        resp_bigrams = set(nltk.bigrams(resp_tokens))
        common_bigrams = ref_bigrams.intersection(resp_bigrams)
        
        ref_trigrams = set(nltk.trigrams(ref_tokens))
        resp_trigrams = set(nltk.trigrams(resp_tokens))
        common_trigrams = ref_trigrams.intersection(resp_trigrams)
        
    except Exception as e:
        # Fallback to simple word splitting if tokenization fails
        ref_words = set(reference_text.lower().split())
        resp_words = set(response_text.lower().split())
        common_words = ref_words.intersection(resp_words)
        missing_words = ref_words - resp_words
        extra_words = resp_words - ref_words
        common_bigrams = set()
        common_trigrams = set()
    
    # Extract score values
    bleu1 = bleu.get('BLEU-1', 0) or 0
    bleu2 = bleu.get('BLEU-2', 0) or 0
    bleu3 = bleu.get('BLEU-3', 0) or 0
    bleu4 = bleu.get('BLEU-4', 0) or 0
    bleu_overall = bleu.get('BLEU', 0) or 0
    
    r1_key = 'ROUGE-1' if 'ROUGE-1' in rouge else ('ROUGE1' if 'ROUGE1' in rouge else None)
    r2_key = 'ROUGE-2' if 'ROUGE-2' in rouge else ('ROUGE2' if 'ROUGE2' in rouge else None)
    rl_key = 'ROUGE-L' if 'ROUGE-L' in rouge else ('ROUGEL' if 'ROUGEL' in rouge else None)
    
    r1_f1 = rouge.get(r1_key, {}).get('f1', 0) if r1_key else 0
    r1_precision = rouge.get(r1_key, {}).get('precision', 0) if r1_key else 0
    r1_recall = rouge.get(r1_key, {}).get('recall', 0) if r1_key else 0
    
    r2_f1 = rouge.get(r2_key, {}).get('f1', 0) if r2_key else 0
    r2_precision = rouge.get(r2_key, {}).get('precision', 0) if r2_key else 0
    r2_recall = rouge.get(r2_key, {}).get('recall', 0) if r2_key else 0
    
    rl_f1 = rouge.get(rl_key, {}).get('f1', 0) if rl_key else 0
    rl_precision = rouge.get(rl_key, {}).get('precision', 0) if rl_key else 0
    rl_recall = rouge.get(rl_key, {}).get('recall', 0) if rl_key else 0
    
    # Build summary
    summary_parts = []
    
    # Overall assessment
    avg_score = (bleu4 + r1_f1 + rl_f1) / 3 if (bleu4 + r1_f1 + rl_f1) > 0 else 0
    if avg_score >= 0.7:
        overall_status = "🟢 **Excellent Match**"
        overall_desc = "The response text shows very high similarity to the reference text."
    elif avg_score >= 0.5:
        overall_status = "🟡 **Good Match**"
        overall_desc = "The response text shows strong similarity with some differences."
    elif avg_score >= 0.3:
        overall_status = "🟠 **Fair Match**"
        overall_desc = "The response text shows moderate similarity but is missing some key content."
    elif avg_score >= 0.1:
        overall_status = "🔴 **Poor Match**"
        overall_desc = "The response text shows limited similarity to the reference text."
    else:
        overall_status = "⚫ **Very Poor Match**"
        overall_desc = "The response text shows minimal to no similarity with the reference text."
    
    summary_parts.append(f"### {overall_status}")
    summary_parts.append(f"{overall_desc}\n")
    
    # Text length analysis
    ref_len = len(reference_text)
    resp_len = len(response_text)
    len_diff = abs(ref_len - resp_len)
    len_ratio = resp_len / ref_len if ref_len > 0 else 0
    
    summary_parts.append("#### 📏 Text Length Analysis")
    summary_parts.append(f"- **Reference text length:** {ref_len:,} characters")
    summary_parts.append(f"- **Response text length:** {resp_len:,} characters")
    
    if len_ratio > 1.2:
        summary_parts.append(f"- **Observation:** Response is {len_ratio:.1f}x longer than reference. This may indicate the response contains additional information not in the reference, which could lower precision scores.")
    elif len_ratio < 0.8:
        summary_parts.append(f"- **Observation:** Response is {1/len_ratio:.1f}x shorter than reference. This may indicate missing content, which could lower recall scores.")
    else:
        summary_parts.append(f"- **Observation:** Response length is similar to reference ({len_ratio:.1f}x ratio), which is good for balanced precision and recall.")
    summary_parts.append("")
    
    # Word overlap analysis
    summary_parts.append("#### 🔤 Word Overlap Analysis")
    word_overlap_ratio = len(common_words) / len(ref_words) if len(ref_words) > 0 else 0
    summary_parts.append(f"- **Common words:** {len(common_words)} out of {len(ref_words)} unique words in reference ({word_overlap_ratio*100:.1f}%)")
    summary_parts.append(f"- **Missing words:** {len(missing_words)} words from reference not found in response")
    summary_parts.append(f"- **Extra words:** {len(extra_words)} words in response not in reference")
    
    if word_overlap_ratio >= 0.8:
        summary_parts.append(f"- **Impact on scores:** High word overlap ({word_overlap_ratio*100:.1f}%) contributes to strong BLEU-1 and ROUGE-1 scores.")
    elif word_overlap_ratio >= 0.5:
        summary_parts.append(f"- **Impact on scores:** Moderate word overlap ({word_overlap_ratio*100:.1f}%) supports decent BLEU-1 and ROUGE-1 scores, but missing words may lower recall.")
    else:
        summary_parts.append(f"- **Impact on scores:** Low word overlap ({word_overlap_ratio*100:.1f}%) significantly impacts BLEU-1 and ROUGE-1 scores. Many reference words are missing from the response.")
    
    # Show examples of missing/extra words (limit to 10)
    if missing_words and len(missing_words) <= 20:
        missing_examples = list(missing_words)[:10]
        summary_parts.append(f"- **Example missing words:** {', '.join(missing_examples)}")
    elif missing_words:
        missing_examples = list(missing_words)[:10]
        summary_parts.append(f"- **Example missing words (showing 10 of {len(missing_words)}):** {', '.join(missing_examples)}")
    
    summary_parts.append("")
    
    # Phrase matching analysis
    summary_parts.append("#### 🔗 Phrase Matching Analysis")
    if len(ref_bigrams) > 0:
        bigram_overlap = len(common_bigrams) / len(ref_bigrams)
        summary_parts.append(f"- **Bigram (2-word phrase) overlap:** {len(common_bigrams)} out of {len(ref_bigrams)} bigrams ({bigram_overlap*100:.1f}%)")
        if bigram_overlap >= 0.6:
            summary_parts.append(f"  - This strong bigram overlap contributes to higher BLEU-2 and ROUGE-2 scores.")
        elif bigram_overlap >= 0.3:
            summary_parts.append(f"  - Moderate bigram overlap supports decent BLEU-2 and ROUGE-2 scores.")
        else:
            summary_parts.append(f"  - Low bigram overlap limits BLEU-2 and ROUGE-2 scores, indicating word order differences.")
    
    if len(ref_trigrams) > 0:
        trigram_overlap = len(common_trigrams) / len(ref_trigrams)
        summary_parts.append(f"- **Trigram (3-word phrase) overlap:** {len(common_trigrams)} out of {len(ref_trigrams)} trigrams ({trigram_overlap*100:.1f}%)")
        if trigram_overlap >= 0.5:
            summary_parts.append(f"  - Strong trigram overlap helps boost BLEU-3 and BLEU-4 scores.")
        elif trigram_overlap >= 0.2:
            summary_parts.append(f"  - Some trigram matches support moderate BLEU-3 and BLEU-4 scores.")
        else:
            summary_parts.append(f"  - Limited trigram overlap significantly constrains BLEU-3 and BLEU-4 scores.")
    summary_parts.append("")
    
    # BLEU score analysis
    summary_parts.append("#### 🎯 BLEU Score Analysis")
    
    if bleu1 >= 0.7:
        summary_parts.append(f"- **BLEU-1 ({bleu1:.4f}):** 🟢 Excellent - Most individual words from the reference appear in the response.")
    elif bleu1 >= 0.5:
        summary_parts.append(f"- **BLEU-1 ({bleu1:.4f}):** 🟡 Good - Many words match, but some reference words are missing.")
    elif bleu1 >= 0.3:
        summary_parts.append(f"- **BLEU-1 ({bleu1:.4f}):** 🟠 Fair - Moderate word overlap, but significant vocabulary differences.")
    else:
        summary_parts.append(f"- **BLEU-1 ({bleu1:.4f}):** 🔴 Poor - Limited word overlap between reference and response.")
    
    if bleu2 >= 0.5:
        summary_parts.append(f"- **BLEU-2 ({bleu2:.4f}):** Strong 2-word phrase matching indicates good word order preservation.")
    elif bleu2 >= 0.3:
        summary_parts.append(f"- **BLEU-2 ({bleu2:.4f}):** Moderate 2-word phrase matching, some word order differences present.")
    else:
        summary_parts.append(f"- **BLEU-2 ({bleu2:.4f}):** Limited 2-word phrase matching suggests word order or phrasing differences.")
    
    if bleu4 >= 0.5:
        summary_parts.append(f"- **BLEU-4 ({bleu4:.4f}):** 🟢 Strong - Excellent 4-word phrase matching, indicating high content similarity.")
    elif bleu4 >= 0.3:
        summary_parts.append(f"- **BLEU-4 ({bleu4:.4f}):** 🟡 Moderate - Some longer phrase matches, but not consistently.")
    elif bleu4 >= 0.1:
        summary_parts.append(f"- **BLEU-4 ({bleu4:.4f}):** 🟠 Low - Few 4-word phrase matches, indicating significant content or phrasing differences.")
    else:
        summary_parts.append(f"- **BLEU-4 ({bleu4:.4f}):** 🔴 Very Low - Minimal 4-word phrase matches, suggesting substantial differences in content or structure.")
    
    if bleu4 < bleu1 * 0.5:
        summary_parts.append(f"  - **Note:** BLEU-4 is much lower than BLEU-1 ({bleu4:.4f} vs {bleu1:.4f}), indicating that while individual words match, longer phrases do not. This suggests paraphrasing or structural differences.")
    elif bleu4 > bleu1 * 0.8:
        summary_parts.append(f"  - **Note:** BLEU-4 is close to BLEU-1, indicating consistent phrase-level matching, which is excellent.")
    
    summary_parts.append("")
    
    # ROUGE score analysis
    summary_parts.append("#### 📊 ROUGE Score Analysis")
    
    if r1_f1 >= 0.7:
        summary_parts.append(f"- **ROUGE-1 F1 ({r1_f1:.4f}):** 🟢 Excellent - Response captures most key information from reference.")
    elif r1_f1 >= 0.5:
        summary_parts.append(f"- **ROUGE-1 F1 ({r1_f1:.4f}):** 🟡 Good - Response captures majority of important information.")
    elif r1_f1 >= 0.3:
        summary_parts.append(f"- **ROUGE-1 F1 ({r1_f1:.4f}):** 🟠 Fair - Response captures some key information but misses details.")
    else:
        summary_parts.append(f"- **ROUGE-1 F1 ({r1_f1:.4f}):** 🔴 Poor - Response fails to capture much of the reference information.")
    
    if r1_precision > r1_recall * 1.3:
        summary_parts.append(f"  - **Precision ({r1_precision:.4f}) > Recall ({r1_recall:.4f}):** Response words are mostly correct but missing content (response may be too short or incomplete).")
    elif r1_recall > r1_precision * 1.3:
        summary_parts.append(f"  - **Recall ({r1_recall:.4f}) > Precision ({r1_precision:.4f}):** Response covers most reference content but includes extra/incorrect words (response may be too verbose or contain errors).")
    else:
        summary_parts.append(f"  - **Balanced Precision ({r1_precision:.4f}) and Recall ({r1_recall:.4f}):** Good balance between accuracy and completeness.")
    
    if r2_f1 >= 0.5:
        summary_parts.append(f"- **ROUGE-2 F1 ({r2_f1:.4f}):** Strong 2-word phrase coverage indicates good information preservation.")
    elif r2_f1 >= 0.3:
        summary_parts.append(f"- **ROUGE-2 F1 ({r2_f1:.4f}):** Moderate phrase coverage, some key phrases may be missing or rephrased.")
    else:
        summary_parts.append(f"- **ROUGE-2 F1 ({r2_f1:.4f}):** Limited phrase coverage suggests significant information loss or rephrasing.")
    
    if rl_f1 >= 0.6:
        summary_parts.append(f"- **ROUGE-L F1 ({rl_f1:.4f}):** 🟢 Strong - Good sentence structure and sequence matching, indicating coherent content flow.")
    elif rl_f1 >= 0.4:
        summary_parts.append(f"- **ROUGE-L F1 ({rl_f1:.4f}):** 🟡 Moderate - Some structural similarity, but sequence differences exist.")
    else:
        summary_parts.append(f"- **ROUGE-L F1 ({rl_f1:.4f}):** 🟠 Low - Limited structural similarity, suggesting different organization or content flow.")
    
    summary_parts.append("")
    
    # Key insights
    summary_parts.append("#### 💡 Key Insights")
    
    insights = []
    
    if bleu4 < 0.1 and bleu1 > 0.3:
        insights.append("While individual words match (BLEU-1), longer phrases don't (BLEU-4), suggesting the response uses similar vocabulary but different phrasing or structure.")
    
    if r1_recall < 0.3:
        insights.append("Low ROUGE-1 recall indicates the response is missing significant portions of the reference content.")
    
    if r1_precision < 0.3:
        insights.append("Low ROUGE-1 precision suggests the response contains many words not present in the reference.")
    
    if len(missing_words) > len(ref_words) * 0.4:
        insights.append(f"A large number of missing words ({len(missing_words)} out of {len(ref_words)}) significantly impacts recall-based metrics like ROUGE.")
    
    if len(extra_words) > len(resp_words) * 0.4:
        insights.append(f"Many extra words in the response ({len(extra_words)} out of {len(resp_words)}) may lower precision scores.")
    
    if bleu4 > 0.5 and rl_f1 > 0.5:
        insights.append("Strong BLEU-4 and ROUGE-L scores indicate excellent content and structural similarity.")
    
    if not insights:
        if avg_score >= 0.7:
            insights.append("Overall excellent match across all metrics, indicating high similarity between reference and response.")
        elif avg_score >= 0.5:
            insights.append("Good overall match with room for improvement in specific areas.")
        else:
            insights.append("Significant differences exist between reference and response across multiple metrics.")
    
    for insight in insights:
        summary_parts.append(f"- {insight}")
    
    return "\n".join(summary_parts)

def prepare_score_summary_export(reference_text: str, response_text: str, metrics: dict, summary_markdown: str) -> dict:
    """
    Structure score and summary data for export.
    """
    bleu = metrics.get('bleu', {})
    rouge = metrics.get('rouge', {})
    
    r1_key = 'ROUGE-1' if 'ROUGE-1' in rouge else ('ROUGE1' if 'ROUGE1' in rouge else None)
    rl_key = 'ROUGE-L' if 'ROUGE-L' in rouge else ('ROUGEL' if 'ROUGEL' in rouge else None)
    
    key_scores = {
        'BLEU': bleu.get('BLEU', 0),
        'BLEU-1': bleu.get('BLEU-1', 0),
        'BLEU-2': bleu.get('BLEU-2', 0),
        'BLEU-3': bleu.get('BLEU-3', 0),
        'BLEU-4': bleu.get('BLEU-4', 0),
        'ROUGE-1 F1': rouge.get(r1_key, {}).get('f1', 0) if r1_key else 0,
        'ROUGE-L F1': rouge.get(rl_key, {}).get('f1', 0) if rl_key else 0
    }
    
    return {
        'generated_at': datetime.now().isoformat(),
        'reference_text': reference_text,
        'response_text': response_text,
        'reference_length': len(reference_text),
        'response_length': len(response_text),
        'metrics': metrics,
        'key_scores': key_scores,
        'summary_markdown': summary_markdown
    }

def build_summary_text_document(summary_data: dict) -> str:
    """
    Create a human-readable text document from summary data.
    """
    lines = []
    lines.append("Prompt Evaluation Score Report")
    lines.append(f"Generated: {summary_data.get('generated_at', datetime.now().isoformat())}")
    lines.append("")
    
    lines.append("Key Scores")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    for metric, value in summary_data.get('key_scores', {}).items():
        lines.append(f"| {metric} | {value:.4f} |")
    lines.append("")
    
    lines.append("Summary & Analysis")
    lines.append(summary_data.get('summary_markdown', '').replace("### ", "\n### "))
    lines.append("")
    
    lines.append("Reference Text (excerpt)")
    ref_text = summary_data.get('reference_text', '')
    lines.append(ref_text if len(ref_text) <= 2000 else ref_text[:2000] + " ...")
    lines.append("")
    
    lines.append("Response Text (excerpt)")
    resp_text = summary_data.get('response_text', '')
    lines.append(resp_text if len(resp_text) <= 2000 else resp_text[:2000] + " ...")
    lines.append("")
    
    return "\n".join(lines)

def build_markdown_report(summary_data: dict) -> str:
    """
    Build a markdown-formatted report that pastes cleanly into Excel cells.
    """
    lines = []
    generated_at = summary_data.get('generated_at', datetime.now().isoformat())
    lines.append(f"**Prompt Evaluation Report** ({generated_at})")
    lines.append("")
    
    lines.append("**Key Scores**")
    lines.append("| Metric | Score |")
    lines.append("|--------|-------|")
    for metric, value in summary_data.get('key_scores', {}).items():
        lines.append(f"| {metric} | {value:.4f} |")
    lines.append("")
    
    lines.append("**Summary & Analysis**")
    summary_md = summary_data.get('summary_markdown', '').strip()
    lines.append(summary_md if summary_md else "_No summary available._")
    lines.append("")
    
    lines.append("**Reference Text (excerpt)**")
    ref_text = summary_data.get('reference_text', '')
    lines.append(ref_text if len(ref_text) <= 2000 else ref_text[:2000] + " ...")
    lines.append("")
    
    lines.append("**Response Text (excerpt)**")
    resp_text = summary_data.get('response_text', '')
    lines.append(resp_text if len(resp_text) <= 2000 else resp_text[:2000] + " ...")
    lines.append("")
    
    lines.append("_Copy this block into a single Excel cell (Alt+Enter to preserve line breaks)._")
    
    return "\n".join(lines)

# Initialize session state
if 'prompts' not in st.session_state:
    st.session_state.prompts = load_prompts()
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = load_api_keys()
if 'model_configs' not in st.session_state:
    st.session_state.model_configs = load_model_configs()
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = None
if 'text_evaluation_results' not in st.session_state:
    st.session_state.text_evaluation_results = None
if 'models_info' not in st.session_state:
    st.session_state.models_info = load_models_info()

# Main title
st.title("🤖 Prompt Evaluation Framework")
st.markdown("Evaluate prompts across multiple AI models")

# Create tabs for different sections
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📝 Prompts", "🔑 API Keys", "⚙️ Model Config", "🚀 Evaluate", "📊 Results History", "📝 Evaluate Text"])

# ========== TAB 1: PROMPTS ==========
with tab1:
    st.header("Enter or Upload Prompt")
    
    # Initialize session state for current prompt
    if 'current_prompt' not in st.session_state:
        st.session_state.current_prompt = ""
    if 'current_name' not in st.session_state:
        st.session_state.current_name = ""
    
    # Input method selection
    input_method = st.radio(
        "Choose input method:",
        ["Text Input", "File Upload"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    if input_method == "Text Input":
        # Text input method
        prompt_name = st.text_input(
            "Prompt Name (optional)",
            value=st.session_state.current_name,
            placeholder="Enter a name for this prompt"
        )
        
        prompt_text = st.text_area(
            "Prompt Text",
            value=st.session_state.current_prompt,
            height=300,
            placeholder="Enter your prompt here..."
        )
        
        # Clear button to reset
        if st.button("Clear", use_container_width=True):
            st.session_state.current_prompt = ""
            st.session_state.current_name = ""
            st.rerun()
    
    else:
        # File upload method
        uploaded_file = st.file_uploader(
            "Upload Prompt File",
            type=['txt', 'md', 'json'],
            help="Upload a .txt, .md, or .json file containing your prompt"
        )
        
        if uploaded_file is not None:
            # Read file content
            try:
                file_content = uploaded_file.read().decode('utf-8')
                prompt_text = st.text_area(
                    "Prompt Text (from file)",
                    value=file_content,
                    height=300
                )
                
                # Auto-fill name from filename if not set
                if not st.session_state.current_name:
                    prompt_name = st.text_input(
                        "Prompt Name (optional)",
                        value=uploaded_file.name.rsplit('.', 1)[0],
                        placeholder="Enter a name for this prompt"
                    )
                else:
                    prompt_name = st.text_input(
                        "Prompt Name (optional)",
                        value=st.session_state.current_name,
                        placeholder="Enter a name for this prompt"
                    )
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                prompt_text = ""
                prompt_name = ""
        else:
            prompt_text = ""
            prompt_name = st.text_input(
                "Prompt Name (optional)",
                value=st.session_state.current_name,
                placeholder="Enter a name for this prompt"
            )
    
    # Save prompt button
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("💾 Save Prompt", type="primary", use_container_width=True):
            if prompt_text and prompt_text.strip():
                # Generate name if not provided
                if not prompt_name or not prompt_name.strip():
                    prompt_name = f"Prompt {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                else:
                    prompt_name = prompt_name.strip()
                
                # Create new prompt entry
                new_prompt = {
                    'id': get_next_id(st.session_state.prompts),
                    'name': prompt_name,
                    'prompt': prompt_text.strip(),
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat()
                }
                
                # Add to prompts list
                st.session_state.prompts.append(new_prompt)
                save_prompts(st.session_state.prompts)
                
                st.success(f"✅ Prompt '{prompt_name}' saved successfully!")
                st.session_state.current_prompt = prompt_text
                st.session_state.current_name = prompt_name
                st.rerun()
            else:
                st.error("Please enter a prompt before saving.")
    
    # Display current prompt stats
    if prompt_text:
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Characters", len(prompt_text))
        with col2:
            st.metric("Words", len(prompt_text.split()))
        with col3:
            st.metric("Lines", len(prompt_text.split('\n')))
    
    # Display saved prompts
    st.divider()
    st.subheader("📚 Saved Prompts")
    if st.session_state.prompts:
        for prompt in st.session_state.prompts:
            with st.expander(f"📝 {prompt['name']}", expanded=False):
                st.caption(f"Created: {datetime.fromisoformat(prompt['created_at']).strftime('%Y-%m-%d %H:%M')}")
                st.text_area(
                    "Preview",
                    prompt['prompt'][:200] + "..." if len(prompt['prompt']) > 200 else prompt['prompt'],
                    key=f"preview_{prompt['id']}",
                    height=100,
                    disabled=True
                )
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Load", key=f"load_{prompt['id']}", use_container_width=True):
                        st.session_state.current_prompt = prompt['prompt']
                        st.session_state.current_name = prompt['name']
                        st.rerun()
                with col2:
                    if st.button("Delete", key=f"delete_{prompt['id']}", use_container_width=True, type="secondary"):
                        st.session_state.prompts = [p for p in st.session_state.prompts if p['id'] != prompt['id']]
                        save_prompts(st.session_state.prompts)
                        st.rerun()
    else:
        st.info("No saved prompts yet. Save your first prompt to get started!")

# ========== TAB 2: API KEYS ==========
with tab2:
    st.header("🔑 API Key Management")
    st.markdown("Manage API keys for different LLM providers. Keys are stored locally and encrypted.")
    
    st.info("💡 **RouteLLM/AbacusAI**: Get your API key from [AbacusAI](https://abacus.ai). RouteLLM provides unified access to multiple models.")
    
    # RouteLLM/AbacusAI API Key
    st.subheader("RouteLLM / AbacusAI")
    routellm_key = st.text_input(
        "RouteLLM API Key",
        value=st.session_state.api_keys.get("routellm", ""),
        type="password",
        help="Enter your RouteLLM API key from AbacusAI",
        key="routellm_key_input"
    )
    
    st.divider()
    
    # Direct Provider API Keys
    st.subheader("Direct Provider API Keys (Optional)")
    st.markdown("You can also use direct API keys for specific providers as a fallback.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### OpenAI")
        openai_key = st.text_input(
            "OpenAI API Key",
            value=st.session_state.api_keys.get("openai", ""),
            type="password",
            help="Enter your OpenAI API key",
            key="openai_key_input"
        )
    
    with col2:
        st.markdown("#### Anthropic")
        anthropic_key = st.text_input(
            "Anthropic API Key",
            value=st.session_state.api_keys.get("anthropic", ""),
            type="password",
            help="Enter your Anthropic API key",
            key="anthropic_key_input"
        )
    
    # Save API keys button
    if st.button("💾 Save API Keys", type="primary", use_container_width=True):
        st.session_state.api_keys = {
            "routellm": routellm_key,
            "openai": openai_key,
            "anthropic": anthropic_key
        }
        save_api_keys(st.session_state.api_keys)
        st.success("✅ API keys saved successfully!")
        st.rerun()
    
    # Show status
    st.divider()
    st.subheader("API Key Status")
    col1, col2, col3 = st.columns(3)
    with col1:
        status = "✅ Configured" if st.session_state.api_keys.get("routellm") else "❌ Not Set"
        st.metric("RouteLLM", status)
    with col2:
        status = "✅ Configured" if st.session_state.api_keys.get("openai") else "❌ Not Set"
        st.metric("OpenAI", status)
    with col3:
        status = "✅ Configured" if st.session_state.api_keys.get("anthropic") else "❌ Not Set"
        st.metric("Anthropic", status)

# ========== TAB 3: MODEL CONFIGURATION ==========
with tab3:
    st.header("⚙️ Model Configuration")
    st.markdown("Configure models for evaluation. Set parameters like temperature and max_tokens.")
    
    # Add new model configuration
    with st.expander("➕ Add New Model Configuration", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            provider = st.selectbox(
                "Provider",
                ["routellm", "openai", "anthropic"],
                help="Choose the API provider"
            )
            
            # Model selection based on provider
            if provider == "routellm":
                # All models available through RouteLLM
                model_options = [
                    # RouteLLM Router
                    "route-llm",
                    # OpenAI Models
                    "gpt-4o-2024-11-20",
                    "gpt-4o-mini",
                    "o4-mini",
                    "o3-pro",
                    "o3",
                    "o3-mini",
                    "gpt-4.1",
                    "gpt-4.1-mini",
                    "gpt-4.1-nano",
                    "gpt-5",
                    "gpt-5-mini",
                    "gpt-5-nano",
                    "gpt-5.1",
                    "gpt-5.1-chat-latest",
                    "openai/gpt-oss-120b",
                    # Anthropic Models
                    "claude-3-7-sonnet-20250219",
                    "claude-sonnet-4-20250514",
                    "claude-opus-4-20250514",
                    "claude-opus-4-1-20250805",
                    "claude-sonnet-4-5-20250929",
                    "claude-haiku-4-5-20251001",
                    # Meta Llama Models
                    "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
                    "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
                    "meta-llama/Meta-Llama-3.1-70B-Instruct",
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    "llama-3.3-70b-versatile",
                    # Google Gemini Models
                    "gemini-2.0-flash-001",
                    "gemini-2.0-pro-exp-02-05",
                    "gemini-2.5-pro",
                    "gemini-2.5-flash",
                    # Qwen Models
                    "qwen-2.5-coder-32b",
                    "Qwen/Qwen2.5-72B-Instruct",
                    "Qwen/QwQ-32B",
                    "Qwen/Qwen3-235B-A22B-Instruct-2507",
                    "Qwen/Qwen3-32B",
                    "qwen/qwen3-coder-480b-a35b-instruct",
                    "qwen/qwen3-Max",
                    # xAI Grok Models
                    "grok-4-0709",
                    "grok-4-fast-non-reasoning",
                    "grok-code-fast-1",
                    # Moonshot Models
                    "moonshotai/kimi-k2-instruct",
                    # DeepSeek Models
                    "deepseek/deepseek-v3.1",
                    "deepseek-ai/DeepSeek-V3.1-Terminus",
                    "deepseek-ai/DeepSeek-R1",
                    "deepseek-ai/DeepSeek-V3.2-Exp"
                ]
            elif provider == "openai":
                model_options = [
                    "gpt-4o-2024-11-20",
                    "gpt-4o-mini",
                    "gpt-4-turbo-preview",
                    "gpt-3.5-turbo"
                ]
            else:  # anthropic
                model_options = [
                    "claude-3-7-sonnet-20250219",
                    "claude-sonnet-4-20250514",
                    "claude-opus-4-20250514",
                    "claude-opus-4-1-20250805",
                    "claude-sonnet-4-5-20250929",
                    "claude-haiku-4-5-20251001"
                ]
            
            model = st.selectbox("Model", model_options)
            
            # Display model information if available
            if model and model in st.session_state.models_info:
                model_info = st.session_state.models_info[model]
                with st.expander("ℹ️ Model Information", expanded=False):
                    st.markdown(f"**{model_info.get('name', model)}**")
                    if model_info.get('description'):
                        st.caption(model_info['description'])
                    if model_info.get('input_price') is not None:
                        st.markdown(f"💰 **Pricing:** ${model_info['input_price']}/M input, ${model_info['output_price']}/M output")
        
        with col2:
            config_name = st.text_input(
                "Configuration Name",
                placeholder="e.g., GPT-4 Creative",
                help="Give this configuration a descriptive name"
            )
        
        # Model parameters
        st.markdown("#### Model Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=0.7,
                step=0.1,
                help="Controls randomness. Lower = more deterministic."
            )
        
        with col2:
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=1,
                max_value=8000,
                value=1000,
                step=100,
                help="Maximum tokens to generate."
            )
        
        if st.button("➕ Add Configuration", type="primary", use_container_width=True):
            if not config_name:
                st.error("Please enter a configuration name.")
            else:
                new_config = {
                    'id': get_next_id(st.session_state.model_configs),
                    'name': config_name,
                    'provider': provider,
                    'model': model,
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'created_at': datetime.now().isoformat()
                }
                st.session_state.model_configs.append(new_config)
                save_model_configs(st.session_state.model_configs)
                st.success(f"✅ Configuration '{config_name}' added!")
                st.rerun()
    
    # Display existing configurations
    st.divider()
    st.subheader("📋 Saved Model Configurations")
    
    if st.session_state.model_configs:
        for config in st.session_state.model_configs:
            with st.expander(f"⚙️ {config['name']} ({config['provider']} - {config['model']})", expanded=False):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.json({
                        "Provider": config['provider'],
                        "Model": config['model'],
                        "Temperature": config['temperature'],
                        "Max Tokens": config['max_tokens']
                    })
                with col2:
                    if st.button("Delete", key=f"del_config_{config['id']}", use_container_width=True, type="secondary"):
                        st.session_state.model_configs = [c for c in st.session_state.model_configs if c['id'] != config['id']]
                        save_model_configs(st.session_state.model_configs)
                        st.rerun()
    else:
        st.info("No model configurations yet. Add your first configuration above!")

# ========== TAB 4: EVALUATE ==========
with tab4:
    st.header("🚀 Evaluate Prompt")
    st.markdown("Send your prompt to multiple configured models and compare responses.")
    
    # Reference text section (for BLEU/ROUGE evaluation)
    st.subheader("📄 Reference Text (for Evaluation Metrics)")
    st.markdown("Upload a PDF or enter reference text to evaluate model outputs using BLEU/ROUGE scores.")
    
    ref_input_method = st.radio(
        "Reference Input Method:",
        ["PDF Upload", "Text Input", "None"],
        horizontal=True,
        key="ref_input_method"
    )
    
    reference_text = None
    pdf_uploaded = False
    
    if ref_input_method == "PDF Upload":
        uploaded_pdf = st.file_uploader(
            "Upload PDF File",
            type=['pdf'],
            help="Upload a PDF file to extract reference text for evaluation",
            key="pdf_uploader"
        )
        
        if uploaded_pdf is not None:
            # Choose extraction method
            extraction_method = st.radio(
                "Extraction Method:",
                ["Traditional (PyPDF2/pdfplumber)", "LLM Extraction"],
                horizontal=True,
                key="extraction_method"
            )
            
            if extraction_method == "LLM Extraction":
                # Select LLM for extraction
                if not st.session_state.model_configs:
                    st.warning("⚠️ No model configurations found. Please configure models in the 'Model Config' tab first.")
                    reference_text = None
                else:
                    extraction_model_options = {f"{c['name']} ({c['provider']} - {c['model']})": c for c in st.session_state.model_configs}
                    selected_extraction_model_name = st.selectbox(
                        "Select LLM for PDF Extraction",
                        options=list(extraction_model_options.keys()),
                        help="Choose which LLM to use for extracting text from the PDF"
                    )
                    
                    if selected_extraction_model_name:
                        selected_extraction_config = extraction_model_options[selected_extraction_model_name]
                        
                        # Extraction prompt selection
                        st.markdown("**Extraction Prompt:**")
                        if st.session_state.prompts:
                            prompt_input_method = st.radio(
                                "Prompt Input Method:",
                                ["Select Saved Prompt", "Enter Custom Prompt", "Use Default"],
                                horizontal=True,
                                key="extraction_prompt_method"
                            )
                            
                            custom_extraction_prompt = None
                            
                            if prompt_input_method == "Select Saved Prompt":
                                prompt_options = {p['name']: p['prompt'] for p in st.session_state.prompts}
                                selected_extraction_prompt_name = st.selectbox(
                                    "Select Prompt for Extraction",
                                    options=list(prompt_options.keys()),
                                    help="Choose a saved prompt to use for PDF extraction",
                                    key="extraction_prompt_select"
                                )
                                if selected_extraction_prompt_name:
                                    custom_extraction_prompt = prompt_options[selected_extraction_prompt_name]
                                    with st.expander("📄 Preview Selected Prompt", expanded=False):
                                        st.text(custom_extraction_prompt)
                            
                            elif prompt_input_method == "Enter Custom Prompt":
                                custom_extraction_prompt = st.text_area(
                                    "Custom Extraction Prompt",
                                    height=100,
                                    placeholder="Enter custom instructions for how the LLM should extract text from the PDF...",
                                    help="Provide custom instructions for PDF extraction",
                                    key="custom_extraction_prompt"
                                )
                            
                            # If "Use Default", custom_extraction_prompt remains None
                        else:
                            # No saved prompts, show text area or default option
                            prompt_input_method = st.radio(
                                "Prompt Input Method:",
                                ["Enter Custom Prompt", "Use Default"],
                                horizontal=True,
                                key="extraction_prompt_method"
                            )
                            
                            custom_extraction_prompt = None
                            
                            if prompt_input_method == "Enter Custom Prompt":
                                custom_extraction_prompt = st.text_area(
                                    "Custom Extraction Prompt",
                                    height=100,
                                    placeholder="Enter custom instructions for how the LLM should extract text from the PDF...",
                                    help="Provide custom instructions for PDF extraction",
                                    key="custom_extraction_prompt"
                                )
                        
                        # Display selected model prominently
                        if selected_extraction_config:
                            model_name = selected_extraction_config.get('model', 'Unknown')
                            provider_name = selected_extraction_config.get('provider', 'Unknown').upper()
                            config_name = selected_extraction_config.get('name', 'Unnamed Config')
                            
                            st.info(f"🤖 **Model Selected for Extraction:** {model_name} ({provider_name}) | Config: {config_name}")
                        
                        if st.button("🔍 Extract Text Using LLM", type="primary", use_container_width=True):
                            # Check API key
                            provider = selected_extraction_config.get('provider', 'routellm')
                            if not st.session_state.api_keys.get(provider):
                                st.error(f"⚠️ Missing API key for {provider}. Please configure it in the 'API Keys' tab.")
                            else:
                                with st.spinner(f"Extracting text from PDF using {selected_extraction_config['model']}..."):
                                    try:
                                        router = LLMRouter()
                                        # Prepare extraction prompt (handle None and empty strings)
                                        extraction_prompt_to_use = None
                                        if custom_extraction_prompt and custom_extraction_prompt.strip():
                                            extraction_prompt_to_use = custom_extraction_prompt.strip()
                                        
                                        reference_text = extract_text_from_pdf_with_llm(
                                            uploaded_pdf,
                                            router,
                                            selected_extraction_config,
                                            st.session_state.api_keys,
                                            extraction_prompt_to_use
                                        )
                                        
                                        if reference_text:
                                            model_name = selected_extraction_config.get('model', 'Unknown')
                                            provider_name = selected_extraction_config.get('provider', 'Unknown').upper()
                                            
                                            st.success(f"✅ Successfully extracted {len(reference_text)} characters")
                                            st.info(f"🤖 **Model Used:** {model_name} ({provider_name}) | **Provider:** {provider_name}")
                                            
                                            pdf_uploaded = True
                                            # Store in session state
                                            st.session_state.extracted_reference_text = reference_text
                                            st.session_state.extraction_model_config = selected_extraction_config
                                            # Store PDF file and extraction prompt for later use in evaluation
                                            uploaded_pdf.seek(0)  # Reset file pointer
                                            st.session_state.pdf_file_bytes = uploaded_pdf.read()
                                            st.session_state.extraction_prompt = extraction_prompt_to_use
                                            # Show complete extracted text
                                            st.markdown("### 📄 Extracted Text")
                                            st.caption(f"🤖 **Extracted using:** {model_name} ({provider_name})")
                                            st.text_area("Extracted Text", value=reference_text, height=400, disabled=True, key="llm_extracted_text", label_visibility="collapsed")
                                        else:
                                            st.error("❌ Failed to extract text using LLM. The extraction returned no text.")
                                            st.info("💡 **Troubleshooting tips:**\n- Check if the PDF contains extractable text (not just images)\n- Try traditional extraction first to verify the PDF is readable\n- Verify your API key is correct\n- Check if the model supports the extraction task\n- For image-based PDFs, ensure you're using a vision-capable model (e.g., GPT-4 Vision, Claude 3, Gemini Pro Vision)\n- The system will automatically detect and extract images from PDFs when using vision-capable models")
                                    except Exception as e:
                                        error_msg = str(e)
                                        # Check if it's an image-based PDF error
                                        if "image-based" in error_msg.lower() or "traditional methods" in error_msg.lower():
                                            st.warning("⚠️ **Image-based PDF detected**")
                                            st.info("""
                                            **This PDF appears to be image-based (scanned document).**
                                            
                                            The LLM extraction will still be attempted, but note that:
                                            - Most text-based LLMs cannot process image PDFs directly
                                            - You may need a vision-capable model (e.g., GPT-4 Vision, Claude 3 with vision)
                                            - The extraction may return a description rather than extracted text
                                            
                                            **Alternative solutions:**
                                            - Use OCR software to convert the PDF to text first
                                            - Use a vision-capable LLM model
                                            - Try traditional extraction if the PDF has selectable text layers
                                            """)
                                        else:
                                            st.error(f"❌ Error extracting text with LLM: {error_msg}")
                                            with st.expander("🔍 Debug Information", expanded=False):
                                                st.markdown(f"**Error Details:**\n```\n{error_msg}\n```")
                                                st.markdown("**Possible causes:**")
                                                st.markdown("- PDF file may be corrupted or unreadable")
                                                st.markdown("- API key may be invalid or expired")
                                                st.markdown("- Model may not support the extraction task")
                                                st.markdown("- Network connectivity issues")
                                                st.markdown("- API rate limits exceeded")
                                                st.markdown("- For image-based PDFs, use a vision-capable model")
                        
                        # Use previously extracted text if available
                        if st.session_state.get('extracted_reference_text'):
                            reference_text = st.session_state.extracted_reference_text
                            pdf_uploaded = True
                            # Display model info for previously extracted text
                            if st.session_state.get('extraction_model_config'):
                                extraction_config = st.session_state.extraction_model_config
                                model_name = extraction_config.get('model', 'Unknown')
                                provider_name = extraction_config.get('provider', 'Unknown').upper()
                                st.info(f"📄 **Using previously extracted text** | 🤖 **Model:** {model_name} ({provider_name})")
            else:
                # Traditional extraction
                if st.button("🔍 Extract Text (Traditional)", type="primary", use_container_width=True):
                    with st.spinner("Extracting text from PDF..."):
                        try:
                            reference_text = extract_text_from_pdf(uploaded_pdf)
                            if reference_text:
                                st.success(f"✅ Successfully extracted {len(reference_text)} characters from PDF")
                                pdf_uploaded = True
                                st.session_state.extracted_reference_text = reference_text
                                # Store PDF file for later use in evaluation
                                uploaded_pdf.seek(0)  # Reset file pointer
                                st.session_state.pdf_file_bytes = uploaded_pdf.read()
                                st.session_state.extraction_prompt = None  # Traditional extraction doesn't use a prompt
                                # Show complete extracted text
                                st.markdown("### 📄 Extracted Text")
                                st.text_area("Extracted Text", value=reference_text, height=400, disabled=True, key="traditional_extracted_text", label_visibility="collapsed")
                            else:
                                st.error("❌ Failed to extract text from PDF. Please try LLM extraction or use text input.")
                        except Exception as e:
                            st.error(f"❌ Error processing PDF: {str(e)}")
                
                # Use previously extracted text if available
                if st.session_state.get('extracted_reference_text'):
                    reference_text = st.session_state.extracted_reference_text
                    pdf_uploaded = True
    
    elif ref_input_method == "Text Input":
        reference_text = st.text_area(
            "Reference Text",
            height=150,
            placeholder="Enter the reference text to compare model outputs against...",
            key="reference_text_input"
        )
    
    st.divider()
    
    # Select prompt
    st.subheader("📝 Prompt to Evaluate")
    
    # Show info if in PDF extraction mode
    if pdf_uploaded and reference_text and st.session_state.get('pdf_file_bytes'):
        extraction_prompt = st.session_state.get('extraction_prompt')
        if extraction_prompt:
            st.info(f"ℹ️ **PDF Extraction Mode**: Models will extract text from the PDF using the same extraction prompt used for the base response. The prompt below will be used if no extraction prompt was set.")
        else:
            st.info(f"ℹ️ **PDF Extraction Mode**: Models will extract text from the PDF using the prompt you enter below.")
    
    if st.session_state.prompts:
        prompt_options = {p['name']: p['prompt'] for p in st.session_state.prompts}
        selected_prompt_name = st.selectbox(
            "Select Saved Prompt",
            options=["-- Enter New Prompt --"] + list(prompt_options.keys())
        )
        
        if selected_prompt_name != "-- Enter New Prompt --":
            evaluation_prompt = prompt_options[selected_prompt_name]
        else:
            evaluation_prompt = st.text_area(
                "Or Enter Prompt Here",
                height=200,
                placeholder="Enter your prompt here..."
            )
    else:
        evaluation_prompt = st.text_area(
            "Enter Prompt",
            height=200,
            placeholder="Enter your prompt here..."
        )
    
    # Select model configurations
    st.divider()
    st.subheader("Select Models to Evaluate")
    
    if not st.session_state.model_configs:
        st.warning("⚠️ No model configurations found. Please configure models in the 'Model Config' tab first.")
    else:
        selected_configs = []
        for config in st.session_state.model_configs:
            if st.checkbox(
                f"{config['name']} ({config['provider']} - {config['model']})",
                key=f"select_{config['id']}"
            ):
                selected_configs.append(config)
        
        # Display selected models prominently
        if selected_configs:
            st.markdown("---")
            st.markdown("### 🤖 Models Selected for Evaluation")
            model_list = []
            for config in selected_configs:
                model_name = config.get('model', 'Unknown')
                provider_name = config.get('provider', 'Unknown').upper()
                config_name = config.get('name', 'Unnamed')
                model_list.append(f"**{config_name}**: {model_name} ({provider_name})")
            st.info(" | ".join(model_list))
            st.markdown("---")
        
        # Evaluate button
        if st.button("🚀 Evaluate Prompt", type="primary", use_container_width=True):
            if not evaluation_prompt or not evaluation_prompt.strip():
                st.error("Please enter a prompt to evaluate.")
            elif not selected_configs:
                st.error("Please select at least one model configuration.")
            else:
                # Check API keys
                required_providers = set(c['provider'] for c in selected_configs)
                missing_keys = []
                for provider in required_providers:
                    if not st.session_state.api_keys.get(provider):
                        missing_keys.append(provider)
                
                if missing_keys:
                    st.error(f"⚠️ Missing API keys for: {', '.join(missing_keys)}. Please configure them in the 'API Keys' tab.")
                else:
                    # Initialize router
                    router = LLMRouter()
                    
                    # Show progress
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Check if we're in PDF extraction mode (base response exists from PDF)
                    is_pdf_extraction_mode = (
                        pdf_uploaded and 
                        reference_text and 
                        st.session_state.get('pdf_file_bytes') is not None
                    )
                    
                    results = []
                    
                    if is_pdf_extraction_mode:
                        # PDF extraction mode: Each model extracts text from PDF using the same extraction prompt
                        status_text.text("Extracting text from PDF using selected models...")
                        pdf_file_bytes = st.session_state.get('pdf_file_bytes')
                        extraction_prompt = st.session_state.get('extraction_prompt')
                        
                        # If no extraction prompt was stored (traditional extraction), use the evaluation prompt
                        if not extraction_prompt:
                            extraction_prompt = evaluation_prompt
                        
                        # Create a BytesIO object from stored bytes
                        pdf_file = BytesIO(pdf_file_bytes)
                        
                        for idx, config in enumerate(selected_configs):
                            progress = int((idx / len(selected_configs)) * 50)
                            progress_bar.progress(progress)
                            status_text.text(f"Extracting with {config['model']} ({idx + 1}/{len(selected_configs)})...")
                            
                            try:
                                # Reset file pointer for each extraction
                                pdf_file.seek(0)
                                
                                # Extract text using the same extraction prompt
                                extracted_text = extract_text_from_pdf_with_llm(
                                    pdf_file,
                                    router,
                                    config,
                                    st.session_state.api_keys,
                                    extraction_prompt
                                )
                                
                                if extracted_text:
                                    results.append({
                                        'success': True,
                                        'model': config.get('model', 'Unknown'),
                                        'response': extracted_text,
                                        'usage': {},  # Usage info not available from extraction
                                        'provider': config.get('provider', 'Unknown'),
                                        'error': None
                                    })
                                else:
                                    results.append({
                                        'success': False,
                                        'model': config.get('model', 'Unknown'),
                                        'response': None,
                                        'usage': None,
                                        'provider': config.get('provider', 'Unknown'),
                                        'error': 'Failed to extract text from PDF'
                                    })
                            except Exception as e:
                                results.append({
                                    'success': False,
                                    'model': config.get('model', 'Unknown'),
                                    'response': None,
                                    'usage': None,
                                    'provider': config.get('provider', 'Unknown'),
                                    'error': str(e)
                                })
                    else:
                        # Normal mode: Route prompt to models
                        status_text.text("Sending requests to models...")
                        results = router.route_to_models(
                            evaluation_prompt,
                            selected_configs,
                            st.session_state.api_keys
                        )
                    
                    progress_bar.progress(50)
                    status_text.text("✅ Model responses received! Calculating evaluation metrics...")
                    
                    # Calculate evaluation metrics if reference text is provided
                    evaluation_metrics = {}
                    if reference_text and reference_text.strip():
                        for result in results:
                            if result.get('success') and result.get('response'):
                                try:
                                    metrics = evaluate_output(reference_text, result['response'])
                                    result['evaluation_metrics'] = metrics
                                    evaluation_metrics[result.get('model', 'unknown')] = metrics
                                except Exception as e:
                                    st.warning(f"⚠️ Error calculating metrics for {result.get('model', 'unknown')}: {str(e)}")
                                    result['evaluation_metrics'] = {
                                        'bleu': {},
                                        'rouge': {},
                                        'error': str(e)
                                    }
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Evaluation complete!")
                    
                    # Get extraction method info
                    extraction_llm_info = None
                    if pdf_uploaded and st.session_state.get('extraction_model_config'):
                        extraction_llm_info = {
                            'model': st.session_state.extraction_model_config.get('model'),
                            'provider': st.session_state.extraction_model_config.get('provider'),
                            'name': st.session_state.extraction_model_config.get('name')
                        }
                    
                    # Prepare evaluation data with full metadata
                    timestamp = datetime.now().isoformat()
                    evaluation_data = {
                        'timestamp': timestamp,
                        'prompt': evaluation_prompt,
                        'prompt_name': selected_prompt_name if selected_prompt_name != "-- Enter New Prompt --" else "Unnamed Prompt",
                        'reference_text': reference_text if reference_text else None,
                        'reference_source': 'PDF' if pdf_uploaded else ('Text' if reference_text else None),
                        'extraction_method': 'LLM' if extraction_llm_info else ('Traditional' if pdf_uploaded else None),
                        'extraction_llm': extraction_llm_info,
                        'model_configs': selected_configs,
                        'results': results,
                        'evaluation_metrics': evaluation_metrics if evaluation_metrics else None,
                        'metadata': {
                            'num_models': len(selected_configs),
                            'num_successful': sum(1 for r in results if r.get('success')),
                            'num_failed': sum(1 for r in results if not r.get('success')),
                            'total_tokens': sum(r.get('usage', {}).get('total_tokens', 0) for r in results if r.get('success')),
                            'prompt_length': len(evaluation_prompt),
                            'prompt_words': len(evaluation_prompt.split()),
                            'has_reference': bool(reference_text),
                            'reference_length': len(reference_text) if reference_text else 0
                        }
                    }
                    
                    # Save evaluation to storage
                    saved_evaluation = save_evaluation(evaluation_data)
                    
                    # Store in session state for immediate display
                    st.session_state.evaluation_results = {
                        'id': saved_evaluation['id'],
                        'prompt': evaluation_prompt,
                        'timestamp': timestamp,
                        'results': results,
                        'reference_text': reference_text,
                        'extraction_method': evaluation_data.get('extraction_method'),
                        'extraction_llm': extraction_llm_info,
                        'saved': True
                    }
                    
                    st.success(f"✅ Evaluation saved! (ID: {saved_evaluation['id']})")
                    st.rerun()
        
        # Display results
        if st.session_state.evaluation_results:
            st.divider()
            st.subheader("📊 Evaluation Results")
            
            results = st.session_state.evaluation_results['results']
            
            # Display prompt
            with st.expander("📝 Evaluated Prompt", expanded=False):
                st.text(st.session_state.evaluation_results['prompt'])
            
            # Display reference text prominently if available
            if st.session_state.evaluation_results.get('reference_text'):
                ref_text = st.session_state.evaluation_results['reference_text']
                extraction_method = st.session_state.evaluation_results.get('extraction_method', 'Unknown')
                extraction_llm = st.session_state.evaluation_results.get('extraction_llm')
                
                st.markdown("---")
                st.markdown("### 📄 Reference Text (Base Response)")
                if extraction_llm:
                    model_name = extraction_llm.get('model', 'Unknown')
                    provider_name = extraction_llm.get('provider', 'Unknown').upper()
                    config_name = extraction_llm.get('name', 'Unknown')
                    st.info(f"🤖 **Model Used for Extraction:** {model_name} ({provider_name}) | **Config:** {config_name} | **Method:** {extraction_method}")
                elif extraction_method == 'Traditional':
                    st.info(f"📄 **Extraction Method:** Traditional (PyPDF2/pdfplumber) | **No LLM model used**")
                
                st.text_area(
                    "Reference Text",
                    value=ref_text,
                    height=250,
                    disabled=True,
                    key="reference_text_display",
                    label_visibility="collapsed"
                )
                st.caption("ℹ️ This is the base reference text. All model outputs below are compared against this text to calculate BLEU/ROUGE scores.")
                st.markdown("---")
            
            # Create comparison table if reference text exists and we have metrics
            if st.session_state.evaluation_results.get('reference_text') and results:
                successful_results = [r for r in results if r.get('success') and r.get('evaluation_metrics') and not r.get('evaluation_metrics', {}).get('error')]
                if successful_results:
                    # Score Reference Guide
                    with st.expander("📖 Score Reference Guide - Understanding BLEU & ROUGE Scores", expanded=False):
                        st.markdown("""
                        ### Understanding Evaluation Scores
                        
                        All scores range from **0.0 to 1.0**, where higher scores indicate better similarity to the reference text.
                        
                        #### 🎯 BLEU Scores
                        **BLEU (Bilingual Evaluation Understudy)** measures precision - how many words/phrases from the model output appear in the reference text.
                        
                        **BLEU Variants Explained:**
                        - **BLEU-1**: Measures overlap of **single words** (unigrams) - most lenient, easiest to score high
                        - **BLEU-2**: Measures overlap of **2-word phrases** (bigrams) - more strict
                        - **BLEU-3**: Measures overlap of **3-word phrases** (trigrams) - even more strict
                        - **BLEU-4**: Measures overlap of **4-word phrases** (4-grams) - most strict, requires longer exact matches
                        - **BLEU** (overall): Standard BLEU score using 4-gram precision with brevity penalty - the most commonly used metric
                        
                        **Why BLEU-4 is Important:**
                        BLEU-4 is the most strict and commonly used metric because it requires longer exact phrase matches, making it a better indicator of true similarity. Higher BLEU-4 scores mean the model output contains more exact 4-word phrases that match the reference text.
                        
                        | Score Range | Interpretation |
                        |------------|----------------|
                        | **0.70 - 1.00** | 🟢 **Excellent** - Very high similarity, almost identical content |
                        | **0.50 - 0.69** | 🟡 **Good** - Strong similarity with minor differences |
                        | **0.30 - 0.49** | 🟠 **Fair** - Moderate similarity, some key content matches |
                        | **0.10 - 0.29** | 🔴 **Poor** - Low similarity, limited content overlap |
                        | **0.00 - 0.09** | ⚫ **Very Poor** - Minimal to no similarity |
                        
                        **Note:** All BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4, and overall BLEU) are calculated and available in the "Detailed Evaluation Metrics" section below.
                        
                        #### 📊 ROUGE Scores
                        **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** measures both precision and recall - how well the model output captures the key information from the reference.
                        
                        | Score Range | Interpretation |
                        |------------|----------------|
                        | **0.70 - 1.00** | 🟢 **Excellent** - Captures most/all key information accurately |
                        | **0.50 - 0.69** | 🟡 **Good** - Captures majority of important information |
                        | **0.30 - 0.49** | 🟠 **Fair** - Captures some key information, missing details |
                        | **0.10 - 0.29** | 🔴 **Poor** - Captures limited information |
                        | **0.00 - 0.09** | ⚫ **Very Poor** - Fails to capture meaningful information |
                        
                        **ROUGE Variants:**
                        - **ROUGE-1**: Measures overlap of unigrams (single words)
                        - **ROUGE-2**: Measures overlap of bigrams (2-word phrases) - more strict
                        - **ROUGE-L**: Measures longest common subsequence - captures sentence structure
                        
                        #### 💡 Key Insights
                        - **BLEU** focuses on exact word/phrase matching (precision)
                        - **ROUGE** focuses on information coverage (recall + precision)
                        - **Higher scores** = Better alignment with reference text
                        - **Lower scores** = More differences from reference text
                        - Scores are **relative** - compare models against each other, not absolute thresholds
                        """)
                    
                    st.markdown("### 📊 Score Comparison Summary")
                    st.markdown("Quick comparison of all models against the reference text:")
                    
                    # Prepare data for comparison table
                    comparison_data = []
                    for result in successful_results:
                        metrics = result.get('evaluation_metrics', {})
                        bleu = metrics.get('bleu', {})
                        rouge = metrics.get('rouge', {})
                        
                        # Get ROUGE scores (handle both formats)
                        r1_key = 'ROUGE-1' if 'ROUGE-1' in rouge else ('ROUGE1' if 'ROUGE1' in rouge else None)
                        r2_key = 'ROUGE-2' if 'ROUGE-2' in rouge else ('ROUGE2' if 'ROUGE2' in rouge else None)
                        rl_key = 'ROUGE-L' if 'ROUGE-L' in rouge else ('ROUGEL' if 'ROUGEL' in rouge else None)
                        
                        comparison_data.append({
                            "Model": result.get('model', 'Unknown'),
                            "Provider": result.get('provider', 'Unknown'),
                            "BLEU": f"{bleu.get('BLEU', 0):.4f}" if bleu.get('BLEU') else "N/A",
                            "BLEU-4": f"{bleu.get('BLEU-4', 0):.4f}" if bleu.get('BLEU-4') else "N/A",
                            "ROUGE-1 F1": f"{rouge[r1_key]['f1']:.4f}" if r1_key and rouge.get(r1_key) else "N/A",
                            "ROUGE-2 F1": f"{rouge[r2_key]['f1']:.4f}" if r2_key and rouge.get(r2_key) else "N/A",
                            "ROUGE-L F1": f"{rouge[rl_key]['f1']:.4f}" if rl_key and rouge.get(rl_key) else "N/A",
                        })
                    
                    if comparison_data:
                        import pandas as pd
                        df = pd.DataFrame(comparison_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                    st.markdown("---")
            
            # Display results side by side
            num_results = len(results)
            if num_results > 0:
                st.markdown("### 🤖 Model Responses & Scores")
                st.markdown("Compare each model's output and evaluation scores side by side:")
                
                # Determine number of columns (max 3, but adjust based on number of results)
                max_cols = min(num_results, 3)
                cols = st.columns(max_cols)
                
                for idx, result in enumerate(results):
                    col_idx = idx % max_cols
                    with cols[col_idx]:
                        # Create a container with border for each model
                        with st.container():
                            # Status indicator and model name
                            if result['success']:
                                st.markdown(f"### ✅ {result['model']}")
                            else:
                                st.markdown(f"### ❌ {result['model']}")
                            
                            # Model info
                            st.caption(f"**Provider:** {result['provider']}")
                            
                            # Show scores prominently at the top if available (only if reference text exists)
                            if (result.get('success') and 
                                st.session_state.evaluation_results.get('reference_text') and
                                result.get('evaluation_metrics') and 
                                not result.get('evaluation_metrics', {}).get('error')):
                                metrics = result['evaluation_metrics']
                                bleu = metrics.get('bleu', {})
                                rouge = metrics.get('rouge', {})
                                
                                # Key scores summary
                                st.markdown("**📊 Key Scores (vs Reference):**")
                                score_cols = st.columns(3)
                                with score_cols[0]:
                                    bleu4_val = bleu.get('BLEU-4', 0) if bleu.get('BLEU-4') is not None else 0
                                    st.metric("BLEU-4", f"{bleu4_val:.4f}")
                                with score_cols[1]:
                                    r1_key = 'ROUGE-1' if 'ROUGE-1' in rouge else ('ROUGE1' if 'ROUGE1' in rouge else None)
                                    r1_f1 = f"{rouge[r1_key]['f1']:.4f}" if r1_key and rouge.get(r1_key) else "0.0000"
                                    st.metric("ROUGE-1 F1", r1_f1)
                                with score_cols[2]:
                                    rl_key = 'ROUGE-L' if 'ROUGE-L' in rouge else ('ROUGEL' if 'ROUGEL' in rouge else None)
                                    rl_f1 = f"{rouge[rl_key]['f1']:.4f}" if rl_key and rouge.get(rl_key) else "0.0000"
                                    st.metric("ROUGE-L F1", rl_f1)
                                
                                st.markdown("---")
                            
                            # Response
                            if result['success']:
                                # Show response in expander to keep it clean
                                with st.expander("📝 View Generated Response", expanded=False):
                                    st.text_area(
                                        "Model Response",
                                        value=result['response'],
                                        height=200,
                                        key=f"response_{idx}",
                                        disabled=True,
                                        label_visibility="collapsed"
                                    )
                                    
                                    # Response stats (only if meaningful)
                                    if result.get('response'):
                                        response_text = result['response']
                                        if response_text and len(response_text) > 0:
                                            st.caption(f"📏 Length: {len(response_text)} characters")
                                    
                                    # Usage stats (only if available)
                                    if result.get('usage'):
                                        usage = result['usage']
                                        total_tokens = usage.get('total_tokens')
                                        prompt_tokens = usage.get('prompt_tokens')
                                        completion_tokens = usage.get('completion_tokens')
                                        if total_tokens and total_tokens != 'N/A':
                                            if prompt_tokens and completion_tokens and prompt_tokens != 'N/A' and completion_tokens != 'N/A':
                                                st.caption(f"💾 Tokens: {total_tokens} (Prompt: {prompt_tokens}, Completion: {completion_tokens})")
                                            else:
                                                st.caption(f"💾 Tokens: {total_tokens}")
                                
                                # Detailed Evaluation Metrics (BLEU/ROUGE) - only show if we have reference text
                                if result.get('evaluation_metrics') and st.session_state.evaluation_results.get('reference_text'):
                                    metrics = result['evaluation_metrics']
                                    if not metrics.get('error'):
                                        with st.expander("📊 Detailed Evaluation Metrics", expanded=False):
                                            # BLEU Scores
                                            if metrics.get('bleu'):
                                                st.markdown("#### BLEU Scores")
                                                bleu = metrics['bleu']
                                                col1, col2, col3, col4, col5 = st.columns(5)
                                                with col1:
                                                    st.metric("BLEU", f"{bleu.get('BLEU', 0):.4f}")
                                                with col2:
                                                    st.metric("BLEU-1", f"{bleu.get('BLEU-1', 0):.4f}")
                                                with col3:
                                                    st.metric("BLEU-2", f"{bleu.get('BLEU-2', 0):.4f}")
                                                with col4:
                                                    st.metric("BLEU-3", f"{bleu.get('BLEU-3', 0):.4f}")
                                                with col5:
                                                    st.metric("BLEU-4", f"{bleu.get('BLEU-4', 0):.4f}")
                                            
                                            # ROUGE Scores
                                            if metrics.get('rouge'):
                                                st.markdown("#### ROUGE Scores")
                                                rouge = metrics['rouge']
                                                
                                                # ROUGE-1 (check both formats for compatibility)
                                                r1_key = 'ROUGE-1' if 'ROUGE-1' in rouge else ('ROUGE1' if 'ROUGE1' in rouge else None)
                                                if r1_key:
                                                    st.markdown("**ROUGE-1:**")
                                                    r1 = rouge[r1_key]
                                                    col1, col2, col3 = st.columns(3)
                                                    with col1:
                                                        st.metric("Precision", f"{r1.get('precision', 0):.4f}")
                                                    with col2:
                                                        st.metric("Recall", f"{r1.get('recall', 0):.4f}")
                                                    with col3:
                                                        st.metric("F1", f"{r1.get('f1', 0):.4f}")
                                                
                                                # ROUGE-2
                                                r2_key = 'ROUGE-2' if 'ROUGE-2' in rouge else ('ROUGE2' if 'ROUGE2' in rouge else None)
                                                if r2_key:
                                                    st.markdown("**ROUGE-2:**")
                                                    r2 = rouge[r2_key]
                                                    col1, col2, col3 = st.columns(3)
                                                    with col1:
                                                        st.metric("Precision", f"{r2.get('precision', 0):.4f}")
                                                    with col2:
                                                        st.metric("Recall", f"{r2.get('recall', 0):.4f}")
                                                    with col3:
                                                        st.metric("F1", f"{r2.get('f1', 0):.4f}")
                                                
                                                # ROUGE-L
                                                rl_key = 'ROUGE-L' if 'ROUGE-L' in rouge else ('ROUGEL' if 'ROUGEL' in rouge else None)
                                                if rl_key:
                                                    st.markdown("**ROUGE-L:**")
                                                    rl = rouge[rl_key]
                                                    col1, col2, col3 = st.columns(3)
                                                    with col1:
                                                        st.metric("Precision", f"{rl.get('precision', 0):.4f}")
                                                    with col2:
                                                        st.metric("Recall", f"{rl.get('recall', 0):.4f}")
                                                    with col3:
                                                        st.metric("F1", f"{rl.get('f1', 0):.4f}")
                                                
                                                # Debug: Show available keys if no scores displayed
                                                if not any([r1_key, r2_key, rl_key]):
                                                    st.warning(f"⚠️ ROUGE scores dictionary keys: {list(rouge.keys())}")
                                            else:
                                                st.info("ℹ️ ROUGE scores not available. Check if reference text was provided.")
                                        
                                        # Score Summary & Analysis
                                        st.markdown("---")
                                        st.markdown("#### 📋 Score Summary & Analysis")
                                        st.markdown("Detailed analysis of the evaluation scores based on the actual texts provided.")
                                        
                                        summary_export_data = None
                                        try:
                                            reference_text = st.session_state.evaluation_results.get('reference_text', '')
                                            response_text = result.get('response', '')
                                            
                                            if reference_text and response_text:
                                                summary = generate_score_summary(
                                                    reference_text,
                                                    response_text,
                                                    metrics
                                                )
                                                st.markdown(summary)
                                                summary_export_data = prepare_score_summary_export(
                                                    reference_text,
                                                    response_text,
                                                    metrics,
                                                    summary
                                                )
                                            else:
                                                st.info("ℹ️ Reference or response text not available for detailed analysis.")
                                        except Exception as e:
                                            st.warning(f"⚠️ Could not generate detailed analysis: {str(e)}")
                                        
                                        if summary_export_data:
                                            download_cols = st.columns(3)
                                            model_safe_name = result.get('model', f"model_{idx}").replace("/", "_").replace(" ", "_")
                                            timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
                                            
                                            text_document = build_summary_text_document(summary_export_data)
                                            json_document = json.dumps(summary_export_data, indent=2, ensure_ascii=False)
                                            markdown_report = build_markdown_report(summary_export_data)
                                            
                                            with download_cols[0]:
                                                st.download_button(
                                                    label="⬇️ Download Summary (Text)",
                                                    data=text_document,
                                                    file_name=f"{model_safe_name}_summary_{timestamp_suffix}.txt",
                                                    mime="text/plain",
                                                    key=f"text_summary_download_{idx}"
                                                )
                                            with download_cols[1]:
                                                st.download_button(
                                                    label="⬇️ Download Summary (JSON)",
                                                    data=json_document,
                                                    file_name=f"{model_safe_name}_summary_{timestamp_suffix}.json",
                                                    mime="application/json",
                                                    key=f"json_summary_download_{idx}"
                                                )
                                            with download_cols[2]:
                                                st.download_button(
                                                    label="⬇️ Download Summary (Markdown)",
                                                    data=markdown_report,
                                                    file_name=f"{model_safe_name}_summary_{timestamp_suffix}.md",
                                                    mime="text/markdown",
                                                    key=f"md_summary_download_{idx}"
                                                )
                                            
                                            st.text_area(
                                                "📋 Copy-ready Markdown (Excel friendly)",
                                                value=markdown_report,
                                                height=220,
                                                key=f"md_copy_area_{idx}",
                                                help="Select all (Ctrl+A) and copy to paste into a single Excel cell. Use Alt+Enter inside Excel to keep line breaks."
                                            )
                                    else:
                                        st.warning(f"⚠️ Error calculating metrics: {metrics.get('error')}")
                            else:
                                st.error(f"Error: {result.get('error', 'Unknown error')}")
                        
                        st.divider()

# ========== TAB 5: RESULTS HISTORY ==========
with tab5:
    st.header("📊 Evaluation Results History")
    st.markdown("View and manage all saved evaluation results.")
    
    evaluations = load_evaluations()
    
    if evaluations:
        # Sort by timestamp (newest first)
        evaluations.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        # Filter and search
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("🔍 Search evaluations", placeholder="Search by prompt name or content...")
        with col2:
            show_all = st.checkbox("Show all", value=True)
        
        # Filter evaluations
        filtered_evaluations = evaluations
        if search_term:
            filtered_evaluations = [
                e for e in evaluations
                if search_term.lower() in e.get('prompt_name', '').lower() or
                   search_term.lower() in e.get('prompt', '').lower()
            ]
        
        st.markdown(f"**Total Evaluations:** {len(evaluations)} | **Showing:** {len(filtered_evaluations)}")
        st.divider()
        
        # Display evaluations
        for eval_data in filtered_evaluations:
            eval_id = eval_data['id']
            timestamp = eval_data.get('timestamp', '')
            prompt_name = eval_data.get('prompt_name', 'Unnamed Prompt')
            
            # Format timestamp
            try:
                dt = datetime.fromisoformat(timestamp)
                formatted_time = dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                formatted_time = timestamp
            
            with st.expander(f"📋 Evaluation #{eval_id} - {prompt_name} ({formatted_time})", expanded=False):
                # Metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Models Evaluated", eval_data.get('metadata', {}).get('num_models', len(eval_data.get('results', []))))
                with col2:
                    st.metric("Successful", eval_data.get('metadata', {}).get('num_successful', 0))
                with col3:
                    st.metric("Total Tokens", eval_data.get('metadata', {}).get('total_tokens', 0))
                
                # Prompt
                st.markdown("#### Prompt")
                st.text_area("Prompt", value=eval_data.get('prompt', ''), height=100, disabled=True, key=f"prompt_eval_{eval_id}_{timestamp}", label_visibility="collapsed")
                
                # Reference text info
                if eval_data.get('reference_text'):
                    ref_source = eval_data.get('reference_source', 'Unknown')
                    extraction_method = eval_data.get('extraction_method', 'Unknown')
                    extraction_llm = eval_data.get('extraction_llm')
                    
                    st.markdown("#### Reference Text")
                    with st.expander(f"📄 Reference Text ({ref_source} - {extraction_method})", expanded=False):
                        if extraction_llm:
                            st.caption(f"Extracted using: {extraction_llm.get('name', 'Unknown')} ({extraction_llm.get('model', 'Unknown')})")
                        st.text_area("Reference Text", value=eval_data.get('reference_text', ''), height=150, disabled=True, key=f"ref_text_eval_{eval_id}_{timestamp}", label_visibility="collapsed")
                
                # Model configurations used
                st.markdown("#### Model Configurations")
                if eval_data.get('model_configs'):
                    for config in eval_data['model_configs']:
                        st.json({
                            "Name": config.get('name', ''),
                            "Provider": config.get('provider', ''),
                            "Model": config.get('model', ''),
                            "Temperature": config.get('temperature', ''),
                            "Max Tokens": config.get('max_tokens', '')
                        })
                
                # Results
                st.markdown("#### Results")
                results = eval_data.get('results', [])
                if results:
                    num_cols = min(len(results), 3)
                    cols = st.columns(num_cols)
                    
                    for idx, result in enumerate(results):
                        col_idx = idx % num_cols
                        with cols[col_idx]:
                            if result.get('success'):
                                st.success(f"✅ {result.get('model', 'Unknown')}")
                            else:
                                st.error(f"❌ {result.get('model', 'Unknown')}")
                            
                            st.caption(f"Provider: {result.get('provider', 'Unknown')}")
                            
                            if result.get('success'):
                                st.text_area(
                                    "Response",
                                    value=result.get('response', ''),
                                    height=200,
                                    disabled=True,
                                    key=f"result_eval_{eval_id}_{idx}_{timestamp}"
                                )
                                
                                if result.get('usage'):
                                    usage = result['usage']
                                    st.caption(f"Tokens: {usage.get('total_tokens', 'N/A')}")
                                
                                # Evaluation Metrics (BLEU/ROUGE) in history view
                                if result.get('evaluation_metrics'):
                                    metrics = result['evaluation_metrics']
                                    if not metrics.get('error'):
                                        with st.expander("📊 Evaluation Metrics", expanded=False):
                                            # BLEU Scores
                                            if metrics.get('bleu'):
                                                st.markdown("**BLEU Scores:**")
                                                bleu = metrics['bleu']
                                                st.caption(f"BLEU: {bleu.get('BLEU', 0):.4f} | BLEU-1: {bleu.get('BLEU-1', 0):.4f} | BLEU-2: {bleu.get('BLEU-2', 0):.4f} | BLEU-3: {bleu.get('BLEU-3', 0):.4f} | BLEU-4: {bleu.get('BLEU-4', 0):.4f}")
                                            
                                            # ROUGE Scores
                                            if metrics.get('rouge'):
                                                st.markdown("**ROUGE Scores:**")
                                                rouge = metrics['rouge']
                                                if 'ROUGE1' in rouge:
                                                    r1 = rouge['ROUGE1']
                                                    st.caption(f"ROUGE-1 F1: {r1.get('f1', 0):.4f} (P: {r1.get('precision', 0):.4f}, R: {r1.get('recall', 0):.4f})")
                                                if 'ROUGE2' in rouge:
                                                    r2 = rouge['ROUGE2']
                                                    st.caption(f"ROUGE-2 F1: {r2.get('f1', 0):.4f} (P: {r2.get('precision', 0):.4f}, R: {r2.get('recall', 0):.4f})")
                                                if 'ROUGEL' in rouge:
                                                    rl = rouge['ROUGEL']
                                                    st.caption(f"ROUGE-L F1: {rl.get('f1', 0):.4f} (P: {rl.get('precision', 0):.4f}, R: {rl.get('recall', 0):.4f})")
                            else:
                                st.error(f"Error: {result.get('error', 'Unknown error')}")
                
                # Actions
                st.divider()
                col1, col2, col3 = st.columns(3)
                with col1:
                    # Export JSON
                    eval_json = json.dumps(eval_data, indent=2, ensure_ascii=False)
                    st.download_button(
                        label="📥 Download JSON",
                        data=eval_json,
                        file_name=f"evaluation_{eval_id}_{formatted_time.replace(':', '-').replace(' ', '_')}.json",
                        mime="application/json",
                        key=f"export_eval_{eval_id}_{timestamp}"
                    )
                with col2:
                    # View full details
                    if st.button("👁️ View Details", key=f"view_eval_{eval_id}_{timestamp}", use_container_width=True):
                        st.json(eval_data)
                with col3:
                    # Delete
                    if st.button("🗑️ Delete", key=f"delete_eval_{eval_id}_{timestamp}", use_container_width=True, type="secondary"):
                        if delete_evaluation(eval_id):
                            st.success("✅ Evaluation deleted!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to delete evaluation")
        
        # Bulk export
        st.divider()
        st.subheader("📥 Bulk Export")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export All as JSON", use_container_width=True):
                export_data = json.dumps(evaluations, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download All Evaluations",
                    data=export_data,
                    file_name=f"all_evaluations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        with col2:
            if st.button("📊 Export Summary CSV", use_container_width=True):
                import csv
                import io
                
                output = io.StringIO()
                writer = csv.writer(output)
                
                # Header
                writer.writerow([
                    'ID', 'Timestamp', 'Prompt Name', 'Prompt Length', 'Models Evaluated',
                    'Successful', 'Failed', 'Total Tokens', 'Models'
                ])
                
                # Data rows
                for eval_data in evaluations:
                    metadata = eval_data.get('metadata', {})
                    models = ', '.join([r.get('model', '') for r in eval_data.get('results', [])])
                    writer.writerow([
                        eval_data['id'],
                        eval_data.get('timestamp', ''),
                        eval_data.get('prompt_name', ''),
                        metadata.get('prompt_length', 0),
                        metadata.get('num_models', 0),
                        metadata.get('num_successful', 0),
                        metadata.get('num_failed', 0),
                        metadata.get('total_tokens', 0),
                        models
                    ])
                
                csv_data = output.getvalue()
                st.download_button(
                    label="Download CSV",
                    data=csv_data,
                    file_name=f"evaluations_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    else:
        st.info("No evaluation results yet. Run evaluations in the 'Evaluate' tab to see results here.")

# ========== TAB 6: EVALUATE TEXT ==========
with tab6:
    st.header("📝 Evaluate Text")
    st.markdown("Compare two text inputs directly using BLEU/ROUGE evaluation metrics.")
    
    # Two text fields side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📄 Reference Text")
        reference_text_input = st.text_area(
            "Reference Text",
            height=300,
            placeholder="Enter the reference text (ground truth) to compare against...",
            key="eval_text_reference",
            help="This is the reference or expected text that will be used as the baseline for comparison."
        )
    
    with col2:
        st.subheader("📝 Response Text")
        response_text_input = st.text_area(
            "Response Text",
            height=300,
            placeholder="Enter the response text to evaluate...",
            key="eval_text_response",
            help="This is the text that will be evaluated against the reference text."
        )
    
    st.divider()
    
    # Evaluate button
    if st.button("🚀 Evaluate Text", type="primary", use_container_width=True):
        if not reference_text_input or not reference_text_input.strip():
            st.error("⚠️ Please enter reference text.")
        elif not response_text_input or not response_text_input.strip():
            st.error("⚠️ Please enter response text.")
        else:
            # Calculate evaluation metrics
            with st.spinner("Calculating evaluation metrics..."):
                try:
                    metrics = evaluate_output(reference_text_input.strip(), response_text_input.strip())
                    
                    # Store results in session state for display
                    st.session_state.text_evaluation_results = {
                        'reference_text': reference_text_input.strip(),
                        'response_text': response_text_input.strip(),
                        'metrics': metrics,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.success("✅ Evaluation complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error during evaluation: {str(e)}")
    
    # Display results if available
    if st.session_state.get('text_evaluation_results'):
        results = st.session_state.text_evaluation_results
        metrics = results.get('metrics', {})
        
        st.divider()
        st.subheader("📊 Evaluation Results")
        
        # Check for errors
        if metrics.get('error'):
            st.error(f"❌ Error calculating metrics: {metrics.get('error')}")
        else:
            # Score Reference Guide
            with st.expander("📖 Score Reference Guide - Understanding BLEU & ROUGE Scores", expanded=False):
                st.markdown("""
                ### Understanding Evaluation Scores
                
                All scores range from **0.0 to 1.0**, where higher scores indicate better similarity to the reference text.
                
                #### 🎯 BLEU Scores
                **BLEU (Bilingual Evaluation Understudy)** measures precision - how many words/phrases from the response text appear in the reference text.
                
                **BLEU Variants Explained:**
                - **BLEU-1**: Measures overlap of **single words** (unigrams) - most lenient, easiest to score high
                - **BLEU-2**: Measures overlap of **2-word phrases** (bigrams) - more strict
                - **BLEU-3**: Measures overlap of **3-word phrases** (trigrams) - even more strict
                - **BLEU-4**: Measures overlap of **4-word phrases** (4-grams) - most strict, requires longer exact matches
                - **BLEU** (overall): Standard BLEU score using 4-gram precision with brevity penalty - the most commonly used metric
                
                **Why BLEU-4 is Important:**
                BLEU-4 is the most strict and commonly used metric because it requires longer exact phrase matches, making it a better indicator of true similarity. Higher BLEU-4 scores mean the response text contains more exact 4-word phrases that match the reference text.
                
                | Score Range | Interpretation |
                |------------|----------------|
                | **0.70 - 1.00** | 🟢 **Excellent** - Very high similarity, almost identical content |
                | **0.50 - 0.69** | 🟡 **Good** - Strong similarity with minor differences |
                | **0.30 - 0.49** | 🟠 **Fair** - Moderate similarity, some key content matches |
                | **0.10 - 0.29** | 🔴 **Poor** - Low similarity, limited content overlap |
                | **0.00 - 0.09** | ⚫ **Very Poor** - Minimal to no similarity |
                
                #### 📊 ROUGE Scores
                **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)** measures both precision and recall - how well the response text captures the key information from the reference.
                
                | Score Range | Interpretation |
                |------------|----------------|
                | **0.70 - 1.00** | 🟢 **Excellent** - Captures most/all key information accurately |
                | **0.50 - 0.69** | 🟡 **Good** - Captures majority of important information |
                | **0.30 - 0.49** | 🟠 **Fair** - Captures some key information, missing details |
                | **0.10 - 0.29** | 🔴 **Poor** - Captures limited information |
                | **0.00 - 0.09** | ⚫ **Very Poor** - Fails to capture meaningful information |
                
                **ROUGE Variants:**
                - **ROUGE-1**: Measures overlap of unigrams (single words)
                - **ROUGE-2**: Measures overlap of bigrams (2-word phrases) - more strict
                - **ROUGE-L**: Measures longest common subsequence - captures sentence structure
                
                #### 💡 Key Insights
                - **BLEU** focuses on exact word/phrase matching (precision)
                - **ROUGE** focuses on information coverage (recall + precision)
                - **Higher scores** = Better alignment with reference text
                - **Lower scores** = More differences from reference text
                - Scores are **relative** - compare different responses against the same reference
                """)
            
            # Key Scores Summary
            st.markdown("### 📊 Key Scores Summary")
            bleu = metrics.get('bleu', {})
            rouge = metrics.get('rouge', {})
            
            # Display key scores in columns
            score_cols = st.columns(4)
            with score_cols[0]:
                bleu4_val = bleu.get('BLEU-4', 0) if bleu.get('BLEU-4') is not None else 0
                st.metric("BLEU-4", f"{bleu4_val:.4f}")
            with score_cols[1]:
                bleu_val = bleu.get('BLEU', 0) if bleu.get('BLEU') is not None else 0
                st.metric("BLEU (Overall)", f"{bleu_val:.4f}")
            with score_cols[2]:
                r1_key = 'ROUGE-1' if 'ROUGE-1' in rouge else ('ROUGE1' if 'ROUGE1' in rouge else None)
                r1_f1 = f"{rouge[r1_key]['f1']:.4f}" if r1_key and rouge.get(r1_key) else "0.0000"
                st.metric("ROUGE-1 F1", r1_f1)
            with score_cols[3]:
                rl_key = 'ROUGE-L' if 'ROUGE-L' in rouge else ('ROUGEL' if 'ROUGEL' in rouge else None)
                rl_f1 = f"{rouge[rl_key]['f1']:.4f}" if rl_key and rouge.get(rl_key) else "0.0000"
                st.metric("ROUGE-L F1", rl_f1)
            
            st.divider()
            
            # Detailed Metrics
            st.markdown("### 📈 Detailed Evaluation Metrics")
            
            # BLEU Scores
            if metrics.get('bleu'):
                st.markdown("#### 🎯 BLEU Scores")
                bleu = metrics['bleu']
                col1, col2, col3, col4, col5 = st.columns(5)
                with col1:
                    st.metric("BLEU", f"{bleu.get('BLEU', 0):.4f}")
                with col2:
                    st.metric("BLEU-1", f"{bleu.get('BLEU-1', 0):.4f}")
                with col3:
                    st.metric("BLEU-2", f"{bleu.get('BLEU-2', 0):.4f}")
                with col4:
                    st.metric("BLEU-3", f"{bleu.get('BLEU-3', 0):.4f}")
                with col5:
                    st.metric("BLEU-4", f"{bleu.get('BLEU-4', 0):.4f}")
            
            st.markdown("---")
            
            # ROUGE Scores
            if metrics.get('rouge'):
                st.markdown("#### 📊 ROUGE Scores")
                rouge = metrics['rouge']
                
                # ROUGE-1
                r1_key = 'ROUGE-1' if 'ROUGE-1' in rouge else ('ROUGE1' if 'ROUGE1' in rouge else None)
                if r1_key:
                    st.markdown("**ROUGE-1:**")
                    r1 = rouge[r1_key]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Precision", f"{r1.get('precision', 0):.4f}")
                    with col2:
                        st.metric("Recall", f"{r1.get('recall', 0):.4f}")
                    with col3:
                        st.metric("F1", f"{r1.get('f1', 0):.4f}")
                
                # ROUGE-2
                r2_key = 'ROUGE-2' if 'ROUGE-2' in rouge else ('ROUGE2' if 'ROUGE2' in rouge else None)
                if r2_key:
                    st.markdown("**ROUGE-2:**")
                    r2 = rouge[r2_key]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Precision", f"{r2.get('precision', 0):.4f}")
                    with col2:
                        st.metric("Recall", f"{r2.get('recall', 0):.4f}")
                    with col3:
                        st.metric("F1", f"{r2.get('f1', 0):.4f}")
                
                # ROUGE-L
                rl_key = 'ROUGE-L' if 'ROUGE-L' in rouge else ('ROUGEL' if 'ROUGEL' in rouge else None)
                if rl_key:
                    st.markdown("**ROUGE-L:**")
                    rl = rouge[rl_key]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Precision", f"{rl.get('precision', 0):.4f}")
                    with col2:
                        st.metric("Recall", f"{rl.get('recall', 0):.4f}")
                    with col3:
                        st.metric("F1", f"{rl.get('f1', 0):.4f}")
            
            st.divider()
            
            # Score Summary Analysis
            st.markdown("### 📋 Score Summary & Analysis")
            st.markdown("Detailed analysis of the evaluation scores based on the actual texts provided.")
            
            summary_export_data = None
            try:
                summary = generate_score_summary(
                    results.get('reference_text', ''),
                    results.get('response_text', ''),
                    metrics
                )
                st.markdown(summary)
                summary_export_data = prepare_score_summary_export(
                    results.get('reference_text', ''),
                    results.get('response_text', ''),
                    metrics,
                    summary
                )
            except Exception as e:
                st.warning(f"⚠️ Could not generate detailed analysis: {str(e)}")
            
            if summary_export_data:
                download_cols = st.columns(3)
                timestamp_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
                text_document = build_summary_text_document(summary_export_data)
                json_document = json.dumps(summary_export_data, indent=2, ensure_ascii=False)
                markdown_report = build_markdown_report(summary_export_data)
                
                with download_cols[0]:
                    st.download_button(
                        label="⬇️ Download Summary (Text)",
                        data=text_document,
                        file_name=f"text_evaluation_summary_{timestamp_suffix}.txt",
                        mime="text/plain",
                        key="text_eval_summary_download"
                    )
                with download_cols[1]:
                    st.download_button(
                        label="⬇️ Download Summary (JSON)",
                        data=json_document,
                        file_name=f"text_evaluation_summary_{timestamp_suffix}.json",
                        mime="application/json",
                        key="json_eval_summary_download"
                    )
                with download_cols[2]:
                    st.download_button(
                        label="⬇️ Download Summary (Markdown)",
                        data=markdown_report,
                        file_name=f"text_evaluation_summary_{timestamp_suffix}.md",
                        mime="text/markdown",
                        key="md_eval_summary_download"
                    )
                
                st.text_area(
                    "📋 Copy-ready Markdown (Excel friendly)",
                    value=markdown_report,
                    height=220,
                    key="md_eval_copy_area",
                    help="Select all (Ctrl+A) and copy to paste into a single Excel cell. Use Alt+Enter inside Excel to keep line breaks."
                )
            
            st.divider()
            
            # Text Comparison
            st.markdown("### 📄 Text Comparison")
            comp_col1, comp_col2 = st.columns(2)
            
            with comp_col1:
                st.markdown("**Reference Text:**")
                st.text_area(
                    "Reference",
                    value=results.get('reference_text', ''),
                    height=200,
                    disabled=True,
                    key="text_eval_ref_display",
                    label_visibility="collapsed"
                )
                st.caption(f"📏 Length: {len(results.get('reference_text', ''))} characters")
            
            with comp_col2:
                st.markdown("**Response Text:**")
                st.text_area(
                    "Response",
                    value=results.get('response_text', ''),
                    height=200,
                    disabled=True,
                    key="text_eval_resp_display",
                    label_visibility="collapsed"
                )
                st.caption(f"📏 Length: {len(results.get('response_text', ''))} characters")
            
            # Clear results button
            st.divider()
            if st.button("🔄 Clear Results", use_container_width=True):
                if 'text_evaluation_results' in st.session_state:
                    del st.session_state.text_evaluation_results
                st.rerun()

# Sidebar - removed Quick Links and Tip sections
