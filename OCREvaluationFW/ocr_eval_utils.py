import unicodedata
import re
import jiwer
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz

def normalize_text(text, lowercase=True, remove_soft_hyphen=True):
    if text is None:
        return ""
    # unicode normalize
    text = unicodedata.normalize("NFC", text)
    if remove_soft_hyphen:
        text = text.replace("\u00ad", "")  # soft hyphen
    # normalize newlines and whitespace
    text = re.sub(r'\r\n?', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{2,}', '\n\n', text)
    if lowercase:
        text = text.lower()
    # trim
    return text.strip()

# Shared preprocessing for evaluation metrics (used by both single-run and multi-run)
def preprocess_markdown_for_evaluation(text):
    """
    Preprocess markdown text for fair comparison by normalizing formatting
    while preserving content structure.
    """
    if text is None:
        return ""
    # 1) Unicode normalization to NFC to canonicalize combining sequences
    try:
        text = unicodedata.normalize("NFC", text)
    except Exception:
        pass

    # 2) Remove zero-width and formatting characters that cause visually-identical mismatches
    #    Includes: ZWSP (200B), ZWNJ (200C), ZWJ (200D), BOM/ZWNBSP (FEFF), Word Joiner (2060)
    text = re.sub(r"[\u200B\u200C\u200D\uFEFF\u2060]", "", text)

    # 3) Normalize non-breaking spaces and narrow NBSP to regular spaces
    text = text.replace("\u00A0", " ").replace("\u202F", " ")

    # 4) Collapse excessive whitespace but preserve paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)       # Normalize spaces and tabs

    # 5) Trim per-line to remove trailing and leading spaces around content
    lines = text.split('\n')
    lines = [line.strip() for line in lines]
    text = '\n'.join(lines)

    return text.strip()

# Character Error Rate (CER) helper used by both paths
def cer(gt, pred):
    """Character Error Rate computed as Levenshtein distance normalized to [0, 1].

    We normalize by max(|GT|, |PRED|) so the value is always bounded in [0,1],
    even when there are large insertions relative to the ground truth.
    Uses RapidFuzz's Levenshtein to avoid double-counting substitutions.
    """
    gt_s = gt or ""
    pred_s = pred or ""
    denom = max(1, max(len(gt_s), len(pred_s)))
    try:
        dist = Levenshtein.distance(gt_s, pred_s)
    except Exception:
        # Fallback minimal behavior
        from difflib import SequenceMatcher
        ratio = SequenceMatcher(None, gt_s, pred_s).ratio()
        # Convert similarity ratio to distance approximation using denominator
        dist = int(round((1.0 - ratio) * denom))
    # Ensure numeric and bounded
    val = float(dist) / float(denom)
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val

def tokenize_words(text, language='en'):
    # Basic whitespace tokenization. Replace for language-specific tokenizer if available.
    tokens = re.findall(r'\S+', text)
    return tokens

def analyze_markdown_structure(gt_text: str, ocr_text: str):
    """
    Unified structural analyzer used by both single-run and multi-run paths.
    Returns (structure_metrics, structural_analysis).

    structure_metrics format:
      {
        'headers': {'gt': int, 'ocr': int, 'gt_texts': [str], 'ocr_texts': [str]},
        'tables': {'gt': int, 'ocr': int, 'gt_table_lines': [str], 'ocr_table_lines': [str]},
        'lists': {'gt': int, 'ocr': int},
        'links': {'gt': int, 'ocr': int, 'gt_targets': [str], 'ocr_targets': [str]},
      }

    structural_analysis keys:
      - heading_alignment [0..1]
      - list_accuracy     [0..1]
      - table_preservation[0..1]
      - link_correctness  [0..1]
      - section_ordering  [0..1]
    """
    import difflib
    
    # Ensure inputs are strings
    gt_text = str(gt_text) if gt_text is not None else ""
    ocr_text = str(ocr_text) if ocr_text is not None else ""

    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKC", s or "")
        s = s.lower()
        s = re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9]+', ' ', s)).strip()
        return s

    # Headers (markdown)
    header_pattern = re.compile(r'^(#{1,6})\s*(.+)$', re.MULTILINE)
    gt_headers_matches = header_pattern.findall(gt_text or "")
    ocr_headers_matches = header_pattern.findall(ocr_text or "")
    gt_header_texts = [h[1].strip() for h in gt_headers_matches]
    ocr_header_texts = [h[1].strip() for h in ocr_headers_matches]
    gt_hdr_norm = [_norm(h) for h in gt_header_texts]
    ocr_hdr_norm = [_norm(h) for h in ocr_header_texts]

    # Tables (simple pipe-based detection)
    table_line_pattern = re.compile(r'\|.*\|')
    gt_table_lines = table_line_pattern.findall(gt_text or "")
    ocr_table_lines = table_line_pattern.findall(ocr_text or "")
    gt_tables = len(gt_table_lines)
    ocr_tables = len(ocr_table_lines)

    # Lists (unordered and ordered)
    list_bullet = re.compile(r'^[\*\-\+]\s', re.MULTILINE)
    list_ordered = re.compile(r'^\d+\.\s', re.MULTILINE)
    gt_lists = len(list_bullet.findall(gt_text or "")) + len(list_ordered.findall(gt_text or ""))
    ocr_lists = len(list_bullet.findall(ocr_text or "")) + len(list_ordered.findall(ocr_text or ""))

    # Links
    link_pattern = re.compile(r'\[([^\]]+)\]\(([^)]+)\)')
    gt_links = link_pattern.findall(gt_text or "")
    ocr_links = link_pattern.findall(ocr_text or "")
    gt_targets = [t[1].strip() for t in gt_links]
    ocr_targets = [t[1].strip() for t in ocr_links]
    gt_targets_norm = set(_norm(t) for t in gt_targets if _norm(t))
    ocr_targets_norm = set(_norm(t) for t in ocr_targets if _norm(t))

    # Metrics computation

    # 1) Heading Alignment: greedy ordered matching with fuzzy threshold
    heading_alignment = 1.0
    section_ordering = 1.0
    if gt_hdr_norm or ocr_hdr_norm:
        # Greedy ordered match, keep order by advancing j
        THRESH = 0.80
        matches = []
        j_start = 0
        for i, g in enumerate(gt_hdr_norm):
            best_j = -1
            best_sim = 0.0
            for j in range(j_start, len(ocr_hdr_norm)):
                sim = difflib.SequenceMatcher(None, g, ocr_hdr_norm[j]).ratio()
                if sim > best_sim:
                    best_sim = sim
                    best_j = j
            if best_j >= j_start and best_sim >= THRESH:
                matches.append((i, best_j, best_sim))
                j_start = best_j + 1

        if not matches:
            heading_alignment = 0.0
            section_ordering = 1.0 if (len(gt_hdr_norm) <= 1 and len(ocr_hdr_norm) <= 1) else 0.5
        else:
            avg_sim = sum(m[2] for m in matches) / len(matches)
            coverage = len(matches) / max(1, max(len(gt_hdr_norm), len(ocr_hdr_norm)))
            heading_alignment = max(0.0, min(1.0, avg_sim * coverage))

            # Section ordering via LCS over matched OCR indices
            ocr_index_seq = [m[1] for m in matches]
            # LCS length
            def _lcs_len(seq):
                # patience sorting variant for LIS works only for strictly increasing sequences,
                # but we need LCS for order; since indices already increasing due to greedy, fallback simple:
                # when matches exist, sequence is non-decreasing by construction; treat as perfect order
                # If any inversion slipped, compute classic DP for short sequences
                n = len(seq)
                if n <= 1:
                    return n
                # Quick check: strictly increasing
                if all(seq[k] < seq[k+1] for k in range(n-1)):
                    return n
                # DP LCS with itself is n; to approximate, compute LIS length
                tails = []
                import bisect
                for x in seq:
                    i = bisect.bisect_left(tails, x)
                    if i == len(tails):
                        tails.append(x)
                    else:
                        tails[i] = x
                return len(tails)

            lcs = _lcs_len(ocr_index_seq)
            section_ordering = max(0.0, min(1.0, (lcs / max(1, len(matches))) * coverage))

    # 2) List Accuracy: proportional difference
    if gt_lists == 0 and ocr_lists == 0:
        list_accuracy = 1.0
    else:
        list_accuracy = 1.0 - (abs(gt_lists - ocr_lists) / max(1, max(gt_lists, ocr_lists)))
        list_accuracy = max(0.0, min(1.0, list_accuracy))

    # 3) Table Preservation: enhanced with content similarity
    def _estimate_columns_from_lines(lines):
        cols = []
        for ln in lines:
            # count '|' segments minus edges
            parts = [p for p in ln.strip().split('|') if p.strip() != ""]
            if len(parts) >= 2:
                cols.append(len(parts))
        return cols

    def _extract_table_content(lines):
        """Extract normalized table cell content for comparison"""
        content = set()
        for ln in lines:
            parts = [_norm(p) for p in ln.strip().split('|') if p.strip() != ""]
            content.update(p for p in parts if p)
        return content

    presence = 0.0
    col_score = 0.0
    content_score = 0.0
    
    if gt_tables == 0 and ocr_tables == 0:
        # No tables present in either GT or OCR => metric is not applicable.
        # Use None to indicate "NA" so reporting layers can display it appropriately.
        table_preservation = None
    else:
        # Presence score: how many table lines are preserved
        presence = min(gt_tables, ocr_tables) / max(1, max(gt_tables, ocr_tables))

        # Column structure score
        gt_cols_list = _estimate_columns_from_lines(gt_table_lines)
        ocr_cols_list = _estimate_columns_from_lines(ocr_table_lines)
        if gt_cols_list and ocr_cols_list:
            pairs = min(len(gt_cols_list), len(ocr_cols_list))
            sims = []
            for i in range(pairs):
                a, b = gt_cols_list[i], ocr_cols_list[i]
                sims.append(min(a, b) / max(1, max(a, b)))
            col_score = sum(sims) / len(sims) if sims else 0.0

        # Content similarity score
        gt_content = _extract_table_content(gt_table_lines)
        ocr_content = _extract_table_content(ocr_table_lines)
        if gt_content or ocr_content:
            union = gt_content | ocr_content
            inter = gt_content & ocr_content
            content_score = len(inter) / max(1, len(union))
        else:
            content_score = 1.0

        # Weighted combination: presence (40%), column structure (30%), content (30%)
        table_preservation = max(0.0, min(1.0, 0.4 * presence + 0.3 * col_score + 0.3 * content_score))

    # 4) Link correctness: Jaccard of targets
    if not gt_targets_norm and not ocr_targets_norm:
        link_correctness = 1.0
    else:
        union = gt_targets_norm | ocr_targets_norm
        inter = gt_targets_norm & ocr_targets_norm
        link_correctness = len(inter) / max(1, len(union))

    structure_metrics = {
        'headers': {
            'gt': len(gt_header_texts),
            'ocr': len(ocr_header_texts),
            'gt_texts': [h.lower() for h in gt_header_texts],
            'ocr_texts': [h.lower() for h in ocr_header_texts],
        },
        'tables': {
            'gt': gt_tables,
            'ocr': ocr_tables,
            'gt_table_lines': gt_table_lines,
            'ocr_table_lines': ocr_table_lines,
        },
        'lists': {
            'gt': gt_lists,
            'ocr': ocr_lists,
        },
        'links': {
            'gt': len(gt_targets),
            'ocr': len(ocr_targets),
            'gt_targets': gt_targets,
            'ocr_targets': ocr_targets,
        }
    }

    structural_analysis = {
        'heading_alignment': float(max(0.0, min(1.0, heading_alignment))),
        'list_accuracy': float(max(0.0, min(1.0, list_accuracy))),
        'table_preservation': None if table_preservation is None else float(max(0.0, min(1.0, table_preservation))),
        'link_correctness': float(max(0.0, min(1.0, link_correctness))),
        'section_ordering': float(max(0.0, min(1.0, section_ordering))),
    }

    return structure_metrics, structural_analysis


def compute_top_word_mismatches(gt_text, ocr_text, top_n=10):
    """
    Compute top word mismatches between ground truth and OCR text.
    
    Args:
        gt_text: Ground truth text (preprocessed)
        ocr_text: OCR output text (preprocessed)
        top_n: Number of top mismatches to return
        
    Returns:
        List of tuples: [(gt_word, ocr_word), count] sorted by frequency
    """
    from collections import Counter
    import difflib
    
    gt_words = gt_text.split()
    ocr_words = ocr_text.split()

    matcher = difflib.SequenceMatcher(None, gt_words, ocr_words)
    mismatches = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            for g, o in zip(gt_words[i1:i2], ocr_words[j1:j2]):
                mismatches.append((g, o))
        elif tag == "delete":
            for g in gt_words[i1:i2]:
                mismatches.append((g, "<MISSING>"))
        elif tag == "insert":
            for o in ocr_words[j1:j2]:
                mismatches.append(("<EXTRA>", o))

    counter = Counter(mismatches)
    return counter.most_common(top_n)


def compute_all_metrics(gt_text: str, ocr_text: str):
    """
    Unified metrics computation used by both single-run and multi-run paths.
    Returns a dict with all metrics and raw mismatches.
    """
    # Ensure inputs are strings (type validation)
    gt_text = str(gt_text) if gt_text is not None else ""
    ocr_text = str(ocr_text) if ocr_text is not None else ""
    
    # Preprocess texts
    gt_processed = preprocess_markdown_for_evaluation(gt_text or "")
    ocr_processed = preprocess_markdown_for_evaluation(ocr_text or "")
    
    # Word-level metrics
    try:
        wer = jiwer.wer(gt_processed, ocr_processed)
        # Clamp WER to [0,1] to avoid values >1 due to heavy insertions
        wer = float(max(0.0, min(1.0, float(wer))))
        mer = jiwer.mer(gt_processed, ocr_processed)
        wil = jiwer.wil(gt_processed, ocr_processed)
    except Exception as e:
        print(f"Warning: Error computing word-level metrics: {e}")
        wer = mer = wil = 0.0
    
    # Character error rate
    cer_value = cer(gt_processed, ocr_processed)
    
    # Levenshtein metrics
    try:
        # Ensure strings before computing distance
        gt_str = str(gt_processed) if gt_processed else ""
        ocr_str = str(ocr_processed) if ocr_processed else ""
        lev_distance = int(Levenshtein.distance(gt_str, ocr_str))
        # Normalize by max length to bound in [0,1]
        lev_norm_denom = max(1, max(len(gt_str), len(ocr_str)))
        lev_norm = float(lev_distance) / float(lev_norm_denom)
    except Exception as e:
        print(f"Warning: Error computing Levenshtein distance: {e}")
        lev_distance = 0
        lev_norm = 0.0
    
    # Structural analysis (works on raw markdown, not preprocessed)
    structure_metrics, structural_analysis = analyze_markdown_structure(gt_text or "", ocr_text or "")
    
    # Completeness - ensure strings and proper type conversion
    try:
        ocr_len = int(len(str(ocr_processed)))
        gt_len = int(len(str(gt_processed)))
        completeness = float(min(1.0, ocr_len / max(1, gt_len)))
    except Exception as e:
        print(f"Warning: Error computing completeness: {e}")
        completeness = 0.0
    
    # Mismatches
    top_mismatches_raw = compute_top_word_mismatches(gt_processed, ocr_processed, top_n=10)
    
    return {
        'wer': wer,
        'mer': mer,
        'wil': wil,
        'cer': cer_value,
        'lev_distance': lev_distance,
        'lev_norm': lev_norm,
        'completeness': completeness,
        'structural_accuracy': structure_metrics,
        'structural_analysis': structural_analysis,
        'top_mismatches_raw': top_mismatches_raw
    }