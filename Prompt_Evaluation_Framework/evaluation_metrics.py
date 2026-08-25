"""
Evaluation Metrics Module
Calculates BLEU and ROUGE scores for model outputs
"""
import nltk
from typing import List, Dict, Any
from rouge_score import rouge_scorer
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

def calculate_bleu_score(reference: str, candidate: str) -> Dict[str, float]:
    """
    Calculate BLEU score between reference and candidate text
    
    Args:
        reference: Reference text (ground truth)
        candidate: Candidate text (model output)
    
    Returns:
        Dictionary with BLEU scores (1-gram to 4-gram)
    """
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    
    # Tokenize
    reference_tokens = nltk.word_tokenize(reference.lower())
    candidate_tokens = nltk.word_tokenize(candidate.lower())
    
    # Prepare reference as list of lists (for multiple references support)
    reference_list = [reference_tokens]
    
    # Smoothing function to handle cases where n-grams don't match
    smoothing = SmoothingFunction().method1
    
    scores = {}
    
    # Calculate BLEU-1, BLEU-2, BLEU-3, BLEU-4
    for n in range(1, 5):
        weights = [1.0/n] * n + [0.0] * (4 - n)
        try:
            score = sentence_bleu(reference_list, candidate_tokens, weights=weights, smoothing_function=smoothing)
            scores[f'BLEU-{n}'] = round(score, 4)
        except:
            scores[f'BLEU-{n}'] = 0.0
    
    # Calculate overall BLEU score (standard 4-gram)
    try:
        overall_bleu = sentence_bleu(reference_list, candidate_tokens, smoothing_function=smoothing)
        scores['BLEU'] = round(overall_bleu, 4)
    except:
        scores['BLEU'] = 0.0
    
    return scores

def calculate_rouge_scores(reference: str, candidate: str) -> Dict[str, Dict[str, float]]:
    """
    Calculate ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) between reference and candidate
    
    Args:
        reference: Reference text (ground truth)
        candidate: Candidate text (model output)
    
    Returns:
        Dictionary with ROUGE scores (precision, recall, f1 for each metric)
    """
    try:
        # Use 'rougeL' (with capital L) for the rouge_score library
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference, candidate)
        
        result = {}
        for metric_name, metric_scores in scores.items():
            # Convert to display format: rouge1 -> ROUGE-1, rougeL -> ROUGE-L
            if metric_name == 'rouge1':
                display_name = 'ROUGE-1'
            elif metric_name == 'rouge2':
                display_name = 'ROUGE-2'
            elif metric_name == 'rougeL':
                display_name = 'ROUGE-L'
            else:
                display_name = metric_name.upper()
            
            result[display_name] = {
                'precision': round(metric_scores.precision, 4),
                'recall': round(metric_scores.recall, 4),
                'f1': round(metric_scores.fmeasure, 4)
            }
        
        return result
    except Exception as e:
        # If there's an error, try without stemmer
        try:
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
            scores = scorer.score(reference, candidate)
            
            result = {}
            for metric_name, metric_scores in scores.items():
                if metric_name == 'rouge1':
                    display_name = 'ROUGE-1'
                elif metric_name == 'rouge2':
                    display_name = 'ROUGE-2'
                elif metric_name == 'rougeL':
                    display_name = 'ROUGE-L'
                else:
                    display_name = metric_name.upper()
                
                result[display_name] = {
                    'precision': round(metric_scores.precision, 4),
                    'recall': round(metric_scores.recall, 4),
                    'f1': round(metric_scores.fmeasure, 4)
                }
            
            return result
        except Exception as e2:
            raise Exception(f"ROUGE calculation failed: {str(e2)}")

def evaluate_output(reference: str, candidate: str) -> Dict[str, Any]:
    """
    Calculate both BLEU and ROUGE scores for a candidate output
    
    Args:
        reference: Reference text (ground truth from PDF or provided)
        candidate: Candidate text (model output)
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    if not reference or not candidate:
        return {
            'bleu': {},
            'rouge': {},
            'error': 'Reference or candidate text is empty'
        }
    
    try:
        bleu_scores = calculate_bleu_score(reference, candidate)
        rouge_scores = calculate_rouge_scores(reference, candidate)
        
        return {
            'bleu': bleu_scores,
            'rouge': rouge_scores,
            'error': None
        }
    except Exception as e:
        return {
            'bleu': {},
            'rouge': {},
            'error': str(e)
        }


