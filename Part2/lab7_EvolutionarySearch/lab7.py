"""
Simple Genetic Algorithm for Word Embedding Insertion
======================================================================
A (1+λ) Evolution Strategy for inserting new word embeddings into a trained
Skip-Gram model while preserving the existing embedding space structure.
"""

import torch
import numpy as np
import re
from collections import Counter
from typing import Dict, List, Tuple, Optional, Set

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchvision

from src.lab6_current_best import SkipGramModel, find_similar_words
from src.lab2 import process_text_network

import unittest
import tempfile
import os


# ============================================================================
# DATA LOADING & PREPARATION
# ============================================================================

def load_trained_model(model_path: str, vocab_size: int, 
                       embedding_dim: int, dropout: float) -> Tuple[torch.nn.Module, np.ndarray]:
    """Load trained Skip-Gram model and extract embeddings."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(model_path, map_location=device)
    
    model = SkipGramModel(vocab_size=vocab_size, embedding_dim=embedding_dim, dropout=dropout).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    with torch.no_grad():
        embeddings_tensor = model.get_embeddings()
        embeddings = (embeddings_tensor.cpu().numpy() if isinstance(embeddings_tensor, torch.Tensor) 
                     else embeddings_tensor).astype(np.float32)
    
    print(f"✓ Loaded model: {embeddings.shape[0]} embeddings, dim={embeddings.shape[1]}")
    return model, embeddings


def create_mappings(nodes: List[str]) -> Tuple[Dict[str, int], Dict[int, str], Dict[str, np.ndarray]]:
    """Create word-to-index and index-to-word mappings."""
    word_to_idx = {word: idx for idx, word in enumerate(nodes)}
    idx_to_word = {idx: word for idx, word in enumerate(nodes)}
    return word_to_idx, idx_to_word


def compute_embedding_stats(embeddings: np.ndarray) -> Dict[str, float]:
    """Compute statistics needed for fitness evaluation."""
    norms = np.linalg.norm(embeddings, axis=1)
    return {
        'mean_norm': np.mean(norms),
        'std_norm': np.std(norms),
        'global_std': np.std(embeddings)
    }


def get_cifar100_vocabulary() -> List[str]:
    """Download CIFAR-100 and extract class names."""
    print("\nLoading CIFAR-100 vocabulary...")
    dataset = torchvision.datasets.CIFAR100(root='./cifar100_data', train=True, download=True)
    print(f"✓ CIFAR-100 vocabulary loaded: {len(dataset.classes)} classes")
    return dataset.classes


def analyze_vocabulary_overlap(cifar_vocab: List[str], network_vocab: List[str]) -> List[str]:
    """Analyze overlap between CIFAR-100 and network vocabulary."""
    cifar_set, network_set = set(cifar_vocab), set(network_vocab)
    overlapping = sorted(list(cifar_set.intersection(network_set)))
    missing = sorted(list(cifar_set - network_set))
    
    print(f"\n{'='*70}")
    print("VOCABULARY OVERLAP ANALYSIS")
    print(f"{'='*70}")
    print(f"CIFAR-100 vocabulary: {len(cifar_set)} classes")
    print(f"Network vocabulary: {len(network_set)} words")
    print(f"Overlapping words: {len(overlapping)} ({len(overlapping)/len(cifar_set)*100:.1f}%)")
    print(f"Missing from network: {len(missing)}")
    if overlapping:
        print(f"\nFound: {', '.join(overlapping)}")
    if missing:
        print(f"\nMissing: {', '.join(missing)}")
    print(f"{'='*70}\n")
    
    return missing


# ============================================================================
# CONTEXT EXTRACTION
# ============================================================================

import re
from collections import Counter
from typing import List, Dict, Optional, Set

def extract_word_contexts(
    text_file: str,
    target_words: List[str],
    vocab_set: Set[str],
    window: int = 5,
    stopwords: Optional[Set[str]] = None,
    idf=None,
    idf_threshold: float = 1.8,
    idf_exempt_targets: Optional[Set[str]] = None
) -> Dict[str, Counter]:

    contexts = {word: Counter() for word in target_words}
    stop = stopwords if stopwords is not None else set()
    exempt = idf_exempt_targets if idf_exempt_targets is not None else set()

    single_targets = set()
    phrase_targets = {}  # e.g. 'pickup_truck' -> ['pickup','truck']

    for w in target_words:
        if "_" in w:
            parts = [p for p in w.split("_") if p.isalpha()]
            if len(parts) >= 2:
                phrase_targets[w] = parts
            else:
                single_targets.add(w)
        else:
            single_targets.add(w)

    phrases_by_first = {}
    for phrase_key, parts in phrase_targets.items():
        phrases_by_first.setdefault(parts[0], []).append((phrase_key, parts))

    with open(text_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i % 50000 == 0 and i > 0:
                print(f" .........processed {i:,} lines")

            tokens = re.findall(r"\b[a-z]+\b", line.lower())
            n = len(tokens)
            idx = 0

            while idx < n:
                tok = tokens[idx]

                # ---- Case A: phrase targets (underscore tokens)
                matched_phrase = False
                if tok in phrases_by_first:
                    for phrase_key, parts in phrases_by_first[tok]:
                        L = len(parts)
                        if idx + L <= n and tokens[idx:idx+L] == parts:
                            start = max(0, idx - window)
                            end = min(n, idx + L + window)

                            left_ctx = tokens[start:idx]
                            right_ctx = tokens[idx+L:end]

                            for ctx in left_ctx + right_ctx:
                                if ctx not in vocab_set:
                                    continue
                                if ctx in stop:
                                    continue
                                # IMPORTANT: check exemption using phrase_key, not tok
                                if phrase_key not in exempt:
                                    if idf_threshold is not None and idf is not None:
                                        if idf.get(ctx, 0.0) < idf_threshold:
                                            continue
                                contexts[phrase_key][ctx] += 1

                            idx += L
                            matched_phrase = True
                            break

                if matched_phrase:
                    continue

                # ---- Case B: single-token targets (original behavior)
                if tok in single_targets:
                    start = max(0, idx - window)
                    end = min(n, idx + window + 1)

                    for ctx in tokens[start:idx] + tokens[idx+1:end]:
                        if ctx not in vocab_set:
                            continue
                        if ctx in stop:
                            continue
                        if tok not in exempt:
                            if idf_threshold is not None and idf is not None:
                                if idf.get(ctx, 0.0) < idf_threshold:
                                    continue
                        contexts[tok][ctx] += 1

                idx += 1

    print(" Complete \n\nContext statistics: ")
    for word in target_words:
        print(f"    {word:15s}: {sum(contexts[word].values()):6d} contexts, {len(contexts[word]):3d} unique words")

    return contexts








# ============================================================================
# FITNESS FUNCTION
# ============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

def softmax_stable(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / (np.sum(ex) + 1e-12)


def compute_fitness(
    vec: np.ndarray,
    word: str,
    ctx_vecs: Optional[np.ndarray],
    ctx_weights: Optional[np.ndarray],
    ctx_mass_raw,
    neg_vecs: np.ndarray,
    anchor_vecs: Optional[np.ndarray],
    stats_dict: Dict[str, float],
    weights: Dict[str, float]
) -> float:
    """
    Compute a three-term fitness score for a candidate word embedding vector.
    
    This function evaluates how well a candidate vector fits the learned 
    embedding space by combining three complementary metrics:
    1. Corpus likelihood (how well it predicts observed contexts)
    2. Norm matching (how similar its magnitude is to typical embeddings)
    3. Anchor similarity (how similar it is to known reference words)
    
    Args:
        vec: Candidate embedding vector to evaluate.
        word: Target word (for reference, not used in computation).
        ctx_vecs: Context word vectors that co-occur with the target word.
                  Shape: (n_contexts, embedding_dim). May be None if no contexts.
        ctx_weights: Weights for each context (e.g., co-occurrence counts).
                     Shape: (n_contexts,). May be None if no contexts.
        neg_vecs: Negative sample vectors (words that don't co-occur).
                  Shape: (n_negatives, embedding_dim).
        anchor_vecs: Pre-normalized vectors of anchor words for comparison.
                     Shape: (n_anchors, embedding_dim). May be None.
        stats_dict: Dictionary containing embedding statistics:
                    - 'mean_norm': Average L2 norm of embeddings in the space
                    - 'std_norm': Standard deviation of embedding norms
                    - 'global_std': Global standard deviation (if needed)
        weights: Dictionary of weights for each fitness component:
                 - 'corpus': Weight for corpus likelihood term
                 - 'norm': Weight for norm matching term
                 - 'anchor': Weight for anchor similarity term
    
    Returns:
        Combined fitness score in the range [0, 1], where higher is better.
        
    Example:
        >>> vec = np.array([0.5, -0.3, 0.8, 0.1])
        >>> stats = {'mean_norm': 1.0, 'std_norm': 0.2, 'global_std': 0.5}
        >>> weights = {'corpus': 0.5, 'norm': 0.3, 'anchor': 0.2}
        >>> fitness = compute_fitness(vec, 'king', ctx_vecs, ctx_weights, 
        ...                           neg_vecs, anchor_vecs, stats, weights)
        >>> print(f"Fitness: {fitness:.4f}")
        Fitness: 0.7234
    
    Implementation guidelines:
    --------------------------
    Term 1 - Corpus Likelihood (L_corpus_norm):
        - For positive contexts: sum over ctx_weights * log(sigmoid(ctx_vecs · vec))
        - For negative samples: sum over log(sigmoid(-neg_vecs · vec))
        - Add small epsilon (1e-10) inside log for numerical stability
        - Normalize by total samples, then apply sigmoid to map to [0, 1]
        - Default to 0.5 if no samples available
        
    Term 2 - Norm Match (S_norm):
        - Compute L2 norm of the candidate vector
        - Use Gaussian similarity: exp(-((norm - mean_norm)² / (2 * std_norm²)))
        - This rewards vectors with norms close to the typical embedding norm
        
    Term 3 - Anchor Similarity (S_anchor):
        - Normalize the candidate vector (divide by its norm + epsilon)
        - Compute dot products with all anchor vectors (they're pre-normalized)
        - Take the mean similarity across all anchors
        - Default to 0.5 if no anchors provided
        
    Final score:
        - Weighted sum: weights['corpus'] * L_corpus_norm + 
                       weights['norm'] * S_norm + 
                       weights['anchor'] * S_anchor
    
    Notes:
        - Handle None values for optional parameters (ctx_vecs, ctx_weights, anchor_vecs)
        - Use vectorized NumPy operations for efficiency
        - Add small epsilon values to prevent division by zero
    """
    # -----------------------
    # Adaptive fitness weights
    # -----------------------
    ctx_mass = float(ctx_mass_raw) if ctx_mass_raw is not None else 0.0
    if ctx_vecs is None or ctx_weights is None:
        ctx_mass = 0.0


    # override weights based on context strength
    if ctx_mass == 0.0:
        weights = {"corpus": 0.70, "norm": 0.15, "anchor": 0.15}
    elif ctx_mass < 30.0:
        weights = {"corpus": 0.70, "norm": 0.15, "anchor": 0.15}
    elif ctx_mass < 100.0:
        weights = {"corpus": 0.70, "norm": 0.15, "anchor": 0.15}
    else:
        weights = {"corpus": 0.70, "norm": 0.15, "anchor": 0.15}
    
    vec_norm = np.linalg.norm(vec)
    
    # Corpus likelihood term
    L_corpus = 0.0
    if ctx_vecs is not None:
        L_corpus += np.sum(ctx_weights * np.log(sigmoid(np.dot(ctx_vecs, vec)) + 1e-10 ))

    # L_corpus += np.sum(np.log(sigmoid(-np.dot(neg_vecs, vec)) + 1e-10)) # Hint: This can be made better by adding weights to the negatives


    neg_dot = np.dot(neg_vecs, vec)  # (num_negatives,)

    # cosine similarity ΜΟΝΟ για weighting
    vec_n = vec / (np.linalg.norm(vec) + 1e-10)
    neg_n = neg_vecs / (np.linalg.norm(neg_vecs, axis=1, keepdims=True) + 1e-10)
    neg_sim = neg_n @ vec_n

    if ctx_mass >= 200:
        tau = 1.2
        mix = 0.25
    elif ctx_mass >= 30:
        tau = 1.5
        mix = 0.20
    elif ctx_mass > 0:
        tau = 2.0
        mix = 0.15
    else:
        tau = None
        mix = 0.0

    if ctx_mass == 0.0:
        w_neg = np.ones(len(neg_vecs), dtype=np.float32) / len(neg_vecs)
    else:
        w_neg = softmax_stable(neg_sim / tau)
        w_neg = mix * w_neg + (1 - mix) * (np.ones_like(w_neg) / len(w_neg))

    cap = 0.20
    w_neg = np.minimum(w_neg, cap)
    w_neg = w_neg / (np.sum(w_neg) + 1e-12)

    #w_neg = np.ones(len(neg_vecs), dtype=np.float32) / len(neg_vecs)


    L_corpus += np.sum(w_neg * np.log(sigmoid(-neg_dot) + 1e-10))

    # total_samples = (np.sum(ctx_weights) if ctx_vecs is not None else 0) + len(neg_vecs)
    n_pos = 0 if (ctx_vecs is None or ctx_weights is None) else len(ctx_weights)
    total_samples = float(n_pos) + float(len(neg_vecs))


    L_corpus_norm = sigmoid(L_corpus / total_samples) if total_samples > 0 else 0.5

    # embedding belonging term
    S_norm = np.exp(-((vec_norm - stats_dict['mean_norm']) ** 2 ) / (2 * stats_dict['std_norm']**2)) # Edw to chat leei oti isws thelei ena tetragwno sto telos

    # Anchor term
    S_anchor = 0.5
    if anchor_vecs is not None:
        S_anchor = np.mean(np.dot(anchor_vecs, vec / (vec_norm + 1e-10)))
    
    return (weights['corpus'] * L_corpus_norm + \
            weights['norm'] * S_norm + \
            weights['anchor'] * S_anchor)


# ============================================================================
# GENETIC ALGORITHM (1+λ) EVOLUTION STRATEGY
# ============================================================================



def select_by_coverage(
    scored: List[Tuple[str, float]],
    coverage: float = 0.80,
    k_min: int = 10,
    k_max: int = 60
) -> List[Tuple[str, float]]:
    """
    Given a list of (item, score) sorted descending, keep top items until
    cumulative score mass reaches `coverage` of total mass, with bounds.
    """
    if not scored:
        return []

    total = sum(s for _, s in scored)
    if total <= 0:
        return []

    picked = []
    running = 0.0
    for item, s in scored:
        if s <= 0:
            continue
        picked.append((item, s))
        running += s

        K = len(picked)
        if (K >= k_min and (running / total) >= coverage) or (K >= k_max):
            break

    return picked



def initialize_embedding(
    word: str,
    contexts: Dict[str, Counter],
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int],
    stats_dict: Dict[str, float],
    anchors: Optional[Dict[str, List[str]]] = None,
    idf: Optional[Dict[str, float]] = None,
    idf_threshold: Optional[float] = None,
    idf_power_init: float = 1.0,
    coverage: float = 0.80,
    k_min: int = 10,
    k_max: int = 60,
    alpha: float = 0.7,   # context contribution
    beta: float = 0.3     # anchor contribution
) -> np.ndarray:
    """
    Initialization = normalized( alpha * v_ctx + beta * v_anc ) * mean_norm

    - v_ctx: weighted centroid of top contexts selected by coverage mass
    - v_anc: centroid of anchor embeddings (prior)
    - idf_threshold: optional filter to drop too-common contexts at init
    """

    dim = embeddings.shape[1]
    mean_norm = float(stats_dict["mean_norm"])

    # -----------------------------
    # 1) Build v_ctx from contexts
    # -----------------------------
    ctx_vec = None
    ctr = contexts.get(word, None)

    if ctr and len(ctr) > 0:
        scored = []
        for ctx, count in ctr.items():
            if ctx not in word_to_idx:
                continue

            if idf_threshold is not None and idf is not None:
                if idf.get(ctx, 0.0) < idf_threshold:
                    continue

            ctx_idf = idf.get(ctx, 1.0) if idf is not None else 1.0
            score = float(count) * (float(ctx_idf) ** float(idf_power_init))
            scored.append((ctx, score))

        # rank by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        picked = select_by_coverage(scored, coverage=coverage, k_min=k_min, k_max=k_max)

        if picked:
            ws = np.array([s for _, s in picked], dtype=np.float32)
            wsum = float(ws.sum())
            ws = ws / (wsum + 1e-12)

            vecs = np.stack([embeddings[word_to_idx[c]] for c, _ in picked]).astype(np.float32)
            ctx_vec = (ws[:, None] * vecs).sum(axis=0)

            # If something went wrong (rare), drop it
            if float(np.linalg.norm(ctx_vec)) < 1e-12:
                ctx_vec = None

    # -----------------------------
    # 2) Build v_anc from anchors
    # -----------------------------
    anc_vec = None
    if anchors is not None:
        anc_words = anchors.get(word, [])
        anc_vecs = []
        for a in anc_words:
            if a in word_to_idx:
                anc_vecs.append(embeddings[word_to_idx[a]])
        if anc_vecs:
            anc_vec = np.mean(np.stack(anc_vecs).astype(np.float32), axis=0)
            if float(np.linalg.norm(anc_vec)) < 1e-12:
                anc_vec = None

    # -----------------------------
    # 3) Combine (with fallbacks)
    # -----------------------------
    if ctx_vec is None and anc_vec is None:
        # fallback: mean embedding
        v = np.mean(embeddings, axis=0).astype(np.float32)
    elif ctx_vec is None:
        v = anc_vec
    elif anc_vec is None:
        v = ctx_vec
    else:
        v = (alpha * ctx_vec) + (beta * anc_vec)

    # -----------------------------
    # 4) Norm-match to space
    # -----------------------------
    n = float(np.linalg.norm(v) + 1e-12)
    v = (v / n) * mean_norm

    return v


def top_contexts_for_negatives(
    ctr: Counter,
    idf: Optional[Dict[str, float]] = None,
    idf_power: float = 1.0,
    coverage: float = 0.80,
    k_min: int = 5,
    k_max: int = 60
) -> Set[str]:
    if not ctr:
        return set()

    scored = []
    for w, c in ctr.items():
        score = float(c)
        if idf is not None:
            score *= float(idf.get(w, 1.0)) ** float(idf_power)
        scored.append((w, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    total = sum(s for _, s in scored)
    if total <= 0:
        return set(w for w, _ in scored[:k_min])

    chosen = []
    running = 0.0
    for w, s in scored:
        chosen.append(w)
        running += s
        if len(chosen) >= k_min and (running / total) >= coverage:
            break
        if len(chosen) >= k_max:
            break

    return set(chosen)



  


def precompute_fitness_vectors(
    word: str,
    contexts: Dict[str, Counter],
    embeddings: np.ndarray,
    word_to_idx: Dict[str, int],
    vocab_list: List[str],
    anchors: Dict[str, List[str]],
    num_negatives: int = 15,
    idf: Optional[Dict[str, float]] = None,
    idf_power: float = 1.0,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], np.ndarray, Optional[np.ndarray]]:
    """
    Precompute all vectors needed for fitness evaluation.
    
    This function extracts and prepares the three types of vectors used in
    fitness computation: positive context vectors, negative sample vectors,
    and anchor vectors. Precomputing these vectors once improves efficiency
    when evaluating fitness multiple times during optimization.
    
    Args:
        word: Target word being optimized.
        contexts: Dictionary mapping words to their co-occurrence contexts.
                  Each value is a Counter with {context_word: count}.
        embeddings: Pre-trained embedding matrix. Shape: (vocab_size, embedding_dim).
        word_to_idx: Dictionary mapping words to their row indices in embeddings.
        vocab_list: List of all vocabulary words (for negative sampling).
        anchors: Dictionary mapping words to lists of semantically related anchor words.
        num_negatives: Number of negative samples to draw (default: 15).
    
    Returns:
        Tuple of (ctx_vecs, ctx_weights, neg_vecs, anchor_vecs):
        - ctx_vecs: Context word embeddings. Shape: (n_contexts, dim) or None.
        - ctx_weights: Normalized context weights. Shape: (n_contexts,) or None.
        - neg_vecs: Negative sample embeddings. Shape: (num_negatives, dim).
        - anchor_vecs: Normalized anchor embeddings. Shape: (n_anchors, dim) or None.
        
    Example:
        >>> contexts = {'king': Counter({'queen': 50, 'royal': 30})}
        >>> anchors = {'king': ['queen', 'monarch', 'ruler']}
        >>> ctx_v, ctx_w, neg_v, anc_v = precompute_fitness_vectors(
        ...     'king', contexts, embeddings, word_to_idx, vocab_list, anchors
        ... )
        >>> ctx_v.shape  # Positive contexts
        (2, 300)
        >>> neg_v.shape  # Negative samples
        (15, 300)
    
    Implementation guidelines:
    --------------------------
    Part 1 - Positive Context Vectors:
        - Initialize ctx_vecs and ctx_weights to None (for no-context case)
        - If the word has contexts:
            * Iterate through contexts[word].items()
            * For each context word that exists in word_to_idx:
              - Collect its embedding vector
              - Collect its count
            * If any valid contexts found:
              - Convert lists to numpy arrays
              - Normalize weights to sum to 1.0
    
    Part 2 - Negative Sample Vectors:
        - Randomly sample num_negatives words from vocab_list (without replacement)
        - Look up their embeddings and stack into an array
        - Shape should be (num_negatives, embedding_dim)
    
    Part 3 - Anchor Vectors:
        - Initialize anchor_vecs to None (for no-anchor case)
        - If the word has anchors defined:
            * Filter to only anchors that exist in word_to_idx
            * If any valid anchors found:
              - Collect their embeddings into an array
              - Normalize each vector to unit length (L2 norm = 1)
              - Use np.linalg.norm with axis=1, keepdims=True
              - Add small epsilon (1e-10) to prevent division by zero
    
    Notes:
        - Handle missing words gracefully (skip if not in word_to_idx)
        - Return None for optional components if no valid data available
        - Negative samples should be random to avoid bias
        - Anchor normalization enables direct cosine similarity via dot product
    """
    # Positive context
    ctx_vecs, ctx_weights = None, None
    ctx_mass_raw = 0.0

    ctr = contexts.get(word)
    if ctr:
        vec, weights_raw = [], []
        for w, count in ctr.items():
            if w in word_to_idx:
                vec.append(embeddings[word_to_idx[w]])

                # IDF-weighted counts
                if idf is not None:
                    w_idf = idf.get(w, 1.0)
                    weight = float(count) * (float(w_idf) ** float(idf_power))
                else:
                    weight = float(count)

                weights_raw.append(weight)

        if vec:
            ctx_vecs = np.array(vec, dtype=np.float32)
            weights_raw = np.array(weights_raw, dtype=np.float32)

            ctx_mass_raw = float(np.sum(weights_raw))  # <-- RAW strength (counts / idf-counts)

            # keep normalized weights for likelihood term (stable)
            ctx_weights = (weights_raw / (ctx_mass_raw + 1e-12)) if ctx_mass_raw > 0 else None
    
    
    # -----------------------------
    # Negative sampling (CLEAN)
    # -----------------------------
    forbidden = {word}

    # top contexts by coverage
    ctr = contexts.get(word, Counter())
    top_ctx = top_contexts_for_negatives(
        ctr,
        idf=idf,
        idf_power=idf_power,
        coverage=0.80,
        k_min=5,
        k_max=60
    )
    forbidden |= top_ctx

    # anchors
    if anchors is not None:
        forbidden |= set(anchors.get(word, []))

    # allowed negatives
    allowed = [w for w in vocab_list if w not in forbidden and w in word_to_idx]

    if len(allowed) < num_negatives:
        # fallback: sample with replacement
        neg_words = np.random.choice(allowed, num_negatives, replace=True)
    else:
        neg_words = np.random.choice(allowed, num_negatives, replace=False)

    neg_vecs = np.array([embeddings[word_to_idx[n]] for n in neg_words], dtype=np.float32)
    

    # Anchors
    anchor_vecs = None
    if word in anchors:
        anchor_words = [a for a in anchors[word] if a in word_to_idx]
        if anchor_words:
            anchor_vecs = np.array([embeddings[word_to_idx[a]] for a in anchor_words])
            anchor_vecs = anchor_vecs / (np.linalg.norm(anchor_vecs, axis=1, keepdims=True) + 1e-10)

    return ctx_vecs, ctx_weights,ctx_mass_raw, neg_vecs, anchor_vecs


def evolve_embedding(word: str, contexts: Dict[str, Counter], 
                    embeddings: np.ndarray, word_to_idx: Dict[str, int],
                    vocab_list: List[str], stats_dict: Dict[str, float],
                    anchors: Dict[str, List[str]], config: Dict, idf = None, idf_threshold = None, idf_power = 1.0, num_negatives = None) -> np.ndarray:
    """
    Evolve a single word embedding using (1+λ) Evolution Strategy.
    
    Args:
        word: Target word to insert
        contexts: Context word counts for all target words
        embeddings: Existing embedding matrix
        word_to_idx: Word to index mapping
        vocab_list: List of vocabulary words
        stats_dict: Embedding statistics
        anchors: Anchor words for semantic guidance
        config: Configuration dictionary
    
    Returns:
        Optimized embedding vector
    """
    print(f"\n  Evolving: '{word}'", end='')
    
    dim = embeddings.shape[1]
    mutation_sigma = config['ga_mutation_factor'] * stats_dict['global_std']

    
    # Initialize
    best_vec = initialize_embedding(word, contexts, embeddings, word_to_idx,stats_dict=stats_dict,
    anchors=anchors,
    idf= idf,
    idf_threshold= idf_threshold,
    idf_power_init= 1.0,
    coverage= 0.80,
    k_min = 5,
    k_max= 60,
    alpha= 0.7,
    beta= 0.3)
    
    ctx_vecs, ctx_weights, ctx_mass_raw, neg_vecs, anchor_vecs = precompute_fitness_vectors(
        word, contexts, embeddings, word_to_idx, vocab_list, anchors,
        num_negatives=num_negatives, idf=idf, idf_power=idf_power
    )   

    best_fit = compute_fitness(
        best_vec, word, ctx_vecs, ctx_weights, ctx_mass_raw,
        neg_vecs, anchor_vecs, stats_dict, config['fitness_weights'])
    
    # Evolution loop
    for gen in range(config['ga_generations']):

        # Generate offspring and evaluate
        population = best_vec + np.random.normal(0, mutation_sigma, (config['ga_pop_size'], dim))
        all_candidates = np.vstack([best_vec, population])
        
        fitness_scores = [
            compute_fitness(vec, word, ctx_vecs, ctx_weights, ctx_mass_raw,
                            neg_vecs, anchor_vecs, stats_dict, config['fitness_weights'])
            for vec in all_candidates
        ]
        
        # Select best
        best_idx = np.argmax(fitness_scores)
        best_vec = all_candidates[best_idx].copy()
        best_fit = fitness_scores[best_idx]
        
        if gen % 50 == 0:
            print(f" G{gen}={best_fit:.4f}", end='')
    
    print(f" ✓ Final={best_fit:.4f}")
    return best_vec


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_with_inserted_words(nodes: List[str], embeddings: np.ndarray, 
                                  inserted_words: List[str],
                                  output_file: str = "embeddings_with_inserted.png",
                                  sample_size: int = 500):
    """Create t-SNE visualization highlighting inserted words."""
    print("\nGenerating t-SNE visualization with inserted words...")
    
    num_original = len(nodes) - len(inserted_words)
    inserted_indices = set(range(num_original, len(nodes)))
    
    # Sample: prioritize inserted words + random original
    if len(nodes) > sample_size:
        sample_indices = list(inserted_indices) + list(np.random.choice(
            num_original, min(sample_size - len(inserted_words), num_original), replace=False))
    else:
        sample_indices = list(range(len(nodes)))
    
    selected_embeddings = embeddings[sample_indices]
    selected_nodes = [nodes[i] for i in sample_indices]
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sample_indices)-1))
    projection = tsne.fit_transform(selected_embeddings)
    
    # Plot
    plt.figure(figsize=(14, 14))
    
    for i in range(len(projection)):
        is_inserted = sample_indices[i] in inserted_indices
        plt.scatter(projection[i, 0], projection[i, 1], 
                   s=200 if is_inserted else 40,
                   alpha=1.0 if is_inserted else 0.6,
                   c='red' if is_inserted else 'steelblue')
        plt.annotate(selected_nodes[i], (projection[i, 0], projection[i, 1]), 
                    fontsize=11 if is_inserted else 9,
                    alpha=1.0 if is_inserted else 0.8,
                    fontweight='bold' if is_inserted else 'normal')
    
    plt.title(f"t-SNE Visualization: {len(sample_indices)} Words "
              f"({sum(1 for i in sample_indices if i in inserted_indices)} Inserted)",
              fontsize=14, fontweight='bold')
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved t-SNE to {output_file}")
    plt.show()


def run_sanity_checks(model: torch.nn.Module, embeddings: np.ndarray, 
                     nodes: List[str], word_to_idx: Dict[str, int]):
    """Run comprehensive sanity checks on loaded model and embeddings."""
    print("\n" + "="*70)
    print("SANITY CHECKS")
    print("="*70)
    
    print(f"\n1. Model Configuration:")
    print(f"   Training mode: {model.training}")
    print(f"   Device: {next(model.parameters()).device}")
    
    print(f"\n2. Embedding Quality:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Mean: {embeddings.mean():.6f}, Std: {embeddings.std():.6f}")
    print(f"   Min: {embeddings.min():.6f}, Max: {embeddings.max():.6f}")
    print(f"   Contains NaN: {np.isnan(embeddings).any()}, Contains Inf: {np.isinf(embeddings).any()}")
    
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n3. Embedding Norms:")
    print(f"   Mean: {norms.mean():.4f}, Std: {norms.std():.4f}")
    print(f"   Range: [{norms.min():.4f}, {norms.max():.4f}]")
    
    print(f"\n4. Vocabulary Test:")
    for test_word in ['man', 'woman', 'dog', 'car', 'blue']:
        if test_word in word_to_idx:
            word_idx = word_to_idx[test_word]
            print(f"   '{test_word:10s}' → idx={word_idx:4d}, norm={np.linalg.norm(embeddings[word_idx]):.4f}")
            similar = find_similar_words(test_word, nodes, embeddings, top_k=5)
            if similar:
                print(f"      Similar: {', '.join([f'{w}({s:.3f})' for w, s in similar])}")
    
    print("\n" + "="*70)
    print("✓ SANITY CHECKS COMPLETE")
    print("="*70)



"""
Don't forget to add tests!!!!
"""

def run_tests():
    """Run all unit tests."""
    pass


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)    
