"""
Lab 6: Skip-Gram with Negative Sampling (SGNS) for Network Embeddings

Implements Skip-Gram with Negative Sampling to learn embeddings from text networks.
Includes training, evaluation, and visualization tools.

KEY FEATURES:
1. Filters punctuation tokens to prevent hub poisoning
2. Proper negative sampling (5-20 negatives per positive)
3. Weighted sampling by co-occurrence frequency
4. Anti-overfitting: dropout, weight decay, label smoothing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
import networkx as nx
import requests
import zipfile
import json
import os
from typing import List, Dict, Set, Tuple

import unittest
from collections import Counter
from typing import List, Dict, Set, Tuple, Optional



# ============================================================================
# Utilities
# ============================================================================

def download_file(url, out_path):
    """Download a file from URL."""
    print(f"Downloading {url}...")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    print(f"Downloaded to {out_path}")


def prepare_visual_genome_text(zip_url, zip_path="region_descriptions.json.zip", 
                                json_path="region_descriptions.json",
                                output_path="vg_text.txt"):
    """Download, unzip, and process Visual Genome region descriptions."""
    
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping processing.")
        return output_path

    if not os.path.exists(zip_path):
        download_file(zip_url, zip_path)
    
    if not os.path.exists(json_path):
        print(f"Unzipping {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
    
    print(f"Processing {json_path} into {output_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    phrases = [region['phrase'] for img in data for region in img['regions']]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(" . ".join(phrases))
    
    print(f"Processed {len(phrases):,} phrases into {output_path}")
    return output_path


def filter_punctuation_from_network(network_data, punctuation_tokens={'.', ',', '<RARE>', "'"}):
    """
    Remove punctuation tokens from network to prevent hub poisoning.
    
    Punctuation creates massive hubs that bridge unrelated sentences,
    poisoning the graph structure and making embeddings meaningless.
    """
    original_graph = network_data['graph']
    original_nodes = network_data['nodes']
    original_distance_matrix = network_data['distance_matrix']
    
    # Filter nodes
    filtered_nodes = [n for n in original_nodes if n not in punctuation_tokens]
    old_indices = [i for i, n in enumerate(original_nodes) if n not in punctuation_tokens]
    
    # Filter matrices
    filtered_distance_matrix = original_distance_matrix[np.ix_(old_indices, old_indices)]
    
    # Create filtered graph
    filtered_graph = nx.Graph()
    filtered_graph.add_nodes_from(filtered_nodes)
    for u, v in original_graph.edges():
        if u in filtered_nodes and v in filtered_nodes:
            filtered_graph.add_edge(u, v)
    
    print(f"\n🔧 PUNCTUATION FILTER:")
    print(f"  Removed: {punctuation_tokens}")
    print(f"  Nodes: {len(original_nodes):,} → {len(filtered_nodes):,}")
    print(f"  Edges: {original_graph.number_of_edges():,} → {filtered_graph.number_of_edges():,}")
    
    return {
        **network_data,
        'graph': filtered_graph,
        'nodes': filtered_nodes,
        'distance_matrix': filtered_distance_matrix
    }


# ============================================================================
# Dataset
# ============================================================================


"""
SkipGramDataset - Student Starter Code
=======================================

LEARNING OBJECTIVES:
1. Understand how to build a PyTorch Dataset from graph/network data
2. Learn to implement weighted sampling for training pairs
3. Master negative sampling techniques for contrastive learning
4. Handle multi-worker DataLoader scenarios with proper RNG seeding

WHAT YOU'LL IMPLEMENT:
- [ ] _build_contexts(): Extract graph neighborhoods 
- [ ] _generate_weighted_pairs(): Create training pairs with importance weights
- [ ] __getitem__(): Sample negatives on-the-fly with proper exclusions

TESTING YOUR CODE:
Run the unit tests at the bottom to verify your implementation:
    python skipgram_dataset.py
"""

class SkipGramDataset(torch.utils.data.Dataset):
    """
    Skip-Gram dataset for learning node embeddings from a graph structure.
    
    This dataset:
    - Builds (center, context) training pairs from graph neighbors
    - Computes importance weights for weighted sampling
    - Samples negative examples on-the-fly during training
    - Handles multi-worker data loading with independent random streams
    
    Example Usage:
        >>> graph = nx.karate_club_graph()
        >>> nodes = list(graph.nodes())
        >>> dist_matrix = compute_distance_matrix(graph, nodes)  # your function
        >>> dataset = SkipGramDataset(graph, nodes, dist_matrix)
        >>> center, context, negatives = dataset[0]
        >>> print(f"Center: {center}, Context: {context}, Negatives: {negatives[:3]}...")
    """

    def __init__(
        self,
        graph: nx.Graph,
        nodes: List[str],
        distance_matrix: np.ndarray,
        num_negative: int = 15,
        context_size: int = 1,
        cooccur: Optional[Dict[str, Set[str]]] = None,
        stopwords: Optional[Set[str]] = None,
        idf: Optional[Dict[str, float]] = None,
        idf_threshold: float = 4.8,
        low_idf_context_size: int = 2,
        ppmi_matrix: Optional[np.ndarray] = None,
        ppmi_alpha: float = 1.0,
        ctx_idf_beta: float = 1.0,
    ):
        """
        Initialize the Skip-Gram dataset.
        
        Args:
            graph: NetworkX graph where nodes are tokens/words
            nodes: Ordered list of node labels (vocabulary)
            distance_matrix: Precomputed distances between nodes, shape (V, V)
            num_negative: Number of negative samples per positive pair (typically 5-20)
            context_size: Context radius in graph hops (1 = immediate neighbors)
        """
        super().__init__()
        
        # Store basic references
        self.graph = graph
        self.nodes = nodes
        self.node_to_idx = {node: i for i, node in enumerate(nodes)}
        self.vocab_size = len(nodes)
        self.num_negative = num_negative
        self.distance_matrix = distance_matrix

        self.idf = idf or {}
        self.idf_threshold = idf_threshold
        self.low_idf_context_size = low_idf_context_size

        # NEW: store metadata for context filtering
        self.cooccur = cooccur
        self.stopwords = set(stopwords) if stopwords is not None else set()

        self.ppmi = ppmi_matrix  # shape (V,V) or None
        self.ppmi_alpha = ppmi_alpha
        self.ctx_idf_beta = ctx_idf_beta

        # NEW: for stopword-stopword random contexts
        self.stopword_context_k = 8  
        self._context_rng = np.random.default_rng(12345)
        
        # Step 1: Build context sets for each node
        # WHY: We need to know which nodes are "related" to create positive pairs
        self.contexts = self._build_contexts(context_size)
        
        # Step 2: Convert contexts into training pairs and compute weights
        # WHY: PyTorch needs explicit (center, context) pairs, and weighting helps
        #      the model focus on more important relationships
        self.pairs, self.weights = self._generate_weighted_pairs()

        # Keep old number of positive pairs just for "epoch size"
        self._num_pairs = len(self.pairs)
        
        # Build per-center weighted context distributions (Two-stage sampling)
        self._build_center_distributions()
        
        # How many samples per epoch? Keep EXACT same as before (len(self.pairs))
        self.samples_per_epoch = self._num_pairs
        
        # Step 3: Initialize per-worker RNG (lazily, in __getitem__)
        # WHY: Multi-worker DataLoaders need independent random streams
        self._local_rng = None
        
        # Print summary statistics
        self._print_stats()

    # ========================================================================
    # TODO: IMPLEMENT THESE METHODS
    # ========================================================================

    def _build_center_distributions(self):
        """
        Build per-center arrays:
          self.center_to_contexts[center_idx] = np.array([ctx_idx,...])
          self.center_to_probs[center_idx]    = np.array([p1,...]) sums to 1
        Using the SAME weights already computed in _generate_weighted_pairs().
        """
        from collections import defaultdict
    
        tmp_ctx = defaultdict(list)
        tmp_w = defaultdict(list)
    
        for (ci, cj), w in zip(self.pairs, self.weights):
            tmp_ctx[ci].append(cj)
            tmp_w[ci].append(float(w))
    
        self.center_to_contexts = {}
        self.center_to_probs = {}
    
        for ci in range(self.vocab_size):
            ctxs = tmp_ctx.get(ci, [])
            ws = tmp_w.get(ci, [])
    
            if len(ctxs) == 0:
                # Fallback: if a center has no contexts, allow any context except itself
                all_ctx = np.arange(self.vocab_size, dtype=np.int64)
                all_ctx = all_ctx[all_ctx != ci]
                # uniform over all possible contexts
                probs = np.ones(len(all_ctx), dtype=np.float64)
                probs /= probs.sum()
                self.center_to_contexts[ci] = all_ctx.astype(np.int64)
                self.center_to_probs[ci] = probs.astype(np.float64)
                continue
    
            ctx_arr = np.array(ctxs, dtype=np.int64)
            w_arr = np.array(ws, dtype=np.float64)
    
            # Convert weights -> probabilities within this center
            w_arr = np.clip(w_arr, 1e-12, None)
            probs = w_arr / w_arr.sum()
    
            self.center_to_contexts[ci] = ctx_arr
            self.center_to_probs[ci] = probs

    def _build_contexts(self, context_size: int) -> Dict[str, Set[str]]:
        """
        Build a context set for each node (neighbors within context_size hops).
        
        Algorithm:
        1. For each node in self.nodes:
           a. Use nx.single_source_shortest_path_length() to find all reachable nodes
              within context_size hops
           b. Keep only nodes with distance > 0 (exclude self)
           c. Keep only nodes that exist in self.node_to_idx (vocabulary filter)
        2. Return dict: {node_string: set(neighbor_strings)}
        
        HINTS:
        - If a node is not in self.graph, its context should be an empty set()
        - Use cutoff=context_size parameter in nx.single_source_shortest_path_length
        - The returned dict should have an entry for EVERY node in self.nodes
        
        Example:
            If node "cat" has neighbors ["dog", "animal"] within 1 hop:
            contexts["cat"] = {"dog", "animal"}
        
        Args:
            context_size: Maximum number of hops to consider as context
            
        Returns:
            Dictionary mapping each node to its set of context nodes
        """
        contexts = {}
        
        # TODO: Implement context building
        # YOUR CODE HERE
        for node in self.nodes:
            
            # Handle nodes not in graph (give them an empty set within `contexts`)
            if node not in self.graph:
                contexts[node] = set()
                continue

            node_idf = self.idf.get(node, float("inf"))
            radius = context_size if node_idf >= self.idf_threshold else self.low_idf_context_size
            
            # Compute shortest paths within cutoff
            # hint: use nx.single_source_shortest_path_length
            
            lengths = nx.single_source_shortest_path_length(
                self.graph,
                source=node,
                cutoff= radius
            )
            
            # Hop candidates (όπως πριν)
            candidates = {
                other_node
                for other_node, dist in lengths.items()
                if dist > 0 and other_node in self.node_to_idx
            }
    
            # ===============================
            # CASE 1: node ΔΕΝ είναι stopword
            # ===============================
            if node not in self.stopwords:
                # αφαιρούμε stopwords
                candidates = {w for w in candidates if w not in self.stopwords}
    
                # κρατάμε μόνο όσα συνυπάρχουν σε ίδια phrase
                if self.cooccur is not None:
                    allowed = self.cooccur.get(node, set())
                    candidates = {w for w in candidates if w in allowed}
    
                contexts[node] = candidates
                continue
    
            # ===============================
            # CASE 2: node ΕΙΝΑΙ stopword
            # ===============================
            # τυχαίο μικρό subset από stopwords
            pool = set(self.stopwords) - {node}
    
            if len(pool) == 0:
                contexts[node] = set()
                continue
    
            k = min(self.stopword_context_k, len(pool))
            pool_list = list(pool)
            chosen = self._context_rng.choice(len(pool_list), size=k, replace=False)
    
            contexts[node] = {pool_list[i] for i in chosen}
            
        return contexts

    def _generate_weighted_pairs(self) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """
        Generate (center_idx, context_idx) pairs and compute importance weights.
        
        Algorithm:
        1. Iterate through self.contexts to create pairs
        2. For each (center, context) pair, look up the distance from distance_matrix
        3. Convert distances to weights (closer pairs = higher weight)
        4. Apply transformations to prevent overfitting:
           - Sublinear scaling (sqrt) to reduce extreme weights
           - Clipping to prevent dominance by a few pairs
        5. Normalize weights for interpretability
        
        WEIGHT FORMULA:
        Starting from raw distances (where larger = farther):
            raw_weight = (max_distance + 1) - distance  # invert so closer = larger
            weight = sqrt(raw_weight)                    # sublinear scaling
            weight = clip(weight, max=95th_percentile * 3)  # prevent outliers
            weight = normalize(weight)                   # scale to reasonable range
        
        WHY THESE TRANSFORMS:
        - Inversion: Skip-gram should focus on close relationships
        - Sqrt: Prevents a few very-high-frequency pairs from dominating
        - Clipping: Extreme weights can cause training instability
        - Normalization: Makes weights interpretable when printed
        
        HINTS:
        - If there are no pairs, return ([], np.array([], dtype=np.float32))
        - Use self.node_to_idx to convert node strings to indices
        - np.percentile(weights, 95) gives you the 95th percentile
        - Always ensure weights >= 1e-6 to avoid zeros
        
        Returns:
            pairs: List of (center_idx, context_idx) tuples
            weights: numpy array of weights, same length as pairs
        """
        pairs = []
        raw_distances = []
        
        # TODO: Step 1 - Collect pairs and their distances
        # YOUR CODE HERE
        for center_word, context_words in self.contexts.items():
            # Convert center word to index
            
            # For each context word:
            #   - Convert to index
            #   - Append (center_idx, context_idx) to pairs
            #   - Append distance_matrix[center_idx, context_idx] to raw_distances

            center_idx = self.node_to_idx[center_word]
            for ctx_word in context_words:
                context_idx = self.node_to_idx[ctx_word]
        
                # store pair of indices
                pairs.append((center_idx, context_idx))
        
                # store raw distance
                raw_distances.append(self.distance_matrix[center_idx, context_idx])
        
        # Edge case: no pairs found
        if len(pairs) == 0:
            return [], np.array([], dtype=np.float32)
        
        raw_distances = np.array(raw_distances, dtype=np.float32)


        
        # TODO: Step 2 - Convert distances to weights (invert)
        # YOUR CODE HERE
        # weights = ...

        max_dist = raw_distances.max()
        base_w = (max_dist + 1) - raw_distances
        base_w = np.sqrt(base_w)

        # NEW: adjust only if center_idf < threshold
        if self.ppmi is not None and self.idf:
            # φτιάχνουμε ένα multiplier ανά pair
            multipliers = np.ones_like(base_w, dtype=np.float32)
        
            for k, (ci, cj) in enumerate(pairs):
                center_word = self.nodes[ci]
                center_idf = self.idf.get(center_word, float("inf"))
        
                if center_idf < self.idf_threshold:
                    # PPMI term (>=0)
                    pp = float(self.ppmi[ci, cj])
                    # μετατροπή σε (1 + ppmi)^alpha για να μην μηδενίζει τα πάντα
                    pp_term = (1.0 + pp) ** self.ppmi_alpha
        
                    # context idf term
                    ctx_word = self.nodes[cj]
                    ctx_idf = self.idf.get(ctx_word, 0.0)
        
                    # κάνε το idf σχετικό (πχ / threshold) για σταθερότητα
                    idf_term = (1.0 + (ctx_idf / self.idf_threshold)) ** self.ctx_idf_beta
        
                    multipliers[k] = pp_term * idf_term
        
            weights = base_w * multipliers
        else:
            weights = base_w
        
        # TODO: Step 4 - Clip extreme values
        # YOUR CODE HERE
        # clip_threshold = ...
        # weights = ...

        clip_threshold = np.percentile(weights, 95) * 3
        weights = np.clip(weights, a_min=1e-6, a_max=clip_threshold)

        
        # TODO: Step 5 - Normalize weights
        # YOUR CODE HERE
        # weights = ...
        weights = weights / (weights.mean() + 1e-8)
        
        return pairs, weights.astype(np.float32)

    def _available_from_excluded_nodes(self, excluded_nodes: Set[str]) -> np.ndarray:
        excluded_idx = [
            self.node_to_idx[w] for w in excluded_nodes
            if w in self.node_to_idx
        ]
    
        if len(excluded_idx) == 0:
            return np.arange(self.vocab_size, dtype=np.int64)
    
        mask = np.ones(self.vocab_size, dtype=bool)
        mask[excluded_idx] = False
        return np.nonzero(mask)[0].astype(np.int64)

    def __getitem__(self, idx: int) -> Tuple[np.int64, np.int64, np.ndarray]:
        """
        Get a single training example: (center_idx, context_idx, negatives).
        
        Algorithm:
        1. Initialize per-worker RNG if needed (for DataLoader multi-processing)
        2. Retrieve the positive pair at index idx
        3. Build exclusion set: center + all its true contexts
        4. Sample num_negative nodes from vocabulary, excluding the exclusion set
        5. Return (center_idx, context_idx, negatives_array)
        
        WHY EXCLUDE TRUE CONTEXTS:
        If we use a true context node as a "negative" example, we're training
        the model with contradictory signals (it's both positive and negative).
        This confuses learning.
        
        WHY PER-WORKER RNG:
        DataLoader uses multiple worker processes. Each needs an independent
        random stream or they'll all generate identical "random" samples.
        
        HINTS:
        - Use torch.utils.data.get_worker_info() to detect multi-worker mode
        - Seed the RNG with: torch.initial_seed() + worker_id
        - Use self._local_rng.choice() for sampling negatives
        - If available pool < num_negative, use replace=True
        - Handle edge case: if no nodes are available, use the whole vocab
        
        Args:
            idx: Index into self.pairs
            
        Returns:
            center_idx: numpy int64 scalar
            context_idx: numpy int64 scalar  
            negatives: numpy int64 array of shape (num_negative,)
        """
        # TODO: Step 1 - Initialize per-worker RNG (lazy initialization)
        # YOUR CODE HERE
        if self._local_rng is None:
            # Get worker info for multi-processing
            worker_info = torch.utils.data.get_worker_info()
            
            # Compute seed: base_seed + worker_id
            # base_seed = ...
            # seed = ...
            # self._local_rng = ...
            if worker_info is None:
                seed = torch.initial_seed()
            else:
                seed = torch.initial_seed() + worker_info.id
                
            self._local_rng = np.random.default_rng(seed)
        
        # TODO: Step 2 - Get the positive pair
        # YOUR CODE HERE
        # center_idx, context_idx = ...
        # Two-stage positive sampling:
        # 1) pick center uniformly
        center_idx = int(self._local_rng.integers(0, self.vocab_size))
        
        # 2) pick context weighted for this center
        ctx_candidates = self.center_to_contexts[center_idx]
        ctx_probs = self.center_to_probs[center_idx]
        context_idx = int(self._local_rng.choice(ctx_candidates, p=ctx_probs))

         # --------------------------------------------------
        # Step 3: Negative sampling
        # EXCLUDE ONLY center + selected context
        # --------------------------------------------------
        center_node = self.nodes[center_idx]
        context_node = self.nodes[context_idx]

        excluded_nodes = {center_node, context_node}
        available = self._available_from_excluded_nodes(excluded_nodes)

        # Fallback (extremely rare)
        if len(available) == 0:
            available = np.arange(self.vocab_size, dtype=np.int64)

    
        replace = len(available) < self.num_negative
        negatives = self._local_rng.choice(available, size=self.num_negative, replace=replace)
        
        return (
            np.int64(center_idx),
            np.int64(context_idx), 
            negatives.astype(np.int64)
        )

    # ========================================================================
    # PROVIDED HELPER METHODS (no changes needed)
    # ========================================================================

    def get_sample_weights(self) -> np.ndarray:
        """
        Return per-pair weights for WeightedRandomSampler.
        
        Usage:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=dataset.get_sample_weights(),
                num_samples=len(dataset),
                replacement=True
            )
            loader = DataLoader(dataset, sampler=sampler, batch_size=32)
        """
        return self.weights

    def __len__(self) -> int:
        # semantic fairness: dataset length = how many samples per epoch
        return int(self.samples_per_epoch)

    def _print_stats(self):
        """Print dataset statistics for debugging."""
        print("\n📊 SkipGramDataset Statistics:")
        print(f"  Vocabulary size: {self.vocab_size:,}")
        print(f"  Positive pairs: {len(self.pairs):,}")
        print(f"  Negatives per positive: {self.num_negative}")
        print(f"  Total samples per epoch: {len(self) * (1 + self.num_negative):,}")
        
        if self.weights.size > 0:
            print(f"\n  Weight distribution:")
            print(f"    Min: {self.weights.min():.6f}")
            print(f"    Mean: {self.weights.mean():.6f}")
            print(f"    Median: {np.median(self.weights):.6f}")
            print(f"    Max: {self.weights.max():.6f}")
        else:
            print("  ⚠️  No pairs found - check your graph and nodes!")


# ============================================================================
# Model
# ============================================================================

"""
SkipGramModel - Student Starter Code
=====================================

LEARNING OBJECTIVES:
1. Understand dual embedding spaces (center vs context) in Skip-Gram
2. Implement negative sampling loss with label smoothing
3. Learn proper weight initialization for embedding layers
4. Master PyTorch's batched matrix operations

WHAT YOU'LL IMPLEMENT:
- [ ] _init_embeddings(): Initialize embedding weights properly
- [ ] forward(): Compute Skip-Gram Negative Sampling (SGNS) loss
- [ ] get_embeddings(): Extract learned embeddings for downstream use

KEY CONCEPTS:
- Center embeddings: Represent words as "query" vectors
- Context embeddings: Represent words as "key" vectors  
- Why two spaces? Asymmetry helps distinguish "is context of" from "has context"
- Negative sampling: Contrastive learning - push apart unrelated pairs
"""

class SkipGramModel(nn.Module):
    """
    Skip-Gram model with Negative Sampling (SGNS).
    
    Architecture:
        - center_embeddings: Embedding(V, D) - represents words as query vectors
        - context_embeddings: Embedding(V, D) - represents words as key vectors
        - dropout: Regularization applied to center embeddings
    
    Why two embedding matrices?
        In Skip-Gram, words play two roles:
        1. As CENTER: "What contexts does this word appear in?"
        2. As CONTEXT: "What centers is this word a context for?"
        
        These are asymmetric relationships. Using separate embeddings lets the
        model learn different representations for each role, improving quality.
    
    Training objective:
        Maximize: P(context | center) for true pairs
        Minimize: P(negative | center) for random pairs
        
    Example:
        >>> model = SkipGramModel(vocab_size=1000, embedding_dim=128)
        >>> center = torch.tensor([5, 10])      # batch of 2 center words
        >>> context = torch.tensor([8, 15])     # their true contexts
        >>> negatives = torch.randint(0, 1000, (2, 10))  # 10 negatives each
        >>> loss = model(center, context, negatives)
        >>> print(loss.shape)  # torch.Size([2]) - loss per example
    """

    def __init__(self, vocab_size: int, embedding_dim: int, dropout: float = 0.3):
        """
        Initialize Skip-Gram model.
        
        Args:
            vocab_size: Size of vocabulary (number of unique nodes/words)
            embedding_dim: Dimensionality of embedding vectors (typically 50-300)
            dropout: Dropout probability for regularization (prevents overfitting)
        """
        super().__init__()
        
        # Two embedding matrices: one for center words, one for context words
        # WHY: Asymmetric roles in Skip-Gram (see class docstring)
        self.center_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Dropout for regularization (applied only to center embeddings during training)
        # WHY: Prevents model from memorizing training pairs, improves generalization
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings with proper scaling
        self._init_embeddings()

    def _init_embeddings(self):
        """
        Initialize embedding weights using uniform distribution.
        
        Why initialization matters:
            - Too large: Training becomes unstable (exploding gradients)
            - Too small: Learning is slow (vanishing gradients)  
            - Rule of thumb: scale inversely with embedding dimension
        
        Standard practice for Skip-Gram:
            - Use uniform distribution: U(-scale, scale)
            - Scale = 0.5 / sqrt(embedding_dim) OR 0.5 / embedding_dim
            - We use 0.5 / embedding_dim for slightly more conservative init
        
        TODO: Initialize both embedding matrices
        HINTS:
        - Access embedding dimension via: self.center_embeddings.embedding_dim
        - Use nn.init.uniform_(tensor, low, high) to initialize in-place
        - Apply same initialization to both center_embeddings and context_embeddings
        """
        # TODO: Compute initialization scale
        # YOUR CODE HERE
        # scale = ...
        scale = 0.5 / self.center_embeddings.embedding_dim

        
        # TODO: Initialize center_embeddings.weight with uniform distribution
        # YOUR CODE HERE
        # nn.init.uniform_(...)
        nn.init.uniform_(self.center_embeddings.weight, -scale, scale)

        
        # TODO: Initialize context_embeddings.weight with uniform distribution  
        # YOUR CODE HERE
        # nn.init.uniform_(...)
        nn.init.uniform_(self.context_embeddings.weight, -scale, scale)
        

    def forward(
        self, 
        center: torch.Tensor,      # shape: (batch_size,)
        context: torch.Tensor,     # shape: (batch_size,)
        negatives: torch.Tensor,   # shape: (batch_size, num_negatives)
        apply_dropout: bool = True,
        label_smoothing: float = 0.1
    ) -> torch.Tensor:
        """
        Compute Skip-Gram Negative Sampling loss.
        
        Algorithm:
        1. Look up embeddings for center, context, and negative words
        2. Compute positive score: similarity(center, context)
        3. Compute negative scores: similarity(center, each negative)
        4. Apply label smoothing to targets (anti-overfitting)
        5. Compute binary cross-entropy loss using log-sigmoid
        6. Return negative loss (we'll minimize this, which maximizes log-likelihood)
        
        Mathematical formulation:
            Positive loss: -log(σ(center · context))
            Negative loss: -Σ log(σ(-center · negative_i))
            
            With label smoothing (α = 0.1):
            - True positive target: 0.9 instead of 1.0
            - True negative target: 0.9 instead of 1.0
            This prevents overconfident predictions
        
        Args:
            center: Batch of center word indices, shape (B,)
            context: Batch of true context word indices, shape (B,)
            negatives: Batch of negative word indices, shape (B, K)
            apply_dropout: Whether to apply dropout to center embeddings
            label_smoothing: Smoothing factor (0 = no smoothing, 0.1 = mild)
            
        Returns:
            loss: Per-example loss, shape (B,). Caller typically does loss.mean()
        
        HINTS:
        - Use self.center_embeddings(center) to look up embeddings
        - Dot product: torch.sum(a * b, dim=1) for element-wise mult + sum
        - Batch matrix multiply: torch.bmm(A, B) where A is (B,K,D), B is (B,D,1)
        - Log-sigmoid: F.logsigmoid(x) computes log(1/(1+exp(-x))) stably
        - Label smoothing formula: smoothed_target = (1 - α) for positive
        """
        
        # TODO: Step 1 - Look up center embeddings (with optional dropout)
        # YOUR CODE HERE
        # if apply_dropout:
        #     center_emb = ...
        # else:
        #     center_emb = ...
        # Shape: (batch_size, embedding_dim)
        center_emb = self.center_embeddings(center)
        if apply_dropout:
            center_emb = self.dropout(center_emb)
        
        # TODO: Step 2 - Look up context embeddings (no dropout on context)
        # YOUR CODE HERE  
        # context_emb = ...
        # Shape: (batch_size, embedding_dim)
        context_emb = self.context_embeddings(context)
        
        # TODO: Step 3 - Look up negative embeddings
        # YOUR CODE HERE
        # negative_emb = ...
        # Shape: (batch_size, num_negatives, embedding_dim)
        negative_emb = self.context_embeddings(negatives)

        
        # TODO: Step 4 - Compute positive score (dot product)
        # YOUR CODE HERE
        # pos_score = ...
        # Shape: (batch_size,)
        # HINT: Element-wise multiply center_emb * context_emb, then sum over embedding_dim
        pos_score = torch.sum(center_emb * context_emb, dim=1)
        
        # TODO: Step 5 - Compute negative scores (batched dot products)
        # YOUR CODE HERE
        # neg_score = ...
        # Shape: (batch_size, num_negatives)
        # HINT: torch.bmm(negative_emb, center_emb.unsqueeze(2)).squeeze(2)
        #       - unsqueeze(2) makes center_emb shape (B, D, 1) for batch matmul
        #       - squeeze(2) removes the last dimension after matmul

        center_expanded = center_emb.unsqueeze(1)  # (B,1,D)
        neg_score = torch.bmm(center_expanded, negative_emb.transpose(1,2)).squeeze(1)
        
        # TODO: Step 6 - Apply label smoothing and compute losses
        # YOUR CODE HERE
        pos_target = 1.0 - label_smoothing
        neg_target = label_smoothing
        
        # Positive loss with label smoothing
        # Standard loss: -log(sigmoid(pos_score))
        # With smoothing: -(α * log(sigmoid(pos_score)) + (1-α) * log(sigmoid(-pos_score)))
        # Rewrite: -(α * log(σ(s)) + (1-α) * log(1-σ(s)))
        # pos_target = ...  # 1.0 - label_smoothing
        # pos_loss = ...
        # Shape: (batch_size,)
        pos_loss = pos_target * F.logsigmoid(pos_score) \
         + (1 - pos_target) * F.logsigmoid(-pos_score)
        
        # Negative loss with label smoothing  
        # Standard loss: -log(sigmoid(-neg_score))
        # With smoothing: -(α * log(sigmoid(neg_score)) + (1-α) * log(sigmoid(-neg_score)))
        # neg_loss = ...
        # Shape: (batch_size,) after .sum(1)
        neg_loss = (1 - neg_target) * F.logsigmoid(-neg_score) \
         + neg_target * F.logsigmoid(neg_score)

        neg_loss = neg_loss.sum(dim=1)

        
        # TODO: Step 7 - Combine and return negative total loss
        # YOUR CODE HERE
        return -(pos_loss + neg_loss)
        
        

    def get_embeddings(self) -> np.ndarray:
        """
        Extract the learned center embeddings as a numpy array.
        
        Why center embeddings?
            Both center and context embeddings contain learned information, but:
            - Center embeddings are what we optimized as "query" vectors
            - They're used during training with dropout (more robust)
            - Convention: use center embeddings for downstream tasks
            
        Alternative: You could average center + context embeddings, but this
        is less common and may not improve quality.
        
        Returns:
            embeddings: numpy array of shape (vocab_size, embedding_dim)
        
        TODO: Extract center embeddings and convert to numpy
        HINTS:
        - Use .weight to access the embedding matrix
        - Use .detach() to remove from computation graph
        - Use .cpu() to move to CPU (in case model is on GPU)
        - Use .numpy() to convert to numpy array
        """
        # TODO: Return center embeddings as numpy array
        # YOUR CODE HERE
        # return ...
        return (
        self.center_embeddings.weight
        .detach()   # remove from computation graph
        .cpu()      # move to CPU if needed
        .numpy()    # convert to NumPy array
    )



# ============================================================================
# Training
# ============================================================================

def train_embeddings(
    network_data,
    embedding_dim=128,
    batch_size=512,
    epochs=20,
    learning_rate=0.001,
    num_negative=15,
    validation_fraction=0.05,
    context_size=1,
    dropout=0.3,
    weight_decay=1e-4,
    label_smoothing=0.1,
    patience=3,
    device=None,
    save_plot=True,
    cooccur=None,
    stopwords=None,
    idf=None, 
    ppmi_matrix=None,
    idf_threshold=4.8, 
    low_idf_context_size=2,
    ppmi_alpha=1.0, 
    ctx_idf_beta=1.0
):
    """
    Train Skip-Gram embeddings with weighted sampling.
    
    Args:
        network_data: Dict with 'graph', 'nodes', 'distance_matrix'
        embedding_dim: Embedding dimensionality
        batch_size: Training batch size
        epochs: Maximum epochs
        learning_rate: Initial learning rate
        num_negative: Negatives per positive (5-20 recommended)
        validation_fraction: Fraction for validation
        context_size: Graph distance for context (1=neighbors)
        dropout: Dropout rate (default: 0.3)
        weight_decay: L2 regularization (default: 1e-4)
        label_smoothing: Label smoothing factor (default: 0.1)
        patience: Early stopping patience
        device: 'cuda' or 'cpu'
        save_plot: Save training curve
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Filter punctuation
    network_data = filter_punctuation_from_network(network_data)
    nodes = network_data['nodes']
    graph = network_data['graph']
    distance_matrix = network_data['distance_matrix']
    
    # Split edges
    all_edges = list(graph.edges())
    np.random.shuffle(all_edges)
    split_idx = int(len(all_edges) * (1 - validation_fraction))
    
    train_graph = nx.Graph()
    train_graph.add_nodes_from(nodes)
    train_graph.add_edges_from(all_edges[:split_idx])
    
    val_graph = nx.Graph()
    val_graph.add_nodes_from(nodes)
    val_graph.add_edges_from(all_edges[split_idx:])
    
    print(f"\nTrain edges: {len(all_edges[:split_idx]):,}, Val edges: {len(all_edges[split_idx:]):,}")
    
    # Create datasets
    train_dataset = SkipGramDataset(train_graph, nodes, distance_matrix, num_negative, context_size, cooccur=cooccur, stopwords=stopwords, idf=idf, idf_threshold=idf_threshold, low_idf_context_size=low_idf_context_size, ppmi_matrix=ppmi_matrix, ppmi_alpha=ppmi_alpha, ctx_idf_beta=ctx_idf_beta)
    
    val_dataset = SkipGramDataset(val_graph, nodes, distance_matrix, num_negative, context_size, cooccur=cooccur, stopwords=stopwords, idf=idf, idf_threshold=idf_threshold, low_idf_context_size=low_idf_context_size, ppmi_matrix=ppmi_matrix, ppmi_alpha=ppmi_alpha, ctx_idf_beta=ctx_idf_beta)
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,     # ok because dataset ignores idx for sampling
        pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    # Initialize model
    model = SkipGramModel(len(nodes), embedding_dim, dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print(f"\nTraining on {device}")
    print(f"Vocab: {len(nodes)}, Embed dim: {embedding_dim}, Context: {context_size}, Negatives: {num_negative}")
    print(f"Regularization: dropout={dropout}, weight_decay={weight_decay}, label_smoothing={label_smoothing}")
    
    # Training loop
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        total_loss = 0.0
        
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{epochs}", leave=False)
        for i, (centers, contexts, negs) in train_pbar:
            centers, contexts, negs = centers.to(device), contexts.to(device), negs.to(device)
            
            loss = model(centers, contexts, negs, True, label_smoothing).mean()
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            train_pbar.set_postfix({'train_loss': f'{total_loss / (i + 1):.4f}'})
        
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # Validate
        model.eval()
        total_val_loss = 0.0
        
        val_pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc="Validating", leave=False)
        with torch.no_grad():
            for i, (centers, contexts, negs) in val_pbar:
                centers, contexts, negs = centers.to(device), contexts.to(device), negs.to(device)
                
                batch_loss = model(centers, contexts, negs, False, 0.0).mean().item()
                total_val_loss += batch_loss
                val_pbar.set_postfix({'val_loss': f'{total_val_loss / (i + 1):.4f}'})
        
        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch:02d}  train={train_loss:.4f}  val={val_loss:.4f}  lr={optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0  
            best_model_state = model.state_dict()          
            save_data = {
                'model_state_dict': best_model_state,
                'nodes': nodes,
                'vocab_size': len(nodes),
                'embedding_dim': embedding_dim
            }
            torch.save(save_data, "best_model.pth")        
            print(f"  → Best model (val_loss={best_val_loss:.4f}), saved to best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Save plot
    if save_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, 'o-', label='Train', linewidth=2, markersize=6)
        plt.plot(val_losses, 's-', label='Validation', linewidth=2, markersize=6)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_loss.png', dpi=150)
        print("\nSaved loss plot to training_loss.png")
        plt.close()
    
    return {
        'nodes': nodes,
        'embeddings': model.get_embeddings(),
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses
    }


# ============================================================================
# Analysis
# ============================================================================

def find_similar_words(word, nodes, embeddings, top_k=10):
    """Find most similar words using cosine similarity."""
    if word not in nodes:
        return []
    
    idx = nodes.index(word)
    target_vec = embeddings[idx]
    
    similarities = (embeddings @ target_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target_vec) + 1e-10)
    top_indices = np.argsort(-similarities)[1:top_k+1]
    
    return [(nodes[i], float(similarities[i])) for i in top_indices]


def solve_analogy(word_a, word_b, word_c, nodes, embeddings, top_k=5):
    """Solve word analogies: word_a is to word_b as word_c is to ?"""
    node_to_idx = {n: i for i, n in enumerate(nodes)}
    if not all(w in node_to_idx for w in [word_a, word_b, word_c]):
        return []
    
    target_vec = embeddings[node_to_idx[word_b]] - embeddings[node_to_idx[word_a]] + embeddings[node_to_idx[word_c]]
    similarities = (embeddings @ target_vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(target_vec) + 1e-10)
    
    exclude = {node_to_idx[w] for w in [word_a, word_b, word_c]}
    results = [(nodes[i], float(similarities[i])) for i in np.argsort(-similarities) if i not in exclude][:top_k]
    
    return results


def visualize_embeddings(nodes, embeddings, output_file="embeddings_tsne.png", 
                        sample_size=200, annotate=True):
    """Create t-SNE visualization of embeddings."""
    n_samples = min(sample_size, len(nodes))
    selected_embeddings = embeddings[:n_samples]
    selected_nodes = nodes[:n_samples]
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1))
    projection = tsne.fit_transform(selected_embeddings)
    
    plt.figure(figsize=(14, 14))
    plt.scatter(projection[:, 0], projection[:, 1], s=40, alpha=0.6, c='steelblue')
    
    if annotate:
        for i, word in enumerate(selected_nodes):
            plt.annotate(word, (projection[i, 0], projection[i, 1]), fontsize=9, alpha=0.8)
    
    plt.title(f"t-SNE Visualization of Top {n_samples} Word Embeddings")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved t-SNE to {output_file}")
    plt.close()


def analyze_embeddings(nodes, embeddings, 
                       similarity_examples=None,
                       analogy_examples=None,
                       cluster_seeds=None):
    """Comprehensive analysis of learned embeddings."""
    print("\n" + "="*80)
    print("EMBEDDING ANALYSIS")
    print("="*80)
    
    print(f"\nVocabulary: {len(nodes):,}  Embedding dim: {embeddings.shape[1]}")
    
    # Similarity statistics
    sample_emb = embeddings[:min(100, len(embeddings))]
    norms = np.linalg.norm(sample_emb, axis=1, keepdims=True)
    normalized = sample_emb / (norms + 1e-10)
    sim_matrix = normalized @ normalized.T
    sim_values = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    
    print(f"\nSimilarity stats (100 word sample):")
    print(f"  Mean: {sim_values.mean():.4f}  Std: {sim_values.std():.4f}")
    print(f"  Min: {sim_values.min():.4f}  Max: {sim_values.max():.4f}")
    
    # Nearest neighbors
    if similarity_examples:
        print("\n" + "="*80)
        print("NEAREST NEIGHBORS")
        print("="*80)
        for word in similarity_examples:
            similar = find_similar_words(word, nodes, embeddings, top_k=8)
            print(f"\nMost similar to '{word}':")
            if not similar:
                print("  (not in vocabulary)")
            else:
                for token, score in similar:
                    print(f"  {token:15s}  similarity={score:.4f}")
    
    # Analogies
    if analogy_examples:
        print("\n" + "="*80)
        print("WORD ANALOGIES (a:b :: c:?)")
        print("="*80)
        for a, b, c in analogy_examples:
            results = solve_analogy(a, b, c, nodes, embeddings, top_k=3)
            print(f"\n{a}:{b} :: {c}:?")
            if results:
                for token, score in results:
                    print(f"  {token:15s}  score={score:.4f}")
            else:
                print("  (words not in vocabulary)")
    
    # Semantic clusters
    if cluster_seeds:
        print("\n" + "="*80)
        print("SEMANTIC CLUSTERS")
        print("="*80)
        for seed in cluster_seeds:
            if seed in nodes:
                cluster = find_similar_words(seed, nodes, embeddings, top_k=5)
                print(f"\n'{seed}': {', '.join([w for w, _ in cluster])}")
    
    print("\n" + "="*80)


"""
Unit Tests for Skip-Gram with Negative Sampling

Starter skeleton for testing SkipGramDataset and SkipGramModel classes.
Students should implement their own test cases inside the provided class structures.
"""

class TestSkipGramDataset(unittest.TestCase):
    """Tests for SkipGramDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Placeholder setup for dataset initialization
        pass

    # IMPLEMENT YOUR TESTS


class TestSkipGramModel(unittest.TestCase):
    """Tests for SkipGramModel class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Placeholder setup for model initialization
        pass

    # IMPLEMENT YOUR TESTS    


class TestIntegration(unittest.TestCase):
    """Integration tests for dataset and model working together."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        # Placeholder setup for integration testing
        pass

    # IMPLEMENT YOUR TESTS    


def run_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("RUNNING SKIP-GRAM UNIT TESTS")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestSkipGramDataset))
    suite.addTests(loader.loadTestsFromTestCase(TestSkipGramModel))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegration))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
        print(f"Total tests run: {result.testsRun}")
    else:
        print("❌ SOME TESTS FAILED!")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
    print("=" * 70)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
