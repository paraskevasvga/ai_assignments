"""
CW2: Neuro-Symbolic AI System
Student Name: [Your Name]
Student ID: [Your ID]

This module implements a neuro-symbolic AI system that combines:
- Computer Vision (CIFAR-100 object recognition)
- Natural Language Processing (Skip-gram word embeddings)
- Symbolic Planning (PDDL planning)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Set, List, Dict, Tuple, Optional,FrozenSet
from pathlib import Path
import warnings
import os
import torchvision.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import heapq
from dataclasses import dataclass, field


from torchvision import models

from lab9 import (
    validate_user_conditions,
    create_custom_problem,
    PDDLParser,
    ActionGrounder,
    bfs_search,
    print_plan_execution,
    Predicate,
    State,
    Action,
    TOOLS
)


# lab imports


# ============================================================================
# SECTION 1: CIFAR-100 SEMANTIC EXPANSION
# ============================================================================

# DO NOT CHANGE THIS FUNCTION's signature
def build_my_embeddings(checkpoint_path: str = "best_skipgram_523words.pth") -> Tuple[Dict[str, int], np.ndarray]:
    """
    Load and return your trained Skip-gram embeddings.
    
    This function serves as the entry point for loading your final embedding model
    that contains all Visual Genome words AND all 100 CIFAR-100 classes.
    
    Args:
        checkpoint_path: Path to your saved model checkpoint
        
    Returns:
        vocab: Dictionary mapping words to indices {word: index}
        embeddings: Numpy array of shape (vocab_size, embedding_dim)
        
    Example:
        >>> vocab, embeddings = build_my_embeddings()
        >>> print(f"Vocabulary size: {len(vocab)}")
        >>> print(f"Embedding dimension: {embeddings.shape[1]}")
        >>> print(f"'airplane' index: {vocab.get('airplane', 'NOT FOUND')}")
    """
    # TODO: Implement this function
    # 1. Load your checkpoint file
    # 2. Extract the vocabulary dictionary
    # 3. Extract the embedding matrix
    # 4. Ensure vocabulary contains all required words (Visual Genome + CIFAR-100)


    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # 2. Extract vocabulary (list -> dict)
    # The checkpoint stores vocab as a list in embedding order
    vocab_list = checkpoint["vocab"]  # List[str]
    vocab: Dict[str, int] = {word: idx for idx, word in enumerate(vocab_list)}


    # 3. Extract embedding matrix (torch.Tensor -> np.ndarray)
    embeddings = checkpoint["embeddings"]

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    else:
        embeddings = np.asarray(embeddings)

    embeddings = embeddings.astype(np.float32)


    # 4. Sanity checks
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")

    if embeddings.shape[0] != len(vocab):
        raise ValueError(
            f"Embedding rows ({embeddings.shape[0]}) "
            f"do not match vocab size ({len(vocab)})"
        )

    
    return vocab, embeddings


# ============================================================================
# SECTION 2: NEURO-SYMBOLIC AI - MULTI-MODAL PLANNING
# ============================================================================


#Ιmage Encoder
class ImageEncoder(nn.Module):
    """
    MobileNetV3-based image encoder with trainable projection head.
    
    This model consists of two parts:
    1. Frozen pretrained MobileNetV3 backbone (feature extractor)
    2. Trainable projection head (maps features to embedding space)
    
    Args:
        proj_dim (int): Dimension of the output projection embeddings
        device (str): Device to place the model on ("cuda" or "cpu")
    
    Attributes:
        backbone: Frozen MobileNetV3 feature extractor (output: 576-dim)
        projection: Trainable MLP that projects to proj_dim
    """
    
    def __init__(self, proj_dim=64, device="cuda"):
        super().__init__()
        self.device = device
        
        # TODO: Load pretrained MobileNetV3-Small
        # Use models.mobilenet_v3_small with DEFAULT weights
        # Extract all layers except the final classifier using list(base.children())[:-1]
        # Wrap in nn.Sequential, move to device, and set to eval mode
        
        # TODO: Freeze the backbone parameters
        # Loop through self.backbone.parameters() and set requires_grad = False
        
        # TODO: Create trainable projection head
        # Architecture: Linear(576 -> 512) -> BatchNorm1d(512) -> ReLU -> Linear(512 -> proj_dim)
        # Use nn.Sequential to chain the layers
        # Move to device using .to(device)

        # base = models.mobilenet_v3_small(weights=models.MobileNet_V3_S_Small_Weights.DEFAULT)
        try:
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            base = models.mobilenet_v3_small(weights=weights)
        except AttributeError:
            base = models.mobilenet_v3_small(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1]).to(device).eval()
        
        for p in self.backbone.parameters():
            p.requires_grad = False

        self.projection = nn.Sequential(
            
            # THESE ARE THE LAYERS THAT WE ARE GOING TO TRAIN
            # We can CHANGE the layers here. 
            nn.Linear(576, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            nn.Dropout(p=0.2), # DOKIMI

            nn.Linear(512, 256), # DOKIMI NA PROSTHESW KAI ALLO LAYER
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),

            nn.Linear(256, proj_dim)

            ).to(device)
        
    
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Args:
            x (torch.Tensor): Input images of shape (batch_size, 3, 224, 224)
        
        Returns:
            tuple: (backbone_features, projected_embeddings)
                - backbone_features: Raw features from MobileNet (batch_size, 576)
                - projected_embeddings: Projected embeddings (batch_size, proj_dim)
        """
        # TODO: Extract features using the frozen backbone
        # Use torch.no_grad() context to save memory
        # Flatten the output to shape (batch_size, 576) using .flatten(1)
        
        # TODO: Project features through the trainable projection head
        # Pass the flattened features through self.projection
        
        # Return both the backbone features and projected embeddings as a tuple
        with torch.no_grad():
            feats = self.backbone(x).flatten(1)
        out = self.projection(feats)
        return feats, out
    

#LOAD SKIPGRAM
def load_skipgram_523(skipgram_path: str , device: torch.device):
    """
    Loads your saved dict:
      torch.save({
        'embeddings': Tensor [V, D],
        'vocab': list[str],
        'inserted_words': ...,
        'metadata': ...
      }, path)
    Returns:
      vocab_list, word2idx, E_norm (Tensor [V, D] normalized for cosine)
    """
    if not os.path.exists(skipgram_path):
        raise FileNotFoundError(f"Skip-gram checkpoint not found: {skipgram_path}")

    ckpt = torch.load(skipgram_path, map_location="cpu")
    if not isinstance(ckpt, dict) or "embeddings" not in ckpt or "vocab" not in ckpt:
        raise ValueError("Bad skip-gram checkpoint format. Expected dict with 'embeddings' and 'vocab'.")

    E = ckpt["embeddings"]
    vocab_list = ckpt["vocab"]

    if not torch.is_tensor(E):
        raise TypeError("'embeddings' must be a torch.Tensor.")
    if not isinstance(vocab_list, (list, tuple)) or not all(isinstance(w, str) for w in vocab_list):
        raise TypeError("'vocab' must be a list[str].")

    E = E.to(device=device, dtype=torch.float32)
    E = F.normalize(E, p=2, dim=1)  # important for cosine retrieval

    word2idx = {w: i for i, w in enumerate(vocab_list)}
    return vocab_list, word2idx, E

#LOAD PROJECTION, best_cifar100_projection
def load_projection_model(projection_path: str, device: torch.device) -> Tuple[nn.Module, torch.nn.Parameter]:
    """
    Loads the Lab8 projection model checkpoint:
      checkpoint['model_state_dict']
      checkpoint['logit_scale']
      checkpoint (optional) ['proj_dim'] or config
    Returns:
      (vision_model.eval(), logit_scale parameter on device)
    """
    if not os.path.exists(projection_path):
        raise FileNotFoundError(f"Projection checkpoint not found: {projection_path}")

    checkpoint = torch.load(projection_path, map_location=device)

    # proj_dim 
    proj_dim = 96 

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError("Bad projection checkpoint format. Expected dict with 'model_state_dict'.")

    vision_model = ImageEncoder(proj_dim=proj_dim, device=str(device)).to(device)
    vision_model.load_state_dict(checkpoint["model_state_dict"])
    vision_model.eval()

    # Restore learned logit_scale (if present)
    if "logit_scale" in checkpoint:
        logit_scale = torch.nn.Parameter(torch.tensor(checkpoint["logit_scale"], device=device))
    else:
        # fallback: neutral scale
        logit_scale = torch.nn.Parameter(torch.tensor(1.0, device=device))

    return vision_model, logit_scale


CIFAR100_WORDS_RAW = """
apple aquarium-fish baby bear beaver bed bee beetle bicycle bottle
bowl boy bridge bus butterfly camel can castle caterpillar cattle
chair chimpanzee clock cloud cockroach couch crab crocodile cup
dinosaur dolphin elephant flatfish forest fox girl hamster house
kangaroo keyboard lamp lawn-mower leopard lion lizard lobster man
maple motorcycle mountain mouse mushroom oak orange orchid otter
palm pear pickup-truck pine plain plate poppy porcupine possum
rabbit raccoon ray road rocket rose sea seal shark shrew skunk
skyscraper snail snake spider squirrel streetcar sunflower
sweet-pepper table tank telephone television tiger tractor train
trout tulip turtle wardrobe whale willow wolf woman worm
""".strip().split()



def canonicalize_word(w: str) -> str:
    return w.strip().lower()

CIFAR_ALIAS = {
    "maple": "maple_tree",
    "oak": "oak_tree",
    "palm": "palm_tree",
    "pine": "pine_tree",
    "willow": "willow_tree",
}

REVERSE_CIFAR_ALIAS = {v: k for k, v in CIFAR_ALIAS.items()}


def candidate_forms(w: str):
    # ό,τι είχες ήδη...
    w = w.strip().lower()
    cands = {w, w.replace("-", "_"), w.replace("_", "-"), w.replace(" ", "_"),
             w.replace("-", " "), w.replace("_", " ")}

    # extra: tree aliases if not present
    cands.add(w + "_tree")
    cands.add(w + "-tree")
    cands.add(w + " tree")
    return list(cands)

def build_cifar_retrieval_bank(vocab_list, word2idx, E_words):
    cifar_vocab = []
    cifar_indices = []
    missing = []

    for w in CIFAR100_WORDS_RAW:
        # 1) alias override
        if w in CIFAR_ALIAS:
            w_query = CIFAR_ALIAS[w]
            if w_query in word2idx:
                idx = word2idx[w_query]
                cifar_vocab.append(vocab_list[idx])
                cifar_indices.append(idx)
                continue
            # αν για κάποιο λόγο λείπει, πέφτουμε στο generic search

        # 2) generic search through candidate forms
        found_idx = None
        for cand in candidate_forms(w):
            if cand in word2idx:
                found_idx = word2idx[cand]
                break

        if found_idx is None:
            missing.append(w)
            continue

        cifar_vocab.append(vocab_list[found_idx])
        cifar_indices.append(found_idx)

    if len(cifar_vocab) == 0:
        raise RuntimeError("No CIFAR words found in vocab.")

    if missing:
        print(f"[WARN] Missing {len(missing)} CIFAR words (e.g. {missing[:8]}). Continuing.")

    E_cifar = E_words[cifar_indices]
    return cifar_vocab, E_cifar



# ------------------------------
# Image preprocessing for MobileNetV3
# ------------------------------
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]

_normalize = T.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)


def is_cifar_token(token: str) -> bool:
    # token είναι από vocab_list, π.χ. "pickup_truck" ή "oak_tree"
    # θέλουμε να δούμε αν αντιστοιχεί σε κάποια CIFAR κλάση (με aliases/variants)
    tok = token.lower()
    for w in CIFAR100_WORDS_RAW:
        if tok in candidate_forms(w):
            return True
    return False

def identify_object(
    input_data: Union[torch.Tensor, str],
    vision_model: nn.Module,
    E_cifar: torch.Tensor,
    cifar_vocab: List[str],
    word2idx: Dict[str, int],
    vocab_list: List[str],
    E_words,                 
    topk: int = 10,          
    require_text_to_be_cifar: bool = True,
) -> Optional[str]:
    """
    Returns:
      - object name string (canonical form present in vocab)
      - None if cannot identify / invalid
    """

    # ---- TEXT INPUT ----
    if isinstance(input_data, str):
        raw = canonicalize_word(input_data)
        # try to match to vocabulary using variants
        matched = None
        for cand in candidate_forms(raw):
            if cand in word2idx:
                matched = cand
                break
        if matched is None:
            return None

        if require_text_to_be_cifar:
            # ensure it's a CIFAR object name (in any variant form)
            ok = False
            for w in CIFAR100_WORDS_RAW:
                if matched in candidate_forms(w):
                    ok = True
                    break
            if not ok:
                return None

        return vocab_list[word2idx[matched]]

    # ---- IMAGE INPUT ----
    if torch.is_tensor(input_data):
        x = input_data

        # Ensure [B,3,H,W]
        if x.dim() == 3:
            x = x.unsqueeze(0)
        if x.dim() != 4 or x.size(1) != 3:
            return None

        device = next(vision_model.parameters()).device
        x = x.to(device=device).float()

        # If in [0,255], scale to [0,1]
        if x.max() > 1.5:
            x = x / 255.0

        # Resize+Normalize (MobileNet expects ImageNet norm)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = _normalize(x)


        with torch.no_grad():
            _, z = vision_model(x)  # [B,D]

        z = F.normalize(z, p=2, dim=1)      # normalize image embedding

        # 1) similarities against ALL 523 words
        sims_all = (z @ E_words.T).squeeze(0)  # [523]

        # 2) top-K shortlist
        top_vals, top_idx = torch.topk(sims_all, k=min(topk, sims_all.numel()))

        # 3) filter shortlist to CIFAR tokens
        best_token = None
        best_val = None
        for val, idx in zip(top_vals.tolist(), top_idx.tolist()):
            tok = vocab_list[idx]
            if is_cifar_token(tok):
                best_token = tok
                best_val = val
                break
        
        
        # 4) if none found => out-of-domain
        if best_token is None:
            return None

        return best_token

    return None

PROBLEM_FILE = "problem.pddl"  # ή το path του δικού σου template



# ==================== ACTION GROUNDING ====================

# ==================== CONSTANTS ====================

TOOLS = {'knife', 'dslr'}
LOCATIONS = {'lab', 'outdoors'}


class ActionGrounder:
    def __init__(self, schemas: Dict[str, Action], objects: Dict[str, Set[str]]):
        self.schemas = schemas
        self.all_items = sorted(objects.get('item', set()))
        self.cifar_objects = sorted(set(self.all_items) - TOOLS)
        self.locations = sorted(LOCATIONS)
    
    def ground_all(self) -> List[Action]:
        """Ground all action schemas with concrete objects."""
        grounded = []
        
        for name, schema in self.schemas.items():
            if name == 'walk-between-rooms':
                grounded.extend(self._ground_walk(schema))
            elif name in ['pick-up', 'put-down']:
                grounded.extend(self._ground_item_location(schema))
            elif name in ['stack', 'unstack']:
                grounded.extend(self._ground_stack(schema))
            elif name in ['slice-object', 'clean-object', 'take-photo']:
                grounded.extend(self._ground_object_action(schema))
        
        return grounded
    
    def _ground_walk(self, schema: Action) -> List[Action]:
        return [schema.instantiate({schema.parameters[0]: l1, schema.parameters[1]: l2})
                for l1 in self.locations for l2 in self.locations if l1 != l2]
    
    def _ground_item_location(self, schema: Action) -> List[Action]:
        return [schema.instantiate({schema.parameters[0]: item, schema.parameters[1]: loc})
                for item in self.all_items for loc in self.locations]
    
    def _ground_stack(self, schema: Action) -> List[Action]:
        return [schema.instantiate({schema.parameters[0]: top, schema.parameters[1]: bottom, schema.parameters[2]: loc})
                for top in self.all_items for bottom in self.all_items 
                if top != bottom for loc in self.locations]
    
    def _ground_object_action(self, schema: Action) -> List[Action]:
        return [schema.instantiate({schema.parameters[0]: obj, schema.parameters[1]: loc})
                for obj in self.cifar_objects for loc in self.locations]

    

# ΙMPLEMENT LOGIC AND SCORES FOR IMPROVING BFS SEARCH
@dataclass(order=True)
class SearchNode:
    # tuple ordering: (g, -tie_score, counter)
    f_score: Tuple[int, int, int]
    state: 'State' = field(compare=False)
    action: Optional['Action'] = field(compare=False)
    parent: Optional['SearchNode'] = field(compare=False)
    g_score: int = field(compare=False)

    def get_plan(self) -> List['Action']:
        plan, node = [], self
        while node.parent:
            plan.append(node.action)
            node = node.parent
        return list(reversed(plan))
    

def analyze_goal_landmarks(goal: FrozenSet['Predicate']):
    clear_targets = set()
    required_tools = set()

    # mapping goal -> tool
    for g in goal:
        if g.name == "documented" and len(g.args) == 1:
            x = g.args[0]
            clear_targets.add(x)
            required_tools.add("dslr")
        elif g.name == "cut-into-pieces" and len(g.args) == 1:
            x = g.args[0]
            clear_targets.add(x)
            required_tools.add("knife")
        elif g.name == "on-top" and len(g.args) == 2:
            top, bottom = g.args
            clear_targets.add(bottom)

    return clear_targets, required_tools


def holding_item(state: 'State') -> Optional[str]:
    for p in state.predicates:
        if p.name == "holding" and len(p.args) == 1:
            return p.args[0]
    return None


def is_clear(state: 'State', x: str) -> bool:
    for p in state.predicates:
        if p.name == "clear" and len(p.args) == 1 and p.args[0] == x:
            return True
    return False

def agent_location(state: 'State') -> Optional[str]:
    for p in state.predicates:
        if p.name == "agent-at" and len(p.args) == 1:
            return p.args[0]
    return None


def is_hand_empty(state: 'State') -> bool:
    return any(p.name == "hand-empty" for p in state.predicates)


def stack_height_above(state: 'State', x: str) -> int:
    # build bottom->top map from (on-top top bottom)
    bottom_to_top = {}
    for p in state.predicates:
        if p.name == "on-top" and len(p.args) == 2:
            top, bottom = p.args
            bottom_to_top[bottom] = top

    h, cur = 0, x
    seen = set()
    while cur in bottom_to_top and cur not in seen:
        seen.add(cur)
        cur = bottom_to_top[cur]
        h += 1
    return h


def tie_score(state: 'State', goal: FrozenSet['Predicate']) -> int:
    # 1) core: satisfied goals
    sat = sum(1 for g in goal if g in state.predicates)
    unsat = len(goal) - sat
    score = 1000 * sat - 50 * unsat

    clear_targets, required_tools = analyze_goal_landmarks(goal)
    held = holding_item(state)

    # 2) conditional clear-first
    for x in clear_targets:
        if not is_clear(state, x):
            # clear-first regime
            if is_hand_empty(state):
                score += 120

            # penalize being stuck holding a tool while clearing
            if held in {"knife", "dslr"}:
                score -= 40

            # stack progress (smaller height better)
            score -= 30 * stack_height_above(state, x)
        else:
            # once clear, tool becomes useful
            if required_tools and held in required_tools:
                score += 120
            # If target is already clear but we still need a tool,
            # prefer being ready to pick it up: hand-empty and not holding unrelated items.
            
            if required_tools and held is not None and held not in required_tools:
                score -= 30   # holding "bowl"/"plate" is bad when we need DSLR/knife

            if required_tools and is_hand_empty(state):
                score += 20   # hand-empty is good (ready to pick up tool)
            
            loc = agent_location(state)
            if loc is not None and required_tools and held is not None and held not in required_tools and loc == "lab":
                score += 10   # slight preference to drop clutter before leaving lab
    
    # 3) gentle push if tool needed but not held (only small; not absolute)
    if required_tools and (held not in required_tools):
        score -= 20

    return score

def h_cost(state: 'State', goal: FrozenSet['Predicate']) -> int:
    """
    Convert tie_score (bigger is better) into a non-negative cost h (smaller is better).
    """
    C = 50000  # constant shift; can be tuned
    return max(0, C - tie_score(state, goal))


def bfs_search(initial: 'State',
               goal: FrozenSet['Predicate'],
               actions: List['Action'],
               max_iter: int = 50000,
               verbose: bool = True) -> Optional[List['Action']]:

    # 1) already there
    if initial.satisfies(goal):
        if verbose:
            print("Goal Already Satisfied!")
        return []

    # 2) Setup frontier as PRIORITY QUEUE
    counter = 0
    start_h = h_cost(initial, goal)
    start_f = 0 + start_h  # g=0
    start_node = SearchNode(
        f_score=(start_f, 0, counter),
        state=initial,
        action=None,
        parent=None,
        g_score=0
    )

    frontier = []
    heapq.heappush(frontier, start_node)

    # depth-optimal visited: store best g found for a state
    best_g = {initial: 0}

    iters = 0
    if verbose:
        print("\nBest-BFS with Heuristic Starting")

    while frontier and iters < max_iter:
        iters += 1

        if verbose and iters % 1000 == 0:
            print(f"    Iter {iters}, frontier: {len(frontier)}, visited: {len(best_g)}")

        # 1) pop best node (min by (f, g, counter))
        curr = heapq.heappop(frontier)
        g = curr.g_score

        # stale entry check (classic when using heap)
        if best_g.get(curr.state, 10**9) != g:
            continue

        # 2) goal test
        if curr.state.satisfies(goal):
            if verbose:
                print(f"✓ Goal found! Iterations: {iters}, Plan length: {curr.g_score}")
            return curr.get_plan()

        # 3) expand successors
        for act in actions:
            if curr.state.is_applicable(act):
                next_state = curr.state.apply_action(act)
                ng = g + 1

                # only keep if we found a strictly better depth
                if ng < best_g.get(next_state, 10**9):
                    best_g[next_state] = ng

                    counter += 1

                    h = h_cost(next_state, goal)
                    f = ng + h  # A*: f = g + h

                    child = SearchNode(
                        f_score=(f, ng, counter),
                        state=next_state,
                        action=act,
                        parent=curr,
                        g_score=ng
                    )
                    heapq.heappush(frontier, child)

    if verbose:
        print(f"✗ No solution. Iterations: {iters}")
    return None

def canonicalize_pred_str(s: str) -> str:
    # replace any alias constants back to CIFAR names (maple_tree -> maple)
    for alias, canon in REVERSE_CIFAR_ALIAS.items():
        s = s.replace(alias, canon)
    # normalize CIFAR underscores to PDDL hyphens inside predicates
    s = s.replace("_", "-")
    return s


#VALIDATE THE PLAN FOUND
def validate_plan(init_state: 'State', goal: FrozenSet['Predicate'], plan_actions: List['Action']) -> bool:
    """Replay the plan to ensure every action is applicable and the goal is achieved."""
    cur = init_state
    for i, act in enumerate(plan_actions, start=1):
        if not cur.is_applicable(act):
            print(f"[PLAN VALIDATION ERROR] Step {i}: action not applicable: {act}")
            return False
        cur = cur.apply_action(act)

    if not cur.satisfies(goal):
        print("[PLAN VALIDATION ERROR] Final state does not satisfy goal.")
        return False
        
    return True

# DO NOT CHANGE THIS FUNCTION's signature
def plan_generator(input_data: Union[torch.Tensor, str],    # ASSUME default CIFAR-100 image dimensions
                  initial_state: List[str],                 # Consistent with Lab9 syntax
                  goal_state: List[str],                    # Consistent with Lab9 syntax
                  domain_file: str = "domain.pddl",
                  skipgram_path: str = "best_skipgram_523words.pth",
                  projection_path: str = "best_cifar100_projection.pth") -> Optional[List[str]]:
    """
    !!!WARNING!!!: Treat this as pseudocode. You may need to modify the logic. 
    
    Main entry point for the neuro-symbolic planning system.
    
    This function implements the complete pipeline from perception to planning.
    
    Args:
        input_data: Either an image tensor OR object name string
        initial_state: List of predicates describing initial state                      
        goal_state: List of predicates describing goal state                   
        domain_file: Path to the PDDL domain file
        skipgram_path: Path to Skip-gram embeddings checkpoint
        projection_path: Path to CIFAR-100 projection model checkpoint
        
    Returns:
        A list of action strings representing the plan, 
            OR None if:
                - The object cannot be identified
                - No valid plan exists
                - ...
        
    Example:
        >>> image = # CIFAR-100 image
        >>> initial = ["on table"]
        >>> goal = ["in basket"]
        >>> plan = plan_generator(image, initial, goal, "domain.pddl")        
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ---- Step 0: Load models ----
    try:
        vocab_list, word2idx, E_words = load_skipgram_523(skipgram_path, device)
        vision_model, logit_scale = load_projection_model(
            projection_path, device
        )
        cifar_vocab, E_cifar = build_cifar_retrieval_bank(vocab_list, word2idx, E_words)
    except Exception:
        print("[ERROR] Failed to load models")
        return None

    # ---- Step 1: Identify object ----
    obj_name = identify_object(
        input_data=input_data,
        vision_model=vision_model,
        E_cifar=E_cifar,
        cifar_vocab=cifar_vocab,
        word2idx=word2idx,
        vocab_list= vocab_list,
        E_words=E_words,
        topk=10,
        require_text_to_be_cifar=True,  # set False if instructor clarifies otherwise
    )
    if obj_name is None:
        print("[ERROR] Object could not be identified")
        return None
    
    # convert vocab token (e.g., maple_tree) back to CIFAR canonical name (maple)
    obj_name = REVERSE_CIFAR_ALIAS.get(obj_name, obj_name)
    obj_name = obj_name.replace("_", "-")

                    
    # --- Step 2: Parse PDDL domain ---
    actions_dict = PDDLParser.parse_domain(domain_file)
    
    
    # --- Step 3: Build a problem instance (temp .pddl) from user predicates ---
    # IMPORTANT: initial_state / goal_state είναι List[str] σε Lab9 syntax.
    initial_state = [canonicalize_pred_str(s) for s in initial_state]
    goal_state    = [canonicalize_pred_str(s) for s in goal_state]

    # ✅ Ground ungrounded predicates for IMAGE INPUT (or even for both)
    initial_state = [s.replace("?x", obj_name) for s in initial_state]
    goal_state    = [s.replace("?x", obj_name) for s in goal_state]

    initial_set = set(initial_state)
    goal_set    = set(goal_state)

    # (προαιρετικό αλλά χρήσιμο) fail-fast αν έχουν typos
    try:
        validate_user_conditions(initial_set)
        validate_user_conditions(goal_set)
    except Exception as e:
        print(f"[ERROR] Invalid predicates in input: {e}")
        return None
    
    # Φτιάχνουμε temporary problem file βασισμένο σε template PROBLEM_FILE
    # και κάνουμε override init/goal για το συγκεκριμένο obj_name
    temp_prob = create_custom_problem(PROBLEM_FILE, initial_set, goal_set, obj_name)

    try:
        # Parse the *new* problem
        objs, init, goal = PDDLParser.parse_problem(temp_prob)

    finally:
        if os.path.exists(temp_prob):
            os.unlink(temp_prob)
    
    # Ground actions 
    grounder = ActionGrounder(actions_dict, objs)
    grounded_actions = grounder.ground_all()

    plan = bfs_search(init, goal, grounded_actions, max_iter=80000, verbose=True)
    if plan is None:
        print("[ERROR] No valid plan found")
        return None

    #Validate that the plan is applicable
    if not validate_plan(init, goal, plan):
        print("[ERROR] Plan validation failed")
        return None
    
    return [str(a) for a in plan]
        
