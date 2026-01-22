import argparse
from collections import Counter
from urllib.parse import urlparse
import requests
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
import sys
import unittest
from unittest.mock import patch, mock_open, MagicMock

# =============================================================================
# SHARED UTILITIES
# =============================================================================

def load_from_source(source):
    """
    Load data from file path or URL.
    
    Args:
        source: File path (str) or URL (str)
    
    Returns:
        bytes: Raw content (text or binary)
    
    Raises:
        requests.HTTPError: If URL request fails
        FileNotFoundError: If file does not exist
    """
    parsed = urlparse(source)
    if parsed.scheme in ("http", "https"):
        response = requests.get(source)
        response.raise_for_status()
        return response.content
    else:
        with open(source, "rb") as f:
            return f.read()


def build_unweighted_graph(nodes, adjacency_counts):
   
    G = nx.Graph() # Replace this line

    G.add_nodes_from(nodes)

    for (a, b), count in adjacency_counts.items():  # Replace [] with adjacency_counts.items()

        if a in nodes and b in nodes and not G.has_edge(a,b):  # Replace False with the correct condition

            # STEP 5: Add an edge between nodes a and b
            # HINT: Use G.add_edge(a, b)
            G.add_edge(a,b)

    return G

    # ********************************************************************
    # ==================== STUDENT CODE SECTION END ======================
    # ********************************************************************


def compute_distance_matrix(nodes, adjacency_counts, distance_mode="inverted"):
    n = len(nodes)  # Replace this line
    node_index = {node:i for i,node in enumerate(nodes)}  # Replace this line

  
    count_matrix = np.zeros((n,n),dtype=float)  # Replace this line

    
    for (a, b), count in adjacency_counts.items():  # Replace [] with adjacency_counts.items()
        if a in node_index and b in node_index:  # Replace False with condition: a in node_index and b in node_index
            # Add the count to the appropriate matrix position
            count_matrix[node_index[a], node_index[b]]=count# Replace this line

    
    count_matrix = (count_matrix + count_matrix.T)/2
    np.fill_diagonal(count_matrix,0)

    # print(f"Removed self-connections (diagonal set to 0)")

    # STEP 6: Convert counts to distances based on distance_mode
    # HINT: Use if-elif-else to handle three cases: 'direct', 'inverted', and invalid

    if distance_mode == "direct":  # Replace "" with "direct"
        # Direct mode: distance equals count (higher count = higher distance)
        # HINT: distance_matrix = count_matrix.copy()
        distance_matrix = count_matrix.copy()  # Replace this line
        # print(f"Using direct distance mode")

    elif distance_mode == "inverted":  # Replace "" with "inverted"
        # Inverted mode: higher count = lower distance
        # HINT: First find max_count using np.max(count_matrix)
        max_count = np.max(count_matrix)  # Replace this line
        # print(f"Max count in matrix: {max_count}")

        if max_count > 0:
            # HINT: distance = (max_count + 1) - count_matrix
            distance_matrix = (max_count + 1) - count_matrix  # Replace this line
            # print(f"Using inverted distance: (max_count + 1) - count")
        else:
            # No edges exist: set all distances to 1 (except diagonal which is 0)
            # HINT: Use np.ones_like(count_matrix) to create matrix of 1s
            distance_matrix = np.ones_like(count_matrix)  # Replace this line
            # print(f"No edges found, setting all distances to 1")

    else:
        # Invalid distance_mode
        # HINT: raise ValueError("distance_mode must be 'direct' or 'inverted'")
        raise ValueError("distance_mode must be 'direct' or 'inverted'")  # Replace this line

    np.fill_diagonal(distance_matrix,0)  # Replace this line

    return distance_matrix, count_matrix

    # ********************************************************************
    # ==================== STUDENT CODE SECTION END ======================
    # ********************************************************************


def visualize_network(G, distance_matrix, nodes, node_colors=None, node_labels=None,
                       figsize=(14, 14), title="Network Graph", jitter=0.2, random_state=42):
    """
    Visualize network in matrix-style grid layout with edge distance labels.
    
    Nodes are arranged in a grid pattern with small random jitter to prevent
    edge overlap. Edge labels show distance values from the distance matrix.
    
    Args:
        G: NetworkX graph
        distance_matrix: numpy array of distances (n x n)
        nodes: ordered list of nodes (must match distance_matrix order)
        node_colors: list of colors for nodes (default: all 'lightblue')
        node_labels: dict of node -> label string (default: str(node)[:10])
        figsize: matplotlib figure size tuple
        title: plot title string
        jitter: max random offset for node positions (default: 0.2)
        random_state: random seed for reproducible jitter (default: 42)
    """
    n = len(nodes)
    node_index = {node: i for i, node in enumerate(nodes)}
    side = int(np.ceil(np.sqrt(n)))
    
    # Grid positions with jitter to avoid edge overlap
    pos = {}
    rng = np.random.RandomState(random_state)
    for i, node in enumerate(nodes):
        row = i // side
        col = i % side
        jitter_x = rng.uniform(-jitter, jitter)
        jitter_y = rng.uniform(-jitter, jitter)
        pos[node] = (col + jitter_x, -row + jitter_y)
    
    plt.figure(figsize=figsize)
    ax = plt.gca()
    ax.set_title(title)
    
    # Default node properties
    if node_colors is None:
        node_colors = ["lightblue"] * n
    if node_labels is None:
        node_labels = {node: str(node)[:10] for node in nodes}
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=900, node_color=node_colors,
                           edgecolors="black", linewidths=1, ax=ax)
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=0.7, ax=ax)
    
    # Edge labels (distances)
    edge_labels = {}
    for u, v in G.edges():
        if u in node_index and v in node_index:
            dist_val = distance_matrix[node_index[u], node_index[v]]
            edge_labels[(u, v)] = f"{dist_val:.1f}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=8, ax=ax, label_pos=0.5)
    
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# =============================================================================
# TEXT PROCESSING
# =============================================================================

def tokenize_text(text, char_like=["'"]):
    
    text = text.lower()
    tokens = []
    current_word = []
    punctuation = {',','.'}

    # print(f"Processing text: '{text[:50]}...'")  # Debug helper (optional)

    # STEP 2: Process each character in the text
    for char in text:
        if char.isalpha() or char in char_like:
            current_word.append(char)
        elif char in punctuation:
            if current_word:
                tokens.append(''.join(current_word))
                current_word = []
            tokens.append(char)
        else:
            if current_word:
                tokens.append(''.join(current_word))
                current_word = []
    if current_word:
        tokens.append(''.join(current_word))
    return tokens

    # ********************************************************************
    # ==================== STUDENT CODE SECTION END ======================
    # ********************************************************************


def replace_rare_tokens(tokens, rare_threshold=0.01, rare_token="<RARE>"):

    # ********************************************************************
    # ==================== STUDENT CODE SECTION START ====================
    # ********************************************************************
    
    punctuation = {',','.'}

    # STEP 1: Extract only words (non-punctuation) for threshold calculation
    # HINT: Use list comprehension to keep only tokens not in punctuation
    # e.g., [t for t in tokens if t not in punctuation]

    # print(f"Original tokens: {tokens}")  # Debug helper (optional)
    # print(f"Rare threshold: {rare_threshold} ({rare_threshold*100}%)")

    word_tokens = [t for t in tokens if t not in punctuation]  # Replace this line

    # print(f"Word tokens only: {word_tokens}")

    # STEP 2: Calculate total word count and handle edge case
    total_words = len(word_tokens)  # Replace this line
    # print(f"Total words: {total_words}")

    if total_words == 0:
        # print("No words found - returning original tokens unchanged")
        return tokens[:], set(), Counter(tokens)

    # STEP 3: Count frequency of each word
    # HINT: Use Counter(word_tokens)
    word_counts = Counter(word_tokens)  # Replace this line
    # print(f"Word counts: {dict(word_counts)}")

    # STEP 4: Identify rare words based on threshold
    # HINT: Use set comprehension:
    # {word for word, count in word_counts.items() if count / total_words < rare_threshold}
    rare_set = {word for word,count in word_counts.items() if count/total_words<rare_threshold}  # Replace this line
    # print(f"Rare words identified: {rare_set}")

    # STEP 5: Create new token list with replacements
    # HINT: Use list comprehension:
    # [rare_token if token in rare_set else token for token in tokens]
    new_tokens = [rare_token if token in rare_set else token for token in tokens]  # Replace this line
    # print(f"Tokens after replacement: {new_tokens}")

    # STEP 6: Count frequencies in the final token list
    final_counts = Counter(new_tokens)  # Replace this line
    # print(f"Final token counts: {dict(final_counts)}")

    # STEP 7: Return all three results as a tuple
    return new_tokens, rare_set, final_counts

    # ********************************************************************
    # ==================== STUDENT CODE SECTION END ======================
    # ********************************************************************


def get_text_adjacencies(tokens):
    
    # ********************************************************************
    # ==================== STUDENT CODE SECTION START ====================
    # ********************************************************************

    # STEP 1: Initialize Counter to store adjacency counts
    adjacency_counts = Counter()

    # print(f"Analyzing adjacencies for {len(tokens)} tokens")  # Debug helper (optional)

    # STEP 2: Loop through all consecutive pairs of tokens
    # HINT: We need to examine pairs at positions (0,1), (1,2), ..., (n-2, n-1)
    # HINT: This means i goes from 0 to len(tokens)-2, so use range(len(tokens) - 1)
    for i in range(len(tokens)-1):  # Replace 0 with the correct range

        # STEP 3: Get the current pair of tokens
        # HINT: Current token is at index i, next token is at index i+1
        current_token = tokens[i]  # Replace this line
        next_token = tokens[i+1]    # Replace this line

        # print(f"  Pair {i}: '{current_token}' -> '{next_token}'")  # Debug helper (optional)

        # STEP 4: Only count pairs where tokens are different (exclude self-loops)
        # HINT: Check if current_token != next_token
        if current_token != next_token:  # Replace False with the correct condition

            # STEP 5: Create the pair tuple and increment its count
            # HINT: Use tuple (current_token, next_token) as the key
            # HINT: Increment count with adjacency_counts[(current_token, next_token)] += 1
            adjacency_counts[(current_token,next_token)] += 1  # Replace this line with your code

    # STEP 6: Return the Counter with all pair frequencies
    return adjacency_counts

    # ********************************************************************
    # ==================== STUDENT CODE SECTION END ======================
    # ********************************************************************





from collections import Counter
from collections import defaultdict

def process_text_network(source, rare_threshold=0.01, rare_token="<RARE>",
                          distance_mode="inverted", verbose=True, nsample_tokens=20):
    
    # Load and tokenize
    content = load_from_source(source)
    text = content.decode('utf-8', errors='ignore')
    
    if verbose:
        print(f"Loaded text: {len(text)} characters")
    
    tokens = tokenize_text(text)
    
    if verbose:
        print(f"Tokenized: {len(tokens)} tokens")
        print(f"Sample tokens: {list(set(tokens))[:nsample_tokens]}")
    
    # Handle rare tokens
    processed_tokens, rare_set, token_counts = replace_rare_tokens(
        tokens, rare_threshold, rare_token)
    
    if verbose:
        print(f"Replaced {len(rare_set)} rare tokens (threshold={rare_threshold})")
        print(f"Final vocabulary: {len(token_counts)} unique tokens")
        print(f"Sample tokens: {list(set(processed_tokens))[:nsample_tokens]}")

    
    # Build adjacencies and graph
    adjacency_counts = get_text_adjacencies(processed_tokens)
    nodes = sorted(token_counts.keys(), key=lambda x: (-token_counts[x], x))
    graph = build_unweighted_graph(nodes, adjacency_counts)
    distance_matrix, count_matrix = compute_distance_matrix(nodes, adjacency_counts, distance_mode)
    
    if verbose:
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
        print(f"Top tokens by frequency:")
        for i, node in enumerate(nodes[:10]):
            print(f"  {i+1:2d}. '{node}' (freq={token_counts[node]})")
    
    return {
        'graph': graph,
        'nodes': nodes,
        'adjacency_counts': adjacency_counts,
        'distance_matrix': distance_matrix,
        'count_matrix': count_matrix,
        'token_counts': token_counts,
        'rare_tokens': rare_set,
        'original_tokens': tokens
    }









# =============================================================================
# IMAGE PROCESSING
# =============================================================================

def preprocess_image(image_data, target_size=(128, 128), quantize_levels=16):
    """
    Load and preprocess image: resize and quantize colors.
    
    Quantization maps each pixel channel value to the nearest of N evenly-spaced
    levels. For uniform channels (min == max), all pixels map to level 0.
    
    Performance note: Uses vectorized numpy operations for efficient quantization.
    
    Args:
        image_data: raw image bytes
        target_size: resize dimensions (width, height)
        quantize_levels: number of quantization levels per channel
    
    Returns:
        tuple of:
            - quantized_image: numpy array of quantized indices (height x width x channels)
            - quantization_info: dict mapping channel_idx -> {'min', 'max', 'levels'}
                where 'levels' is the array of quantization boundary values
    """
    img = Image.open(BytesIO(image_data))
    
    # Handle different image modes
    if img.mode == 'RGBA':
        # Composite with white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        background.paste(img, mask=img.split()[-1])
        img = background
    elif img.mode not in ('RGB', 'L'):
        img = img.convert('RGB')
    
    img = img.resize(target_size, Image.Resampling.LANCZOS)
    img_array = np.array(img)
    
    # Add channel dimension for grayscale
    if len(img_array.shape) == 2:
        img_array = img_array[:, :, np.newaxis]
    
    height, width, channels = img_array.shape
    quantized = np.zeros_like(img_array)
    quantization_info = {}
    
    # Quantize each channel independently using vectorized operations
    for c in range(channels):
        channel_data = img_array[:, :, c]
        min_val, max_val = channel_data.min(), channel_data.max()
        
        if min_val == max_val:
            # Uniform channel: all pixels map to level 0
            quantized[:, :, c] = 0
            quantization_info[c] = {'min': min_val, 'max': max_val, 'levels': [min_val]}
        else:
            # Create evenly-spaced quantization levels
            levels = np.linspace(min_val, max_val, quantize_levels)
            
            # Vectorized quantization: find nearest level index for each pixel
            # Using np.searchsorted for O(N*log(M)) instead of O(N*M)
            quantized[:, :, c] = np.searchsorted(levels, channel_data, side='left')
            # Clamp to valid range [0, quantize_levels-1]
            quantized[:, :, c] = np.clip(quantized[:, :, c], 0, quantize_levels - 1)
            
            quantization_info[c] = {'min': min_val, 'max': max_val, 'levels': levels}
    
    return quantized, quantization_info


def get_spatial_adjacencies(quantized_image):
    """
    Analyze spatial relationships between colors by examining neighboring pixels.

    This function treats an image like a graph where each pixel is a node and edges
    connect neighboring pixels. It counts how often different colors appear next to
    each other, which helps us understand the image's structure and color patterns.

    Think of it like analyzing a neighborhood: which house colors tend to be next to
    each other? Do blue houses neighbor red houses often? This is the same idea but
    for pixels.

    Important: We only count adjacencies between DIFFERENT colors. If two adjacent
    pixels have the same color, we don't record that adjacency (no self-loops).

    Parameters:
    -----------
    quantized_image : numpy.ndarray
        3D array with shape (height, width, channels) containing quantized color values
        Each pixel is represented as quantized integers (e.g., [2, 5, 1] for RGB)

    Returns:
    --------
    tuple of (adjacency_counts, unique_colors, color_frequencies)
        adjacency_counts : Counter
            Maps (color1, color2) tuples to frequency of spatial adjacency
        unique_colors : list of tuples
            All unique colors sorted by frequency (most common first)
        color_frequencies : Counter
            Maps color tuples to total pixel counts in the image

    Neighbor Definition (4-connected):
    ----------------------------------
    For each pixel at position (i, j), we check 4 neighbors:
        - Up:    (i-1, j)  ↑
        - Down:  (i+1, j)  ↓
        - Left:  (i, j-1)  ←
        - Right: (i, j+1)  →

    We ignore diagonal neighbors and boundary pixels that would go outside the image.

    Example:
    --------
    For a 2x2 image:  [[A, B],
                       [C, D]]

    Adjacencies checked (assuming all different colors):
    - Pixel A: neighbors B (right) and C (below)
    - Pixel B: neighbors A (left) and D (below)
    - Pixel C: neighbors A (above) and D (right)
    - Pixel D: neighbors B (above) and C (left)

    Color Format:
    -------------
    Colors are stored as tuples, e.g., (2, 5, 1) for a 3-channel quantized pixel.
    This allows the same analysis to work for grayscale (1 channel), RGB (3 channels),
    or other multi-channel images.

    Algorithm Steps:
    ----------------
    1. Extract image dimensions (height, width, channels)
    2. Initialize Counters for adjacencies and color frequencies
    3. Define 4-connected neighbor direction offsets
    4. Scan every pixel in the image with nested loops
    5. For each pixel, convert its color to a tuple
    6. Count the color's frequency
    7. Check all 4 neighbors using direction offsets
    8. For each neighbor, verify it's within image bounds
    9. If valid, get the neighbor's color as a tuple
    10. If colors differ, record the adjacency
    11. Sort unique colors by frequency (descending)
    12. Return all three results
    """

    # ********************************************************************
    # ==================== STUDENT CODE SECTION START ====================
    # ********************************************************************

    from collections import Counter

    # STEP 1: Extract image dimensions
    # HINT: quantized_image.shape gives (height, width, channels)
    # HINT: Use tuple unpacking: height, width, channels = quantized_image.shape
    height, width, channels = 0, 0, 0  # Replace this line

    # print(f"Image dimensions: {height}h × {width}w × {channels}c")  # Debug helper (optional)

    # STEP 2: Initialize Counters for results
    # HINT: Create Counter() for adjacency_counts
    # HINT: Create Counter() for color_frequencies
    adjacency_counts = None  # Replace this line
    color_frequencies = None  # Replace this line

    # STEP 3: Define the 4-connected neighbor directions
    # HINT: List of (row_offset, col_offset) tuples
    # HINT: Up=(-1,0), Down=(1,0), Left=(0,-1), Right=(0,1)
    directions = []  # Replace this line

    # print(f"Using 4-connected neighbors: {directions}")  # Debug helper (optional)

    # STEP 4: Scan every pixel in the image with nested loops
    # HINT: Outer loop: for i in range(height) for rows
    # HINT: Inner loop: for j in range(width) for columns
    for i in range(0):  # Replace 0 with correct range

        for j in range(0):  # Replace 0 with correct range

            # STEP 5: Get current pixel's color as a tuple
            # HINT: quantized_image[i, j] accesses pixel at row i, column j
            # HINT: Convert to tuple: tuple(quantized_image[i, j])
            current_color = None  # Replace this line

            # print(f"Pixel ({i},{j}): color={current_color}")  # Debug helper (optional)

            # STEP 6: Count this color's frequency
            # HINT: Increment color_frequencies[current_color]
            pass  # Replace this line

            # STEP 7: Check all 4-connected neighbors
            # HINT: Loop through directions: for di, dj in directions:
            for di, dj in []:  # Replace [] with directions

                # Calculate neighbor coordinates
                # HINT: ni = i + di (neighbor row)
                # HINT: nj = j + dj (neighbor column)
                ni = 0  # Replace this line
                nj = 0  # Replace this line

                # STEP 8: Check if neighbor is within image bounds
                # HINT: Check 0 <= ni < height and 0 <= nj < width
                if False:  # Replace False with correct condition

                    # STEP 9: Get neighbor's color as a tuple
                    # HINT: Same as step 5 but use neighbor coordinates (ni, nj)
                    neighbor_color = None  # Replace this line

                    # STEP 10: Check if colors are different and record adjacency
                    # HINT: Only count if current_color != neighbor_color
                    if False:  # Replace False with correct condition

                        # Record this adjacency
                        # HINT: Increment adjacency_counts[(current_color, neighbor_color)]
                        pass  # Replace this line

    # print(f"Found {len(color_frequencies)} unique colors")  # Debug helper (optional)
    # print(f"Found {len(adjacency_counts)} unique adjacencies")  # Debug helper (optional)

    # STEP 11: Create sorted list of unique colors
    # HINT: Sort color_frequencies.keys() by frequency (descending), then color value (ascending)
    # HINT: Use sorted() with key=lambda x: (-color_frequencies[x], x)
    # HINT: The negative sign on frequency makes it descending
    unique_colors = []  # Replace this line

    # print(f"Top 3 colors: {unique_colors[:3]}")  # Debug helper (optional)

    # STEP 12: Return all three results as a tuple
    return adjacency_counts, unique_colors, color_frequencies

    # ********************************************************************
    # ==================== STUDENT CODE SECTION END ======================
    # ********************************************************************


def color_to_rgb(color_tuple, quantization_info):
    """
    Convert quantized color tuple back to RGB for visualization.
    
    Maps each quantized index back to its corresponding level value,
    then normalizes to [0, 1] range for matplotlib. Handles grayscale
    by replicating the single channel to RGB.
    
    Args:
        color_tuple: tuple of quantized channel indices
        quantization_info: dict mapping channel_idx -> {'levels': array}
    
    Returns:
        RGB tuple with values in [0, 1] range
    """
    rgb = []
    for c, quant_val in enumerate(color_tuple):
        if c < len(quantization_info):
            levels = quantization_info[c]['levels']
            if isinstance(levels, (list, np.ndarray)) and len(levels) > int(quant_val):
                rgb_val = levels[int(quant_val)]
            else:
                rgb_val = levels[0] if hasattr(levels, '__getitem__') else levels
            rgb.append(rgb_val / 255.0)  # Normalize to [0,1]
        else:
            rgb.append(0.5)
    
    # Handle different channel counts
    if len(rgb) == 1:
        return (rgb[0], rgb[0], rgb[0])  # Grayscale to RGB
    elif len(rgb) >= 3:
        return tuple(rgb[:3])
    else:
        return tuple((rgb + [0, 0, 0])[:3])


def show_quantized_image(quantized_image, quantization_info, figsize=(8, 8)):
    """
    Display the quantized image.
    
    Args:
        quantized_image: numpy array of quantized indices
        quantization_info: quantization metadata from preprocess_image
        figsize: matplotlib figure size
    """
    height, width, channels = quantized_image.shape
    display_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            color_tuple = tuple(quantized_image[i, j])
            rgb = color_to_rgb(color_tuple, quantization_info)
            display_image[i, j] = [int(c * 255) for c in rgb]
    
    plt.figure(figsize=figsize)
    plt.imshow(display_image)
    plt.title(f"Quantized Image ({height}x{width}, {channels} channels)")
    plt.axis('off')
    plt.show()


def process_image_network(source, target_size=(128, 128), quantize_levels=16,
                           distance_mode="inverted", verbose=True):
    """
    Complete image network processing pipeline.
    
    Pipeline:
    1. Load image from source (file or URL)
    2. Resize and quantize colors
    3. Extract spatial adjacencies between different colors
    4. Build unweighted graph and distance matrix
    
    Args:
        source: file path or URL
        target_size: resize dimensions (width, height)
        quantize_levels: quantization levels per channel
        distance_mode: 'direct' or 'inverted' (see compute_distance_matrix)
        verbose: if True, print processing details and show quantized image
    
    Returns:
        dict with keys:
            - graph: NetworkX Graph
            - nodes: list of color tuples (sorted by frequency)
            - adjacency_counts: Counter of (color1, color2) -> count
            - distance_matrix: numpy array (n x n)
            - count_matrix: symmetrized adjacency counts
            - color_frequencies: Counter of color -> pixel count
            - quantized_image: numpy array of quantized image
            - quantization_info: quantization metadata
    """
    # Load and preprocess image
    content = load_from_source(source)
    quantized_image, quantization_info = preprocess_image(content, target_size, quantize_levels)
    
    if verbose:
        print(f"Image processed: {quantized_image.shape}, {quantize_levels} levels per channel")
    
    if verbose:
        # Show quantized image
        show_quantized_image(quantized_image, quantization_info)
    
    # Get spatial adjacencies
    adjacency_counts, unique_colors, color_frequencies = get_spatial_adjacencies(quantized_image)
    
    if verbose:
        print(f"Found {len(unique_colors)} unique colors")
        print(f"Total spatial adjacencies: {sum(adjacency_counts.values())}")
        print("Top colors by frequency:")
        for i, color in enumerate(unique_colors[:10]):
            print(f"  {i+1:2d}. {color} (freq={color_frequencies[color]})")
    
    # Build graph
    graph = build_unweighted_graph(unique_colors, adjacency_counts)
    distance_matrix, count_matrix = compute_distance_matrix(unique_colors, adjacency_counts, distance_mode)
    
    if verbose:
        print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    return {
        'graph': graph,
        'nodes': unique_colors,
        'adjacency_counts': adjacency_counts,
        'distance_matrix': distance_matrix,
        'count_matrix': count_matrix,
        'color_frequencies': color_frequencies,
        'quantized_image': quantized_image,
        'quantization_info': quantization_info
    }


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestUnifiedNetworks(unittest.TestCase):
    """
    Unit tests for the unified_networks module.
    These are just a few examples. You should implement more. 
    You will be assessed on different tests, not the ones listed here.
    
    These tests help you verify that your implementations are working correctly.
    Each test focuses on a specific function and checks its behavior with simple inputs.
    
    Testing Tips:
    -------------
    - Run tests frequently as you implement each function
    - Read the test descriptions to understand expected behavior
    - Use print statements in your code to debug failures
    - Start with simple test cases before adding complex ones
    
    After completing these starter tests, consider adding your own tests for:
    - Edge cases (empty inputs, single elements, etc.)
    - Different parameter values (thresholds, modes, etc.)
    - Error handling (invalid inputs, out-of-bounds, etc.)
    - Integration tests (full pipeline end-to-end)
    """
    
    def test_build_unweighted_graph(self):
        """Test building an unweighted, undirected graph from adjacency counts."""
        
        # ********************************************************************
        # ==================== STUDENT TEST SECTION START ====================
        # ********************************************************************
        
        # STEP 1: Create test data
        # HINT: nodes should be a list of identifiers: ['A', 'B', 'C']
        # HINT: adjacency_counts should be a Counter with edge frequencies
        # Example: Counter({('A', 'B'): 3, ('B', 'C'): 2, ('A', 'C'): 1})
        nodes = []  # Replace this line
        adjacency_counts = Counter()  # Replace this line
        
        # STEP 2: Call the function being tested
        # HINT: G = build_unweighted_graph(nodes, adjacency_counts)
        G = None  # Replace this line
        
        # STEP 3: Assert that G is a NetworkX Graph (undirected)
        # HINT: Use self.assertIsInstance(G, nx.Graph)
        self.fail("Test not implemented yet - replace this with your assertions")
        
        # STEP 4: Check the graph has the correct number of nodes
        # HINT: Use G.number_of_nodes() to get node count
        # HINT: Use self.assertEqual(G.number_of_nodes(), expected_count)
        
        # STEP 5: Check the graph has the correct number of edges
        # HINT: Use G.number_of_edges() to get edge count
        # HINT: Expected: 3 edges (one for each pair in adjacency_counts)
        
        # STEP 6: Verify specific edges exist
        # HINT: Use G.has_edge(node1, node2) to check if edge exists
        # HINT: Use self.assertTrue(G.has_edge('A', 'B'))
        
        # STEP 7: Verify all expected nodes are present
        # HINT: Use set(G.nodes()) to get all nodes as a set
        # HINT: Use self.assertEqual(set(G.nodes()), set(nodes))
        
        # ********************************************************************
        # ==================== STUDENT TEST SECTION END ======================
        # ********************************************************************
    
    def test_compute_distance_matrix_inverted_mode(self):
        """
        Test distance matrix computation in inverted mode.
        
        In inverted mode, frequently co-occurring nodes should have LOW distances.
        Formula: distance = (max_count + 1) - symmetrized_count
        
        Important: The adjacency counts are symmetrized BEFORE computing distances.
        """
        
        # ********************************************************************
        # ==================== STUDENT TEST SECTION START ====================
        # ********************************************************************
        
        # STEP 1: Create test data
        # HINT: nodes = ['A', 'B']
        # HINT: adjacency_counts = Counter({('A', 'B'): 4, ('B', 'A'): 2})
        nodes = []  # Replace this line
        adjacency_counts = Counter()  # Replace this line
        
        # STEP 2: Call the function with "inverted" mode
        # HINT: distance_matrix, count_matrix = compute_distance_matrix(nodes, adjacency_counts, "inverted")
        distance_matrix = None  # Replace this line
        count_matrix = None  # Replace this line
        
        # STEP 3: Verify the count_matrix is symmetrized correctly
        # HINT: Symmetrization averages directional counts: (4 + 2) / 2 = 3.0
        # HINT: Expected count_matrix = [[0, 3], [3, 0]] (diagonal is 0)
        # HINT: Use np.testing.assert_array_almost_equal(count_matrix, expected, decimal=1)
        self.fail("Test not implemented yet - replace this with your assertions")
        
        # STEP 4: Verify the distance_matrix uses inverted formula
        # HINT: max_count = 3.0, so distance = (3 + 1) - count = 4 - count
        # HINT: Expected distance_matrix = [[0, 1], [1, 0]]
        # HINT: A-B: 4 - 3 = 1 (low distance because high count)
        
        # STEP 5: Verify diagonal is zero in both matrices
        # HINT: Use np.diag(matrix) to get diagonal
        # HINT: Use np.testing.assert_array_equal(np.diag(distance_matrix), [0, 0])
        
        # ********************************************************************
        # ==================== STUDENT TEST SECTION END ======================
        # ********************************************************************
    
    def test_tokenize_text_basic(self):
        """
        Test basic text tokenization.
        
        The tokenizer should:
        - Convert to lowercase
        - Keep letters and apostrophes as part of words
        - Keep commas and periods as separate tokens
        - Ignore numbers, symbols, and other punctuation
        """
        
        # ********************************************************************
        # ==================== STUDENT TEST SECTION START ====================
        # ********************************************************************
        
        # STEP 1: Create test input
        # HINT: Use a simple sentence with various elements
        # Example: "Hello, world! How are you?"
        text = ""  # Replace this line
        
        # STEP 2: Call tokenize_text()
        # HINT: tokens = tokenize_text(text)
        tokens = []  # Replace this line
        
        # STEP 3: Define expected output
        # HINT: Expected tokens should be lowercase words and punctuation
        # Example: ["hello", ",", "world", "how", "are", "you"]
        # Note: Exclamation mark (!) is ignored, period (.) would be kept
        expected = []  # Replace this line
        
        # STEP 4: Verify the output matches expected
        # HINT: Use self.assertEqual(tokens, expected)
        self.fail("Test not implemented yet - replace this with your assertions")
        
        # STEP 5: Test that numbers are ignored
        # HINT: Create a text with numbers: "abc123def"
        # HINT: Expected: ["abcdef"] (numbers stripped out)
        
        # ********************************************************************
        # ==================== STUDENT TEST SECTION END ======================
        # ********************************************************************
    
    def test_get_spatial_adjacencies_basic(self):
        """
        Test basic spatial adjacency detection in images.
        
        Creates a simple 2x2 image with 4 different colors and verifies:
        - All colors are detected
        - Adjacency counts are correct
        - Colors are sorted by frequency
        """
        
        # ********************************************************************
        # ==================== STUDENT TEST SECTION START ====================
        # ********************************************************************
        
        # STEP 1: Create a simple 2x2 test image with 4 different colors
        # HINT: Use np.array with shape (2, 2, 3) for RGB
        # HINT: Each pixel should have a unique color tuple
        # Example: [[[0,0,0], [1,1,1]],
        #           [[2,2,2], [3,3,3]]]
        quantized_image = np.array([[[0,0,0], [0,0,0]],
                                    [[0,0,0], [0,0,0]]])  # Replace with 4 different colors
        
        # STEP 2: Call get_spatial_adjacencies()
        # HINT: Returns (adjacency_counts, unique_colors, color_frequencies)
        adjacency_counts = Counter()  # Replace this line
        unique_colors = []  # Replace this line
        color_frequencies = Counter()  # Replace this line
        
        # STEP 3: Verify we found 4 unique colors
        # HINT: Use self.assertEqual(len(unique_colors), 4)
        self.fail("Test not implemented yet - replace this with your assertions")
        
        # STEP 4: Verify each color appears exactly once (frequency = 1)
        # HINT: Loop through colors and check color_frequencies[color] == 1
        # HINT: Or use self.assertEqual(set(color_frequencies.values()), {1})
        
        # STEP 5: Count total adjacencies
        # HINT: In a 2x2 grid with 4-connected neighbors:
        #   - Top-left (0,0): right, down = 2 neighbors
        #   - Top-right (0,1): left, down = 2 neighbors
        #   - Bottom-left (1,0): up, right = 2 neighbors
        #   - Bottom-right (1,1): up, left = 2 neighbors
        #   - Total = 8 directional adjacencies (each edge counted twice)
        # HINT: Use sum(adjacency_counts.values())
        
        # STEP 6: Verify specific adjacency pairs exist
        # HINT: Top-left (0,0,0) should be adjacent to top-right (1,1,1)
        # HINT: Check adjacency_counts[(0,0,0), (1,1,1)] > 0
        # Note: The actual color tuples depend on your test image setup
        
        # ********************************************************************
        # ==================== STUDENT TEST SECTION END ======================
        # ********************************************************************



def run_tests():
    """Run all unit tests."""
    print("=" * 70)
    print("RUNNING UNIT TESTS")
    print("=" * 70)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestUnifiedNetworks)
    
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
    # Only run tests when script is executed directly
    success = run_tests()
