from typing import Optional, List, Tuple
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import heapq
import unittest
import numpy as np
from io import StringIO
import sys


def _validate_inputs(G, start, end):
    """Common input validation for all search algorithms."""
    if start not in G or end not in G:
        print("Start or goal not in graph. Choose valid nodes.")
        return False
    return True

def _initialize_search_state():
    """Initialize common search state variables."""
    return {
        'visited': set(),
        'expanded': [],
        'tree_edges': [],
        'max_depth': 0,
        'step': 0
    }

def _print_frontier_state(step, frontier_data, algorithm):
    """Print current frontier state for any algorithm."""
    print(f"\n--- Step {step} ---")
    if algorithm == 'BFS' or algorithm == 'DFS':
        nodes, depths = frontier_data
        print(f"Frontier queue: {nodes} (depths: {depths}) | size={len(nodes)}")        
    elif algorithm == 'UCS':
        frontier_info = frontier_data
        print(f"Priority queue (cost-ordered): {frontier_info} | size={len(frontier_info)}")

def _print_goal_reached(path, max_depth, max_size, cost=None):
    """Print goal reached message with statistics."""
    print(f"\nGOAL REACHED!")
    print(f"Path: {' → '.join(path)}")
    if cost is not None:
        print(f"Total cost: ${cost:.1f}")
    print(f"Path length: {len(path)-1} edges")
    print(f"Max depth explored: {max_depth}")
    print(f"Peak memory usage: {max_size} nodes in {'queue' if cost is not None else 'queue/stack'}")

def _print_goal_not_found(end, start, max_depth, max_size):
    """Print goal not found message with statistics."""
    print(f"\n❌ Goal '{end}' not reachable from '{start}'")
    print(f"Max depth explored: {max_depth}")
    print(f"Peak memory usage: {max_size} nodes in queue")


def breadth_first_search(
    G: nx.Graph, 
    start: str, 
    end: str
) -> Tuple[Optional[List[str]], List[str], List[Tuple[str, str]], int]:
    """
    Perform Breadth-First Search (BFS) to find a path from start to end node.
    
    BFS explores nodes level by level using a queue (FIFO structure).
    
    Args:
        G: A NetworkX graph
        start: The starting node label
        end: The goal node label
    
    Returns:
        tuple: (path, expanded_nodes, tree_edges, max_depth)
            - path: List of nodes from start to end, or None if no path exists
            - expanded_nodes: List of nodes explored during search
            - tree_edges: List of (parent, child) tuples forming the search tree
            - max_depth: Maximum depth reached during search
    """
    if not _validate_inputs(G, start, end):
        return None, [], [], 0
    
    # ===== YOUR CODE HERE =====
    # TODO: Implement BFS algorithm
    # Hint: You'll need a queue and a way to track visited nodes
    # Hint: Think about what information you need to store for each node


    # initialize the queue

    # The queue contains a tuple with 4 things
    # (1) the name of the node where we are, 
    # (2) path so far, 
    # (3) node depth, 
    # (4) parent node 
    
    queue = deque([(start, [start], 0 , None)])
    state = _initialize_search_state()
    max_queue_size = 1

    print(f"Starting BFS from '{start}' to '{end}'")

    
    # iterate over the queue to get the current node 

    while queue:
        
        state['step'] +=1
        
        max_queue_size = (max_queue_size, len(queue))

        # show frontier and expand node
        queue_nodes = [n for (n,_,_,_) in queue]
        depths = [d for (_,_,d,_) in queue]

        _print_frontier_state(['step'], (queue_nodes, depths),'BFS')

        # pop the oldest item in the queue 
        node, path, depth, parent = queue.popleft()
        print(f"Expanding: '{node}' at depth {depth}")

        if node in state['visited']:
            print(f"Skipping '{node}',already explored")
            continue
        
        state['visited'].add(node)
        state['expanded'].append(node) # KURIWS GIA NA KRATISOUME TO ORDER  
        state['max_depth'] = max(state['max_depth'],depth)

        if parent is not None:
            state['tree_edges'].append((parent,node))

        if node == end:
            _print_goal_reached(path, state['max_depth'],max_queue_size)
            return path, state['expanded'], state['tree_edges'], state['max_depth']
        
        
    
        # neighbors from the current node
        neighbors = list(G.neighbors(node))
        unvisited_neighbors = [n for n in neighbors if n not in state['visited']]

        print(f" Neighbors: {neighbors}")
        print(f" Adding to queue: {unvisited_neighbors}")
        print(f" (will explora at depth {depth+1})")
        
        # add the neighbors to the queue
        for neighbor in unvisited_neighbors:
            queue.append((neighbor, path + [neighbor], depth+1, node))

        

        # check goal state 
    
                        
    
    # ===== END YOUR CODE =====
    
    _print_goal_not_found(end, start, state['max_depth'], max_queue_size)
    return None, state['expanded'], state['tree_edges'], state['max_depth'] 


def depth_first_search_apeiraxti(
    G: nx.DiGraph, 
    start: str, 
    end: str
) -> Tuple[Optional[List[str]], List[str], List[Tuple[str, str]], int]:
    
    if not _validate_inputs(G, start, end):
        return None, [], [], 0
    
    # ===== YOUR CODE HERE =====
    # TODO: Implement DFS algorithm (iterative, not recursive)
    # Hint: Use a stack (LIFO) instead of a queue
    # Hint: Think about how this differs from BFS

    # initialize the queue

    # The queue contains a tuple with 4 things
    # (1) the name of the node where we are, 
    # (2) path so far, 
    # (3) node depth, 
    # (4) parent node 
    
    queue = deque([(start, [start], 0 , None)])
    state = _initialize_search_state()
    max_queue_size = 1

    print(f"Starting BFS from '{start}' to '{end}'")

    paths_to_end = []

    found_end = False
    
    # iterate over the queue to get the current node 

    while queue:
        
        state['step'] +=1
        
        max_queue_size = (max_queue_size, len(queue))

        # show frontier and expand node
        queue_nodes = [n for (n,_,_,_) in queue]
        depths = [d for (_,_,d,_) in queue]

        _print_frontier_state(['step'], (queue_nodes, depths),'DFS')

        # pop the oldest item in the queue 
        node, path, depth, parent = queue.pop()
        print(f"Expanding: '{node}' at depth {depth}")

        #if node in state['visited']:         
        #    print(f"Skipping '{node}',already in the path we exploring")
        #    continue

        if node == end:
            paths_to_end.append(path)
            found_end = True
            continue
        
        state['expanded'].append(node) # KURIWS GIA NA KRATISOUME TO ORDER  
        state['max_depth'] = max(state['max_depth'],depth)

        if parent is not None:
            state['tree_edges'].append((parent,node))
                
    
        # neighbors from the current node
        neighbors = list(G.neighbors(node))
        unvisited_neighbors = [n for n in neighbors if n not in path] # [n for n in neighbors if n not in state['visited']
        unvisited_neighbors = sorted(unvisited_neighbors, reverse=True)


        print(f" Neighbors: {neighbors}")
        print(f" Adding to queue: {unvisited_neighbors}")
        print(f" (will explora at depth {depth+1})")
        
        # add the neighbors to the queue
        for neighbor in unvisited_neighbors:
            queue.append((neighbor, path + [neighbor], depth+1, node))
    
    
    
    
    # ===== END YOUR CODE =====

    if found_end:
        print(paths_to_end)
        current_longest_path = paths_to_end[0]
        for i in paths_to_end:
            if len(i)>len(current_longest_path):
                current_longest_path = i
        _print_goal_reached(path, state['max_depth'],max_queue_size)
        return current_longest_path, state['expanded'], state['tree_edges'], state['max_depth']
    else:       
        _print_goal_not_found(end, start, 0, 0)
        return None, [], [], 0


def uniform_cost_search(
    G: nx.Graph, 
    start: str, 
    end: str,
    distance_matrix=None
) -> Tuple[Optional[List[str]], List[str], List[Tuple[str, str]], int, Optional[float]]:
    """
    Perform Uniform Cost Search (UCS) to find the lowest-cost path from start to end.
    
    UCS expands nodes in order of path cost, guaranteeing an optimal solution.
    Uses a priority queue where priority is the cumulative path cost.
    
    Args:
        G: A NetworkX graph with 'weight' attributes on edges
        start: The starting node label
        end: The goal node label
        distance_matrix: Optional numpy array of edge costs (if provided, used instead of edge weights)
    
    Returns:
        tuple: (path, expanded_nodes, tree_edges, max_depth, total_cost)
            - path: List of nodes from start to end, or None if no path exists
            - expanded_nodes: List of nodes explored during search
            - tree_edges: List of (parent, child) tuples forming the search tree
            - max_depth: Maximum depth reached during search
            - total_cost: Cost of the path found, or None if no path exists
    """
    if not _validate_inputs(G, start, end):
        return None, [], [], 0, 0
    
    # ===== YOUR CODE HERE =====
    # TODO: Implement UCS algorithm
    # Hint: Use heapq for the priority queue (heappush, heappop)
    # Hint: Priority should be cumulative path cost
    # Hint: Edge costs come from either distance_matrix or G[node][neighbor]["weight"]
    

    
    
    # ===== END YOUR CODE =====
    
    _print_goal_not_found(end, start, 0, 0)
    return None, [], [], 0, None




def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """Recursive hierarchy pos for tree drawing."""
    pos = {root:(xcenter,vert_loc)}
    children = list(G.neighbors(root))
    if children:
        dx = width/len(children)
        nextx = xcenter - width/2 - dx/2
        for child in children:
            nextx += dx
            pos.update(hierarchy_pos(G,child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc-vert_gap, xcenter=nextx))
    return pos

def visualize_search_tree(tree_edges, start, end, path, expanded, 
                          title="Search Exploration Tree", 
                          show_distances=False, distance_matrix=None, nodes=None):
    """Visualize a search exploration tree (DFS, BFS, or UCS). Optionally show edge distances/costs."""
    T = nx.DiGraph()
    T.add_edges_from(tree_edges)

    if not T.nodes():
        print("No tree to visualize.")
        return

    pos = hierarchy_pos(T, start)
    colors = {"start": "green", "goal": "red", "path": "orange", "expanded": "yellow"}
    
    # Determine node colors
    node_colors = [colors["start"] if n == start else colors["goal"] if n == end 
                   else colors["path"] if path and n in path else colors["expanded"] 
                   if n in expanded else "lightblue" for n in T.nodes()]

    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(T, pos, arrows=True, arrowstyle="-|>", arrowsize=18,
                          min_target_margin=12, connectionstyle="arc3,rad=0.06", 
                          width=1.5, edge_color="dimgray")
    nx.draw_networkx_nodes(T, pos, node_color=node_colors, node_size=700, 
                          edgecolors="black", linewidths=1.2)
    nx.draw_networkx_labels(T, pos, font_weight="bold")

    # Show distances if requested
    if show_distances:
        if distance_matrix is not None and nodes is not None:
            node_index = {node: i for i, node in enumerate(nodes)}
            labels = {(u, v): round(distance_matrix[node_index[u], node_index[v]], 2) 
                     for u, v in T.edges() if u in node_index and v in node_index}
        else:
            labels = nx.get_edge_attributes(T, "weight")
        nx.draw_networkx_edge_labels(T, pos, edge_labels=labels, font_color="blue")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=label,
                             markerfacecolor=color, markeredgecolor='black', markersize=12)
                      for label, color in [('Start', colors["start"]), ('Goal', colors["goal"]),
                                          ('On solution path', colors["path"]), ('Expanded', colors["expanded"])]]
    plt.legend(handles=legend_elements, loc="upper right", frameon=True, title="Legend")

    plt.title(title, fontsize=14, fontweight="bold")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# =============================================================================
# UNIT TESTS FOR SEARCH ALGORITHMS
# =============================================================================

class TestSearchAlgorithms(unittest.TestCase):
    """Starter tests: one placeholder per algorithm. Students must implement."""

    def setUp(self):
        """Set up any graphs needed for tests.
        Hints:
        - Create a small unweighted graph for BFS/DFS.
        - Create a small weighted graph for UCS (use 'weight' attributes).
        - Consider adding a disconnected graph for later tests you write.
        """
        # TODO: Initialize graphs like: (just as examples)
        # self.graph = nx.Graph()
        # self.graph.add_edges_from([...])
        # self.weighted_graph = nx.Graph()
        # self.weighted_graph.add_edge('A', 'B', weight=1)
        pass

    def tearDown(self):
        """Optional cleanup."""
        pass

    def suppress_output(self):
        """Use this to silence prints while testing."""
        return SuppressOutput()

    # ==================== BFS Starter ====================

    def test_bfs_reaches_goal(self):
        """BFS: ensure a path from start to goal is found on an unweighted graph.
        Hints:
        - Call breadth_first_search on a small graph where a path exists.
        - Assert that a path is returned and it starts/ends at the expected nodes.
        - Optionally, later add an assertion that the path length equals the shortest distance.
        """
        # TODO: Implement the test body
        pass

    # ==================== DFS Starter ====================

    def test_dfs_finds_a_path(self):
        """DFS: ensure some valid path is found (not necessarily shortest).
        Hints:
        - Call depth_first_search on the same unweighted graph.
        - Assert that a path exists and each consecutive pair forms an edge in the graph.
        - Later, add tests demonstrating deeper-first exploration and unreachable cases.
        """
        # TODO: Implement the test body
        pass

    # ==================== UCS Starter ====================

    def test_ucs_optimal_cost(self):
        """UCS: ensure the returned path cost is optimal on a weighted graph.
        Hints:
        - Build a small weighted graph where the cheapest route is not the shortest by hops.
        - Call uniform_cost_search and assert the total cost equals the known optimum.
        - Later, add tests that use a distance_matrix, invalid inputs, and tie-breaking.
        """
        # TODO: Implement the test body
        pass


class SuppressOutput:
    """Context manager to suppress stdout."""
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = StringIO()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self._original_stdout


def run_search_tests():
    """Run all search algorithm unit tests."""
    print("=" * 60)
    print("RUNNING SEARCH ALGORITHM UNIT TESTS")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSearchAlgorithms)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL SEARCH ALGORITHM TESTS PASSED!")
        print(f"Ran {result.testsRun} tests successfully")
    else:
        print("❌ SOME SEARCH ALGORITHM TESTS FAILED!")
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        
        # Print failure details
        if result.failures:
            print("\nFAILURES:")
            for test, traceback in result.failures:
                print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
        
        if result.errors:
            print("\nERRORS:")
            for test, traceback in result.errors:
                print(f"- {test}: {traceback.split('Exception:')[-1].strip()}")
    
    print("=" * 60)
    return result.wasSuccessful()


if __name__ == "__main__":
    # Only run tests when script is executed directly
    success = run_search_tests()
    sys.exit(0 if success else 1)
