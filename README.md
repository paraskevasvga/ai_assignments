# Artificial Intelligence Assignments

This repository contains coursework completed for the Artificial Intelligence module,
focusing on search, heuristic design, and problem-solving on structured state spaces.

---

## Part 1 — Text Network Search on *1984* (Question 6)

In this assignment, a word-level graph was constructed from the text of George Orwell’s
*Nineteen Eighty-Four*, where nodes represent words and edges represent observed transitions
between consecutive words in the text. The task was to perform informed search on this graph
in order to solve a series of optimization and generation problems.

The first part of the work focuses on finding paths through the word network that maximize
either the number of transitions (Longest Path) or the total accumulated cost (Most
Expensive Path). Naïve uninformed search approaches were shown to be insufficient due to the
size of the search space and the presence of cycles, motivating the design of more informed
heuristic strategies.

To address this, an A*-inspired search approach was developed, combining path cost, path
length, and a global “compass” heuristic based on shortest-path distances to sentence-ending
tokens. Different heuristic components were evaluated and tuned, with empirical analysis
guiding the final weighting scheme to balance exploration depth and solution quality.

The second part of the assignment extends this approach to sentence generation (Part b),
where constrained heuristic search is used to generate plausible sentence starts and
continuations within the word network. The solution demonstrates how heuristic guidance can
be adapted beyond classical pathfinding to more open-ended generative search tasks.

Overall, the assignment emphasizes principled heuristic design, experimental evaluation,
and careful reasoning about search behavior in large, cyclic state spaces.

---

## Files
- `cw1.py`: final implementation of the search and heuristic methods
- `report.pdf`: detailed reasoning, experimental results, and discussion
- `lab2.py`: provided lab code used for graph construction and utilities
- `CW1.pdf`: coursework specification (optional)
