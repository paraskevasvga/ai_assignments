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

## Part 2 — Representation Learning, Semantic Alignment & Planning (Coursework 2)

This assignment focuses on learning and exploiting semantic representations across
language, vision, and planning tasks. The work builds progressively across a series
of laboratory exercises (Labs 6–9), combining representation learning with heuristic
search and symbolic planning.

### Lab 6 — Skip-gram Embeddings (Text Representation)
The provided baseline skip-gram model was extended and improved to learn higher-quality
word embeddings from text. Modifications focused on improving semantic coherence and
stability of the learned representations, including refinements to context construction,
sampling strategies, and weighting schemes. The effectiveness of the improvements was
evaluated through intrinsic analysis and downstream performance, as documented in the
report.

### Lab 7 — Evolutionary Optimization
An evolutionary algorithm was implemented to optimize model parameters, exploring
population-based search as an alternative to gradient-driven optimization. Different
mutation and selection strategies were evaluated, with experiments highlighting the
trade-offs between exploration and convergence when applied to representation learning
tasks.

### Lab 8 — Cross-modal Projection with Contrastive Learning
In this lab, a projection model was trained to align image embeddings with text embeddings
produced by the improved skip-gram model. A contrastive loss objective was used to bring
corresponding image–text pairs closer in the shared embedding space, enabling semantic
alignment across modalities. The trained projection model was evaluated on its ability to
retrieve semantically related representations.

### Lab 9 — Action Grounding and Planning
The code for action grounding and planning was provided without modification. This lab was
used as an integration stage, where the previously learned representations were employed
to define effective heuristic functions for symbolic planning. The final planner leverages
these heuristics to guide search in the problem domain, with design choices and performance
analysis described in detail in the report.

Overall, this coursework emphasizes principled representation learning, empirical
evaluation, and the use of learned embeddings to inform heuristic search and planning
decisions.


## Files and Structure

### Part 1 — Text Network Search
- `cw1.py`: final implementation of search and heuristic methods for Question 6
- `report.pdf`: detailed reasoning, experiments, and analysis
- `lab2.py`: provided lab code used for graph construction utilities
- `CW1.pdf`: coursework specification (Question 6)

### Part 2 — Representation Learning & Planning
- `cw2.py`: final coursework script loading trained models and executing the required tasks
- `models/`: trained models (skip-gram embeddings and projection networks)
- `lab6_Skipgram/`: improved skip-gram implementation and experiments
- `lab7_EvolutionarySearch/`: evolutionary optimization experiments
- `lab8_Contrastive_Projection/`: cross-modal embedding alignment using contrastive loss
- `lab9_Action_Grounding/`: provided planning and grounding code used for heuristic design
- `AI-Coursework 2.pdf`: coursework specification

