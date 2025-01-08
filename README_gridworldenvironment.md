
# Policy Gradient Algorithms for Grid World Environment

This project involves implementing policy gradient (PG) algorithms to solve a grid world problem as part of the **AI with Reinforcement Learning** course at CU Denver.

## Overview

The goal is to implement two popular reinforcement learning algorithms:
1. **REINFORCE**
2. **Actor-Critic**

These algorithms are tested on a grid world environment, optimizing the policy to maximize rewards while navigating from the start to the goal state, avoiding pits.

---

## Grid World Environment

The grid world is a 4x4 grid with the following properties:

### Layout

```
---------------------
|  0 |  1 |  2 |  3 |
---------------------
|  4 |  5 |  6 |  7 |
---------------------
|  8 |  9 | 10 | 11 |
---------------------
| 12 | 13 | 14 | 15 |
---------------------
```

### Rewards
- **+100**: Reaching the goal state (cell 15).
- **-70**: Falling into pits (cells 5 and 9).
- **-1**: Each step taken.

### Components
- **Action Space (𝐴)**: 4 actions `[0: up, 1: down, 2: left, 3: right]`.
- **State Space (𝑆)**: 17 states (16 grid cells + 1 absorbing state upon reaching the goal).
- **Transition Function (𝑇)**: Defines state transitions based on actions.
- **Reward Function (𝑅)**: Maps actions and states to rewards.
- **Discount Factor (𝛾)**: Scalar value in [0, 1) for future reward weighting.

---

## Tasks

### 1. Implementation
- Implement **REINFORCE** and **Actor-Critic** algorithms in the provided `PG.py` file.
- Complete the methods `reinforce()` and `actorCritic()` within the `ReinforcementLearning` class.
- Input Parameters:
  - `theta`: Initial policy parameters.
  - Other required hyperparameters for the algorithms.

### 2. Evaluation
- Produce a performance graph with:
  - **X-axis**: Number of episodes (large enough for algorithm convergence).
  - **Y-axis**: Cumulative rewards per episode, averaged over 10 runs.

---

## Getting Started

### Prerequisites
- Python 3.x
- Libraries: `numpy`, `matplotlib`

### Files
- **`PG.py`**: Contains starter code for implementing the algorithms.
- **`MDP.py`**: Defines the grid world environment.

---

## Instructions

1. Clone this repository.
2. Open `PG.py` and implement the `reinforce()` and `actorCritic()` methods.
3. Run the script to train the algorithms on the grid world environment.
4. Plot the results using the cumulative rewards per episode.

---

## Results

The project requires evaluating the optimized policy by comparing the performance of the **REINFORCE** and **Actor-Critic** algorithms in terms of cumulative rewards.

---

