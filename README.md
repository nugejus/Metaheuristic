# Hybrid Jaya Algorithm for CVRP

## Overview

This project implements a hybrid metaheuristic for the Capacitated Vehicle Routing Problem (CVRP) based on the Jaya algorithm.

The method is not a pure Jaya implementation.  
It combines continuous-space Jaya updates with permutation encoding, route decoding, local search, and diversification mechanisms specialized for CVRP.

The final algorithm can be described as:

**Hybrid Jaya = Jaya + random-key encoding + route decoding + local search + restart diversification**

---

## Main Idea

The original Jaya algorithm is designed for continuous optimization.  
Since CVRP is a combinatorial optimization problem, a direct application is not possible.

To adapt Jaya to CVRP, the following hybrid structure is used:

1. represent each customer by a continuous random-key value  
2. sort these values to obtain a customer permutation  
3. decode the permutation into feasible CVRP routes  
4. improve decoded solutions by local search  
5. periodically restart weak individuals to maintain diversity  

---

## Hybrid Components

### 1. Jaya + Random-Key Encoding

Jaya updates a continuous vector in the range `[0, 1]`.

Each customer is assigned a real value.  
Sorting these values produces the customer visit order.

This allows a continuous optimization method to be used for a routing problem.

**Role:**  
Transforms continuous Jaya updates into a permutation-based search process.

---

### 2. Jaya + Route Decoding

The Jaya update itself does not build routes directly.  
It only generates a customer sequence.

That sequence is then converted into CVRP routes using one of the following decoding methods:

- **Greedy split**  
  customers are scanned in order and assigned to the current route until capacity is exceeded

- **Strict-k dynamic programming split**  
  if the target number of vehicles is known, the permutation can be split into exactly `k` feasible routes using DP

**Role:**  
Converts a permutation into a feasible CVRP solution.

---

### 3. Jaya + Local Search

After decoding, the solution is improved by dedicated CVRP local search operators:

- **2-opt** for intra-route improvement
- **Relocate** for moving one customer between routes
- **Swap** for exchanging customers between two routes

This part is essential for improving route quality after the global Jaya update.

**Role:**  
Adds strong local intensification to compensate for the weakness of pure Jaya in combinatorial neighborhoods.

---

### 4. Jaya + Candidate-Based Neighborhood Restriction

Local search is not applied over all possible moves.

Instead, for each customer, only a limited set of nearest neighboring customers is stored in advance.  
This candidate list is used to restrict relocate and swap operations.

**Role:**  
Reduces computational cost while preserving the most promising local moves.

---

### 5. Jaya + Restart Diversification

To reduce premature convergence, part of the worst population is periodically replaced by new randomly initialized solutions.

This restart mechanism helps restore population diversity.

**Role:**  
Improves global exploration and prevents stagnation in poor local optima.

---

## Objective Function

The main objective is the minimization of total route distance.

Additionally, if the benchmark instance specifies the target number of vehicles, a penalty term is added:

\[
f = \text{TotalDistance} + \lambda \cdot |m - k|
\]

where:

- `m` = number of routes in the current solution
- `k` = target number of vehicles
- `λ` = penalty coefficient

This encourages the algorithm to match the benchmark vehicle count while minimizing distance.

---

## Algorithm Structure

The solution procedure can be summarized as follows:

1. load CVRP instance  
2. build distance matrix  
3. extract benchmark metadata such as optimal value and target vehicle count  
4. construct nearest-neighbor candidate lists  
5. generate initial population  
6. convert route solutions to random-key vectors  
7. update the population using Jaya  
8. decode updated vectors into feasible routes  
9. apply local search to elite and selected non-elite solutions  
10. restart part of the worst population when necessary  
11. store the best solution found  

---

## Local Search Details

### 2-opt
Improves the order of customers inside a single route by reversing route segments.

### Relocate
Moves one customer from one route to another if the move is feasible and improves the objective.

### Swap
Exchanges two customers from different routes while preserving feasibility.

These operators are applied in a first-improvement manner.

---

## Parameters

Main parameters of the algorithm:

- `pop_size` — population size
- `max_iter` — maximum number of iterations
- `candidate_k` — number of nearest neighbors used in candidate lists
- `vehicle_penalty` — penalty for mismatch in number of vehicles
- `ls_prob` — probability of applying local search to non-elite solutions
- `elite_ls_count` — number of top solutions always refined by local search
- `restart_ratio` — fraction of worst solutions replaced during restart
- `strict_k_init` — whether to use exact DP split during initialization

---

## Interpretation of Parameters

`pop_size` and `max_iter` do not change the algorithmic operators themselves.  
The update rule, decoding mechanism, and local search structure remain the same.

However, they do affect:

- search diversity
- convergence dynamics
- probability of finding high-quality solutions

Therefore, it is more accurate to say:

> These parameters do not modify the algorithmic structure, but they significantly influence the search process and convergence behavior.

---

## Why This Is a Hybrid Algorithm

This implementation should not be described as a pure Jaya algorithm.

It is hybrid because it combines:

- a **continuous global search mechanism** from Jaya
- a **permutation representation** based on random keys
- a **CVRP-specific decoding procedure**
- a **problem-oriented local search**
- a **restart-based diversification strategy**

In other words, the final performance comes from the interaction of several components, not from Jaya alone.

---

## Strengths

- simple global update mechanism
- effective adaptation of continuous Jaya to combinatorial CVRP
- feasible route construction through decoding
- strong local improvement through 2-opt, relocate, and swap
- reduced local search complexity using candidate neighbors
- better robustness through restart diversification

---

## Limitations

- random-key representation does not always reflect route structure smoothly
- solution quality depends heavily on decoding quality
- candidate-restricted search may miss beneficial moves
- runtime grows quickly with problem size
- pure Jaya contribution is difficult to isolate because performance relies on hybridization

---

## Summary

This project implements a hybrid CVRP solver in which the Jaya algorithm is used as a global search framework, while the actual routing performance is obtained through the combination of:

- random-key permutation encoding
- greedy / strict-k route decoding
- local search operators
- candidate-based neighborhood reduction
- restart-based diversification

The method is therefore best described as a **hybrid Jaya-based metaheuristic for CVRP** rather than a standalone Jaya implementation.