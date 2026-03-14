import argparse
import copy
import re
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import vrplib


@dataclass
class SolutionState:
    routes: List[List[int]]
    route_loads: np.ndarray
    route_costs: np.ndarray
    total_distance: float
    objective: float
    customer_route: np.ndarray
    customer_pos: np.ndarray


class Jaya:
    def __init__(
        self,
        pop_size: int = 30,
        max_iter: int = 150,
        seed: int = 42,
        candidate_k: int = 12,
        vehicle_penalty: Optional[float] = None,
        use_vehicle_penalty: bool = True,
        ls_prob: float = 0.25,
        elite_ls_count: int = 3,
        restart_ratio: float = 0.20,
        strict_k_init: bool = False,
    ):
        self._pop_size = pop_size
        self._max_iter = max_iter
        self._rng = np.random.default_rng(seed)

        self._candidate_k = candidate_k
        self._vehicle_penalty = vehicle_penalty
        self._use_vehicle_penalty = use_vehicle_penalty
        self._ls_prob = ls_prob
        self._elite_ls_count = elite_ls_count
        self._restart_ratio = restart_ratio
        self._strict_k_init = strict_k_init

        # instance data
        self._instance: Optional[Dict[str, Any]] = None
        self._instance_name: str = ""
        self._dist: Optional[np.ndarray] = None
        self._demands: Optional[np.ndarray] = None
        self._capacity: int = 0
        self._depot: int = 0
        self._num_nodes: int = 0
        self._num_customers: int = 0
        self._customers: List[int] = []
        self._customer_array: Optional[np.ndarray] = None
        self._customer_to_pos: Dict[int, int] = {}

        # benchmark metadata
        self._optimal_value: Optional[int] = None
        self._target_vehicles: Optional[int] = None

        # search helpers
        self._candidate_neighbors: Optional[np.ndarray] = None

        # best state
        self._best_solution: Optional[SolutionState] = None
        self._best_cost: float = float("inf")
        self._best_raw_distance: float = float("inf")
        self._fitness_history: List[float] = []

    # ============================================================
    # Instance loading
    # ============================================================

    def _load_instance(self, vrp_file: str) -> None:
        inst = vrplib.read_instance(vrp_file)
        self._instance = inst
        self._instance_name = str(inst.get("name", "Unknown")).strip()

        self._demands = np.asarray(inst["demand"], dtype=np.int32)
        self._num_nodes = len(self._demands)
        self._capacity = int(inst["capacity"])

        depot_raw = inst.get("depot", np.array([1]))
        depot_value = int(np.asarray(depot_raw).reshape(-1)[0])

        # robust 0-based handling
        if 1 <= depot_value < self._num_nodes:
            self._depot = depot_value - 1
        elif 0 <= depot_value <= self._num_nodes:
            self._depot = depot_value
        else:
            raise ValueError(f"Invalid depot index: {depot_value}")

        self._dist = self._build_distance_matrix(inst).astype(np.float64)

        self._customers = [i for i in range(self._num_nodes) if i != self._depot]
        self._customer_array = np.asarray(self._customers, dtype=np.int32)
        self._num_customers = len(self._customers)
        self._customer_to_pos = {c: i for i, c in enumerate(self._customers)}

        self._parse_metadata()

        if self._vehicle_penalty is None:
            # practical default: big enough to strongly discourage wrong vehicle count
            max_edge = float(np.max(self._dist))
            self._vehicle_penalty = max(1000.0, max_edge * 50.0)

        self._build_candidate_neighbors()

        if np.any(self._demands[self._customer_array] > self._capacity):
            raise ValueError("At least one customer demand exceeds vehicle capacity")

    def _build_distance_matrix(self, inst: Dict[str, Any]) -> np.ndarray:
        if "edge_weight" in inst:
            return np.asarray(inst["edge_weight"], dtype=np.float64)

        if "node_coord" not in inst:
            raise ValueError("Instance has neither edge_weight nor node_coord")

        coords = np.asarray(inst["node_coord"], dtype=np.float64)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("node_coord must have shape (n, 2)")

        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=2))

        edge_weight_type = str(inst.get("edge_weight_type", "")).upper().strip()
        if edge_weight_type == "EUC_2D":
            # TSPLIB style integer euclidean
            dist = np.rint(dist)

        return dist

    def _parse_metadata(self) -> None:
        comment = str(self._instance.get("comment", "")).strip()

        # optimal
        m_opt = re.search(
            r"(optimal value|best value|optimal|best)\s*:\s*(\d+)",
            comment,
            re.IGNORECASE,
        )
        if m_opt:
            self._optimal_value = int(m_opt.group(2))

        # target vehicles from name E-n76-k10
        m_k = re.search(r"-k(\d+)\b", self._instance_name, re.IGNORECASE)
        if m_k:
            self._target_vehicles = int(m_k.group(1))
        else:
            m_trucks = re.search(
                r"(min\s*no\s*of\s*trucks|vehicles|trucks)\s*:\s*(\d+)",
                comment,
                re.IGNORECASE,
            )
            if m_trucks:
                self._target_vehicles = int(m_trucks.group(2))

    def _build_candidate_neighbors(self) -> None:
        """
        For each customer, store nearest candidate_k customers (excluding depot, excluding self).
        """
        n = self._num_customers
        k = min(self._candidate_k, max(1, n - 1))
        neighbors = np.empty((n, k), dtype=np.int32)

        cust = self._customer_array
        sub_dist = self._dist[np.ix_(cust, cust)].copy()
        np.fill_diagonal(sub_dist, np.inf)

        for i in range(n):
            idx = np.argpartition(sub_dist[i], kth=k - 1)[:k]
            # stable ordering by actual distance
            idx = idx[np.argsort(sub_dist[i, idx])]
            neighbors[i] = cust[idx]

        self._candidate_neighbors = neighbors

    # ============================================================
    # Objective / evaluation
    # ============================================================

    def _route_demand(self, route: List[int]) -> int:
        if not route:
            return 0
        return int(self._demands[np.asarray(route, dtype=np.int32)].sum())

    def _route_distance(self, route: List[int]) -> float:
        if not route:
            return 0.0
        total = self._dist[self._depot, route[0]]
        for i in range(len(route) - 1):
            total += self._dist[route[i], route[i + 1]]
        total += self._dist[route[-1], self._depot]
        return float(total)

    def _vehicle_penalty_value(self, n_routes: int) -> float:
        if not self._use_vehicle_penalty or self._target_vehicles is None:
            return 0.0
        return self._vehicle_penalty * abs(n_routes - self._target_vehicles)

    def _objective_from_distance(self, total_distance: float, n_routes: int) -> float:
        return total_distance + self._vehicle_penalty_value(n_routes)

    def get_gap(self) -> Optional[float]:
        if self._optimal_value is None or not np.isfinite(self._best_raw_distance):
            return None
        return 100.0 * (self._best_raw_distance - self._optimal_value) / self._optimal_value

    # ============================================================
    # Encoding / decoding
    # ============================================================

    def _solution_to_vector(self, routes: List[List[int]]) -> np.ndarray:
        flat: List[int] = []
        for r in routes:
            flat.extend(r)

        n = len(flat)
        vec = np.zeros(self._num_customers, dtype=np.float64)
        if n == 0:
            return vec

        for pos, c in enumerate(flat):
            vec[self._customer_to_pos[c]] = (pos + 1) / (n + 1)
        return vec

    def _vector_to_permutation(self, vec: np.ndarray) -> List[int]:
        order = np.argsort(vec)
        return [self._customers[i] for i in order]

    def _decode_greedy(self, perm: List[int]) -> List[List[int]]:
        routes: List[List[int]] = []
        current: List[int] = []
        load = 0

        for c in perm:
            d = int(self._demands[c])
            if load + d <= self._capacity:
                current.append(c)
                load += d
            else:
                if current:
                    routes.append(current)
                current = [c]
                load = d

        if current:
            routes.append(current)

        return routes

    def _decode_strict_k_dp(self, perm: List[int], k: int) -> List[List[int]]:
        """
        Exact split to k routes using DP.
        Slower. Use only for optional initialization or controlled cases.
        """
        n = len(perm)
        seg_cost = np.full((n, n), np.inf, dtype=np.float64)

        for i in range(n):
            load = 0
            dist = 0.0
            prev = self._depot
            for j in range(i, n):
                c = perm[j]
                load += int(self._demands[c])
                if load > self._capacity:
                    break
                if j == i:
                    dist = self._dist[self._depot, c]
                else:
                    dist += self._dist[prev, c]
                prev = c
                seg_cost[i, j] = dist + self._dist[c, self._depot]

        dp = np.full((k + 1, n + 1), np.inf, dtype=np.float64)
        prev_cut = np.full((k + 1, n + 1), -1, dtype=np.int32)
        dp[0, 0] = 0.0

        for t in range(1, k + 1):
            for j in range(1, n + 1):
                lo = t - 1
                hi = j - 1
                for i in range(lo, hi + 1):
                    if not np.isfinite(dp[t - 1, i]):
                        continue
                    cost = seg_cost[i, j - 1]
                    if not np.isfinite(cost):
                        continue
                    cand = dp[t - 1, i] + cost
                    if cand < dp[t, j]:
                        dp[t, j] = cand
                        prev_cut[t, j] = i

        if not np.isfinite(dp[k, n]):
            return self._decode_greedy(perm)

        routes: List[List[int]] = []
        t, j = k, n
        while t > 0:
            i = int(prev_cut[t, j])
            routes.append(perm[i:j])
            j = i
            t -= 1
        routes.reverse()
        return routes

    # ============================================================
    # State build / maintenance
    # ============================================================

    def _build_state(self, routes: List[List[int]]) -> SolutionState:
        routes = [r[:] for r in routes if r]
        m = len(routes)

        route_loads = np.zeros(m, dtype=np.int32)
        route_costs = np.zeros(m, dtype=np.float64)

        customer_route = np.full(self._num_nodes, -1, dtype=np.int32)
        customer_pos = np.full(self._num_nodes, -1, dtype=np.int32)

        total_distance = 0.0
        for ridx, route in enumerate(routes):
            load = self._route_demand(route)
            cost = self._route_distance(route)
            route_loads[ridx] = load
            route_costs[ridx] = cost
            total_distance += cost

            for pos, c in enumerate(route):
                customer_route[c] = ridx
                customer_pos[c] = pos

        obj = self._objective_from_distance(total_distance, m)

        return SolutionState(
            routes=routes,
            route_loads=route_loads,
            route_costs=route_costs,
            total_distance=total_distance,
            objective=obj,
            customer_route=customer_route,
            customer_pos=customer_pos,
        )

    def _refresh_indices(self, state: SolutionState) -> None:
        state.customer_route.fill(-1)
        state.customer_pos.fill(-1)
        for ridx, route in enumerate(state.routes):
            for pos, c in enumerate(route):
                state.customer_route[c] = ridx
                state.customer_pos[c] = pos

    def _recompute_route(self, state: SolutionState, ridx: int) -> None:
        route = state.routes[ridx]
        old_cost = state.route_costs[ridx]
        new_cost = self._route_distance(route)
        new_load = self._route_demand(route)

        state.route_costs[ridx] = new_cost
        state.route_loads[ridx] = new_load
        state.total_distance += new_cost - old_cost

    def _recompute_objective(self, state: SolutionState) -> None:
        state.objective = self._objective_from_distance(state.total_distance, len(state.routes))

    def _cleanup_empty_routes(self, state: SolutionState) -> None:
        if all(len(r) > 0 for r in state.routes):
            return
        routes = [r for r in state.routes if r]
        new_state = self._build_state(routes)
        state.routes = new_state.routes
        state.route_loads = new_state.route_loads
        state.route_costs = new_state.route_costs
        state.total_distance = new_state.total_distance
        state.objective = new_state.objective
        state.customer_route = new_state.customer_route
        state.customer_pos = new_state.customer_pos

    # ============================================================
    # Initial population
    # ============================================================

    def _create_initial_solution(self) -> SolutionState:
        perm = self._customers.copy()
        self._rng.shuffle(perm)

        if self._strict_k_init and self._target_vehicles is not None:
            routes = self._decode_strict_k_dp(perm, self._target_vehicles)
        else:
            routes = self._decode_greedy(perm)

        state = self._build_state(routes)

        # cheap initial polish: 2-opt only
        self._run_two_opt_pass(state)
        return state

    # ============================================================
    # Delta helpers
    # ============================================================

    def _prev_node(self, route: List[int], pos: int) -> int:
        return self._depot if pos == 0 else route[pos - 1]

    def _next_node(self, route: List[int], pos: int) -> int:
        return self._depot if pos == len(route) - 1 else route[pos + 1]

    def _delta_remove(self, route: List[int], pos: int) -> float:
        a = self._prev_node(route, pos)
        b = route[pos]
        c = self._next_node(route, pos)
        return float(self._dist[a, c] - self._dist[a, b] - self._dist[b, c])

    def _delta_insert(self, route: List[int], insert_pos: int, customer: int) -> float:
        left = self._depot if insert_pos == 0 else route[insert_pos - 1]
        right = self._depot if insert_pos == len(route) else route[insert_pos]
        return float(self._dist[left, customer] + self._dist[customer, right] - self._dist[left, right])

    # ============================================================
    # 2-opt (O(1) delta)
    # ============================================================

    def _two_opt_route_first_improvement(self, state: SolutionState, ridx: int) -> bool:
        route = state.routes[ridx]
        n = len(route)
        if n < 4:
            return False

        for i in range(n - 2):
            a = self._depot if i == 0 else route[i - 1]
            b = route[i]

            for j in range(i + 1, n - 1):
                c = route[j]
                d = route[j + 1]

                delta = self._dist[a, c] + self._dist[b, d] - self._dist[a, b] - self._dist[c, d]
                if delta < -1e-9:
                    route[i:j + 1] = reversed(route[i:j + 1])
                    self._recompute_route(state, ridx)
                    self._refresh_indices(state)
                    self._recompute_objective(state)
                    return True

            # allow edge to depot on right
            j = n - 1
            c = route[j]
            d = self._depot
            delta = self._dist[a, c] + self._dist[b, d] - self._dist[a, b] - self._dist[c, d]
            if delta < -1e-9:
                route[i:j + 1] = reversed(route[i:j + 1])
                self._recompute_route(state, ridx)
                self._refresh_indices(state)
                self._recompute_objective(state)
                return True

        return False

    def _run_two_opt_pass(self, state: SolutionState) -> None:
        improved = True
        while improved:
            improved = False
            route_order = list(range(len(state.routes)))
            self._rng.shuffle(route_order)
            for ridx in route_order:
                if self._two_opt_route_first_improvement(state, ridx):
                    improved = True
                    break

    # ============================================================
    # Candidate-based relocate
    # ============================================================

    def _candidate_routes_for_customer(self, state: SolutionState, customer: int) -> List[int]:
        pos = self._customer_to_pos[customer]
        neigh = self._candidate_neighbors[pos]
        routes = []
        seen = set()

        for nb in neigh:
            ridx = int(state.customer_route[nb])
            if ridx >= 0 and ridx not in seen:
                routes.append(ridx)
                seen.add(ridx)

        return routes

    def _relocate_first_improvement(self, state: SolutionState) -> bool:
        customers = self._customers.copy()
        self._rng.shuffle(customers)

        for u in customers:
            r1 = int(state.customer_route[u])
            if r1 < 0:
                continue

            route1 = state.routes[r1]
            pos1 = int(state.customer_pos[u])

            if len(route1) <= 1:
                continue

            demand_u = int(self._demands[u])
            delta_rem = self._delta_remove(route1, pos1)

            target_routes = self._candidate_routes_for_customer(state, u)
            self._rng.shuffle(target_routes)

            for r2 in target_routes:
                if r2 == r1:
                    continue

                if state.route_loads[r2] + demand_u > self._capacity:
                    continue

                route2 = state.routes[r2]

                # insertion positions around nearest neighbors only
                positions = set()
                for nb in self._candidate_neighbors[self._customer_to_pos[u]]:
                    if state.customer_route[nb] == r2:
                        p = int(state.customer_pos[nb])
                        positions.add(p)
                        positions.add(p + 1)

                if not positions:
                    positions = {0, len(route2)}
                else:
                    positions.add(0)
                    positions.add(len(route2))

                pos_candidates = list(sorted(p for p in positions if 0 <= p <= len(route2)))
                self._rng.shuffle(pos_candidates)

                for ins_pos in pos_candidates:
                    delta_ins = self._delta_insert(route2, ins_pos, u)
                    delta_total = delta_rem + delta_ins

                    old_n_routes = len(state.routes)
                    new_n_routes = old_n_routes - (1 if len(route1) == 1 else 0)
                    penalty_delta = self._vehicle_penalty_value(new_n_routes) - self._vehicle_penalty_value(old_n_routes)

                    if delta_total + penalty_delta < -1e-9:
                        # apply
                        route1.pop(pos1)
                        route2.insert(ins_pos, u)

                        self._cleanup_empty_routes(state)
                        if r1 < len(state.routes):
                            self._recompute_route(state, r1)
                        if r2 < len(state.routes):
                            self._recompute_route(state, r2)
                        self._refresh_indices(state)
                        self._recompute_objective(state)
                        return True

        return False

    # ============================================================
    # Candidate-based inter-route swap
    # ============================================================

    def _swap_delta_interroute(
        self,
        route1: List[int],
        pos1: int,
        route2: List[int],
        pos2: int,
    ) -> float:
        u = route1[pos1]
        v = route2[pos2]

        a = self._prev_node(route1, pos1)
        b = self._next_node(route1, pos1)
        c = self._prev_node(route2, pos2)
        d = self._next_node(route2, pos2)

        old_cost = self._dist[a, u] + self._dist[u, b] + self._dist[c, v] + self._dist[v, d]
        new_cost = self._dist[a, v] + self._dist[v, b] + self._dist[c, u] + self._dist[u, d]
        return float(new_cost - old_cost)

    def _swap_first_improvement(self, state: SolutionState) -> bool:
        customers = self._customers.copy()
        self._rng.shuffle(customers)

        for u in customers:
            r1 = int(state.customer_route[u])
            if r1 < 0:
                continue
            pos1 = int(state.customer_pos[u])
            route1 = state.routes[r1]
            du = int(self._demands[u])

            for v in self._candidate_neighbors[self._customer_to_pos[u]]:
                r2 = int(state.customer_route[v])
                if r2 < 0 or r2 == r1:
                    continue

                pos2 = int(state.customer_pos[v])
                route2 = state.routes[r2]
                dv = int(self._demands[v])

                new_load1 = state.route_loads[r1] - du + dv
                new_load2 = state.route_loads[r2] - dv + du
                if new_load1 > self._capacity or new_load2 > self._capacity:
                    continue

                delta = self._swap_delta_interroute(route1, pos1, route2, pos2)
                if delta < -1e-9:
                    route1[pos1], route2[pos2] = route2[pos2], route1[pos1]

                    self._recompute_route(state, r1)
                    self._recompute_route(state, r2)
                    self._refresh_indices(state)
                    self._recompute_objective(state)
                    return True

        return False

    # ============================================================
    # Local search
    # ============================================================

    def _local_search(self, state: SolutionState) -> None:
        improved = True
        while improved:
            improved = False

            # 1) relocate
            if self._relocate_first_improvement(state):
                improved = True
                self._run_two_opt_pass(state)
                continue

            # 2) swap
            if self._swap_first_improvement(state):
                improved = True
                self._run_two_opt_pass(state)

    # ============================================================
    # Jaya update
    # ============================================================

    def _jaya_update(self, population: np.ndarray, best: np.ndarray, worst: np.ndarray) -> np.ndarray:
        r1 = self._rng.random(population.shape)
        r2 = self._rng.random(population.shape)
        new_pop = population + r1 * (best - np.abs(population)) - r2 * (worst - np.abs(population))
        return np.clip(new_pop, 0.0, 1.0)

    # ============================================================
    # Solve
    # ============================================================

    def predict(self, vrp_file: str) -> Dict[str, Any]:
        self._load_instance(vrp_file)

        start = time.time()
        self._fitness_history = []
        self._best_solution = None
        self._best_cost = float("inf")
        self._best_raw_distance = float("inf")

        print(
            f"Running FastJayaCVRP | pop={self._pop_size}, iter={self._max_iter}, "
            f"target_k={self._target_vehicles}, candidate_k={self._candidate_k}, "
            f"vehicle_penalty={self._vehicle_penalty:.1f}"
        )

        population_vectors = np.zeros((self._pop_size, self._num_customers), dtype=np.float64)
        population_states: List[SolutionState] = []
        population_costs = np.full(self._pop_size, np.inf, dtype=np.float64)

        # init
        for i in range(self._pop_size):
            st = self._create_initial_solution()

            population_states.append(st)
            population_vectors[i] = self._solution_to_vector(st.routes)
            population_costs[i] = st.objective

            if st.objective < self._best_cost:
                self._best_cost = st.objective
                self._best_raw_distance = st.total_distance
                self._best_solution = copy.deepcopy(st)

        self._fitness_history.append(self._best_cost)

        # main loop
        for it in range(self._max_iter):
            best_idx = int(np.argmin(population_costs))
            worst_idx = int(np.argmax(population_costs))

            new_vectors = self._jaya_update(
                population_vectors,
                population_vectors[best_idx],
                population_vectors[worst_idx],
            )

            # elite local search first
            elite_indices = np.argsort(population_costs)[: min(self._elite_ls_count, self._pop_size)]
            elite_set = set(int(x) for x in elite_indices)

            for i in range(self._pop_size):
                perm = self._vector_to_permutation(new_vectors[i])
                if self._target_vehicles is not None:
                    routes = self._decode_strict_k_dp(perm, self._target_vehicles)
                else:
                    routes = self._decode_greedy(perm)
                st = self._build_state(routes)

                if i in elite_set or self._rng.random() < self._ls_prob:
                    self._local_search(st)

                if st.objective + 1e-9 < population_costs[i]:
                    population_states[i] = st
                    population_vectors[i] = self._solution_to_vector(st.routes)
                    population_costs[i] = st.objective

                    if st.objective + 1e-9 < self._best_cost:
                        self._best_cost = st.objective
                        self._best_raw_distance = st.total_distance
                        self._best_solution = copy.deepcopy(st)

            # restart worst fraction occasionally
            if it >= max(20, self._max_iter // 5):
                n_replace = max(1, int(self._pop_size * self._restart_ratio))
                worst_indices = np.argsort(population_costs)[-n_replace:]

                for idx in worst_indices:
                    if self._rng.random() < 0.35:
                        st = self._create_initial_solution()
                        population_states[idx] = st
                        population_vectors[idx] = self._solution_to_vector(st.routes)
                        population_costs[idx] = st.objective

                        if st.objective + 1e-9 < self._best_cost:
                            self._best_cost = st.objective
                            self._best_raw_distance = st.total_distance
                            self._best_solution = copy.deepcopy(st)

            self._fitness_history.append(self._best_cost)

            if (it + 1) % 20 == 0 or it == 0:
                gap = self.get_gap()
                gap_str = f"{gap:.2f}%" if gap is not None else "N/A"
                elapsed = time.time() - start
                print(
                    f"Iter {it + 1:4d}/{self._max_iter} | "
                    f"Obj {self._best_cost:.2f} | Dist {self._best_raw_distance:.2f} | "
                    f"Gap {gap_str} | Time {elapsed:.1f}s"
                )

        # final polish on best
        final_state = copy.deepcopy(self._best_solution)
        self._local_search(final_state)
        if final_state.objective < self._best_cost:
            self._best_solution = final_state
            self._best_cost = final_state.objective
            self._best_raw_distance = final_state.total_distance

        elapsed = time.time() - start
        print(f"Done in {elapsed:.2f}s")

        return self._prepare_output()

    # ============================================================
    # Output
    # ============================================================

    def _prepare_output(self) -> Dict[str, Any]:
        if self._best_solution is None:
            raise RuntimeError("No solution available")

        routes_1indexed = [[c + 1 for c in route] for route in self._best_solution.routes]

        route_details = []
        for i, route in enumerate(self._best_solution.routes):
            demand = self._route_demand(route)
            dist = self._route_distance(route)
            route_details.append(
                {
                    "route_id": i + 1,
                    "customers_internal": route[:],
                    "customers_1indexed": [c + 1 for c in route],
                    "num_customers": len(route),
                    "demand": int(demand),
                    "distance": round(float(dist), 2),
                    "capacity_used_pct": round(100.0 * demand / self._capacity, 2),
                }
            )

        print("num routes:", len(self._best_solution.routes))
        print("route loads:", self._best_solution.route_loads.tolist())
        print("routes 1-indexed:", [[c+1 for c in r] for r in self._best_solution.routes])
        print("raw distance:", self._best_raw_distance)

        return {
            "instance_name": self._instance_name,
            "objective": round(float(self._best_cost), 2),
            "total_distance": round(float(self._best_raw_distance), 2),
            "num_vehicles": len(self._best_solution.routes),
            "target_vehicles": self._target_vehicles,
            "optimal_value": self._optimal_value,
            "gap_pct": None if self.get_gap() is None else round(float(self.get_gap()), 2),
            "routes_internal": [r[:] for r in self._best_solution.routes],
            "routes_1indexed": routes_1indexed,
            "route_details": route_details,
            "fitness_history": [float(x) for x in self._fitness_history],
        }


def main():
    parser = argparse.ArgumentParser(description="Fast optimized Jaya solver for CVRP")
    parser.add_argument("vrp_file", type=str, help="Path to VRP instance")
    parser.add_argument("--pop_size", type=int, default=30, help="Population size")
    parser.add_argument("--max_iter", type=int, default=60, help="Maximum iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--candidate_k", type=int, default=12, help="Nearest-neighbor candidate list size")
    parser.add_argument("--vehicle_penalty", type=float, default=None, help="Penalty for wrong number of routes")
    parser.add_argument("--no_vehicle_penalty", action="store_true", help="Disable vehicle-count penalty")
    parser.add_argument("--ls_prob", type=float, default=0.25, help="Local-search probability for non-elite solutions")
    parser.add_argument("--elite_ls_count", type=int, default=3, help="Always local-search top elite solutions")
    parser.add_argument("--restart_ratio", type=float, default=0.20, help="Worst-population restart ratio")
    parser.add_argument("--strict_k_init", action="store_true", help="Use exact DP split only at initialization")
    parser.add_argument("--fast", action="store_true", help="Fast mode")
    parser.add_argument("--output", "-o", type=str, help="Optional output file")

    args = parser.parse_args()

    if args.fast:
        args.pop_size = 16
        args.max_iter = 80
        args.candidate_k = 8
        args.elite_ls_count = 2
        args.ls_prob = 0.15
        print("Fast mode enabled")

    solver = Jaya(
        pop_size=args.pop_size,
        max_iter=args.max_iter,
        seed=args.seed,
        candidate_k=args.candidate_k,
        vehicle_penalty=args.vehicle_penalty,
        use_vehicle_penalty=not args.no_vehicle_penalty,
        ls_prob=args.ls_prob,
        elite_ls_count=args.elite_ls_count,
        restart_ratio=args.restart_ratio,
        strict_k_init=args.strict_k_init,
    )

    try:
        result = solver.predict(args.vrp_file)

        print("\n=== Result Summary ===")
        print(f"Instance         : {result['instance_name']}")
        print(f"Objective        : {result['objective']}")
        print(f"Total distance   : {result['total_distance']}")
        print(f"Vehicles used    : {result['num_vehicles']}")
        print(f"Target vehicles  : {result['target_vehicles']}")

        if result["optimal_value"] is not None:
            print(f"Optimal value    : {result['optimal_value']}")
            print(f"Gap              : {result['gap_pct']}%")

        print("\n=== Routes ===")
        for rd in result["route_details"]:
            print(
                f"Route #{rd['route_id']:2d}: {rd['customers_1indexed']} | "
                f"Dist={rd['distance']:.2f}, Demand={rd['demand']}, "
                f"Util={rd['capacity_used_pct']:.1f}%"
            )

        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(f"Instance: {result['instance_name']}\n")
                f.write(f"Objective: {result['objective']}\n")
                f.write(f"Total distance: {result['total_distance']}\n")
                f.write(f"Vehicles used: {result['num_vehicles']}\n")
                f.write(f"Target vehicles: {result['target_vehicles']}\n")
                if result["optimal_value"] is not None:
                    f.write(f"Optimal value: {result['optimal_value']}\n")
                    f.write(f"Gap: {result['gap_pct']}%\n")
                f.write("\nRoutes:\n")
                for rd in result["route_details"]:
                    f.write(
                        f"Route #{rd['route_id']}: {rd['customers_1indexed']} | "
                        f"Distance={rd['distance']:.2f}, Demand={rd['demand']}, "
                        f"Utilization={rd['capacity_used_pct']:.1f}%\n"
                    )
            print(f"\nSaved to: {args.output}")

    except FileNotFoundError:
        print(f"Error: file not found: {args.vrp_file}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()