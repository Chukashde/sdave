#!/usr/bin/env python3
"""
Breakout Local Search (BLS) for the Maximum Cut problem.

Reference:
    Benlic, U., & Hao, J.-K. (2013). "Breakout Local Search for the Max-Cut
    problem." Engineering Applications of Artificial Intelligence, 26(3),
    1162-1173.

Idea in one breath:
    Run a plain hill-climbing descent (flip the vertex that improves the cut
    most, repeat) until you hit a local optimum. Then "break out" of it with an
    adaptive perturbation: usually a *directed* perturbation (tabu-guided,
    least-degrading flips) and, the longer you stay stuck, increasingly a
    *random* perturbation and a *larger* jump. Keep the best cut ever seen.

Input  : an adjacency-list text file (two formats supported, see read_graph()).
Output : best cut value to stdout + a solution file describing the partition.

Usage:
    python bls_maxcut.py graph.txt
    python bls_maxcut.py graph.txt -o solution.txt --time 5 --seed 1
"""

import argparse
import math
import random
import sys
import time


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #
def read_graph(path):
    """
    Parse an undirected, optionally weighted graph from an adjacency-list file.

    Two input styles are auto-detected:

    (1) "node:" adjacency format (e.g. SNAP-style conversions)::

            4039                # optional header: a bare node/edge count, ignored
            0: 1 2 3            # node 0 is adjacent to 1, 2, 3
            1: 0 48 53
            2: 0 20
            15:                 # a node with no neighbours (isolated)

        The first token ends with ':' and names the source vertex; the
        remaining tokens are its neighbours. A line whose first token does
        NOT end with ':' (such as the leading count) is ignored.

    (2) Whitespace / weighted format::

            # a small weighted graph
            a  b:2  c           # node a -> b (weight 2), c (weight 1)
            b  c:1
            c  d:3

        The first token is the source vertex; each neighbour may carry a
        weight as `neighbour:weight` (default 1.0).

    In both styles:
      * the graph is treated as UNDIRECTED, so listing edge (u, v) once is
        enough; if you also list (v, u) it is merged (last weight wins)
      * lines starting with '#' are comments; blank lines are ignored
      * self-loops are dropped (they never cross a cut)

    Returns:
        labels : list of original node ids, index i -> label
        adj    : list (len n) of lists of (neighbour_index, weight)
        m      : number of unique undirected edges
    """
    # ---- first pass: read clean lines and detect the format ----
    lines = []
    with open(path, "r") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            lines.append(line)

    # "node:" style iff some line's first token ends with a colon (e.g. "0:")
    colon_style = any(ln.split()[0].endswith(":") for ln in lines if ln.split())

    index = {}          # label -> internal integer id
    labels = []         # internal id -> label
    edges = {}          # frozenset({i, j}) -> weight  (dedups undirected edges)

    def node_id(label):
        if label not in index:
            index[label] = len(labels)
            labels.append(label)
        return index[label]

    # ---- second pass: build the graph ----
    for line in lines:
        tokens = line.split()
        if not tokens:
            continue

        if colon_style:
            head = tokens[0]
            if not head.endswith(":"):
                # bare count/header or stray line in a "node:" file -> skip,
                # so it is not mistaken for an isolated vertex
                continue
            u = node_id(head[:-1])               # strip the trailing ':'
            neighbour_tokens = tokens[1:]
        else:
            u = node_id(tokens[0])
            neighbour_tokens = tokens[1:]

        for tok in neighbour_tokens:
            if ":" in tok:
                name, w = tok.rsplit(":", 1)
                weight = float(w)
            else:
                name, weight = tok, 1.0
            v = node_id(name)
            if u == v:
                continue  # ignore self-loops (they never cross a cut)
            edges[frozenset((u, v))] = weight

    n = len(labels)
    adj = [[] for _ in range(n)]
    for pair, w in edges.items():
        i, j = tuple(pair)
        adj[i].append((j, w))
        adj[j].append((i, w))

    return labels, adj, len(edges)


def write_solution(path, labels, x, cut_value, runtime, n, m):
    """Write the partition and cut value to a text file."""
    side0 = [labels[i] for i in range(len(x)) if x[i] == 0]
    side1 = [labels[i] for i in range(len(x)) if x[i] == 1]
    with open(path, "w") as fh:
        fh.write(f"# MAX-CUT solution (Breakout Local Search)\n")
        fh.write(f"# nodes={n} edges={m} runtime={runtime:.3f}s\n")
        fh.write(f"cut_value {cut_value:g}\n\n")
        fh.write(f"# Set A ({len(side0)} nodes)\n")
        fh.write("A " + " ".join(map(str, side0)) + "\n\n")
        fh.write(f"# Set B ({len(side1)} nodes)\n")
        fh.write("B " + " ".join(map(str, side1)) + "\n\n")
        fh.write("# per-node side (node side)\n")
        for i in range(len(x)):
            fh.write(f"{labels[i]} {x[i]}\n")


# --------------------------------------------------------------------------- #
# Core BLS
# --------------------------------------------------------------------------- #
class MaxCutBLS:
    """
    State:
        x    : 0/1 side of each vertex
        g    : gain array. g[i] = how much the cut changes if vertex i flips.
               g[i] = sum over neighbours j of  w_ij * (+1 if same side else -1)
        cut  : current cut value, maintained incrementally

    Flipping a vertex is O(degree): we add g[i] to the cut, flip x[i], then
    repair the gains of its neighbours and negate g[i].
    """

    EPS = 1e-9  # tolerance so float weights don't cause infinite descent loops

    def __init__(self, adj, rng):
        self.adj = adj
        self.n = len(adj)
        self.rng = rng
        self.x = [rng.randint(0, 1) for _ in range(self.n)]
        self.g = [0.0] * self.n
        self.cut = 0.0
        self._recompute_from_scratch()

    def _recompute_from_scratch(self):
        g = [0.0] * self.n
        x = self.x
        cut = 0.0
        for i in range(self.n):
            gi = 0.0
            for (j, w) in self.adj[i]:
                gi += w if x[i] == x[j] else -w
                if i < j and x[i] != x[j]:
                    cut += w
            g[i] = gi
        self.g = g
        self.cut = cut

    def flip(self, i):
        """Flip vertex i and keep cut + gains consistent. O(degree(i))."""
        self.cut += self.g[i]
        xi_old = self.x[i]
        self.x[i] ^= 1
        for (j, w) in self.adj[i]:
            if self.x[j] == xi_old:      # j was on the same side as old x[i]
                self.g[j] -= 2.0 * w
            else:                        # j was on the opposite side
                self.g[j] += 2.0 * w
        self.g[i] = -self.g[i]

    def descend(self):
        """Best-improvement hill climbing to a local optimum."""
        while True:
            best_i, best_gain = -1, self.EPS
            g = self.g
            for i in range(self.n):
                if g[i] > best_gain:
                    best_gain, best_i = g[i], i
            if best_i == -1:
                break
            self.flip(best_i)

    def cut_from_scratch(self):
        """Independently recompute the cut value (used for verification)."""
        x = self.x
        total = 0.0
        for i in range(self.n):
            for (j, w) in self.adj[i]:
                if i < j and x[i] != x[j]:
                    total += w
        return total

    def snapshot(self):
        return list(self.x), self.cut


# --------------------------------------------------------------------------- #
# Perturbations
# --------------------------------------------------------------------------- #
def directed_perturbation(s, L, it, tabu_until, tenure, best_cut):
    """
    Tabu-guided, least-degrading jump: repeatedly flip the non-tabu vertex with
    the largest gain (so we lose as little cut weight as possible while still
    moving), marking each flipped vertex tabu for `tenure` steps. Aspiration:
    a tabu vertex is allowed if flipping it would beat the best known cut.
    """
    n = s.n
    for _ in range(L):
        cand, cand_gain = -1, -math.inf
        for i in range(n):
            allowed = it >= tabu_until[i]
            if not allowed and (s.cut + s.g[i] > best_cut + s.EPS):
                allowed = True  # aspiration
            if allowed and s.g[i] > cand_gain:
                cand_gain, cand = s.g[i], i
        if cand == -1:  # everything tabu -> take global best gain
            cand = max(range(n), key=lambda i: s.g[i])
        s.flip(cand)
        tabu_until[cand] = it + tenure
        it += 1
    return it


def random_perturbation(s, L):
    """Flip L distinct random vertices."""
    for i in s.rng.sample(range(s.n), min(L, s.n)):
        s.flip(i)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run_bls(adj, *, seed=0, time_limit=36000.0, max_no_improve=200000,
            L0=None, tabu_tenure=None, p_min=0.1, verbose=True):
    """
    Returns (best_x, best_cut).

    Adaptive control (omega = consecutive non-improving local optima):
      * jump magnitude L grows once we have been stuck a while
      * probability of a *directed* perturbation decays as exp(-omega/decay),
        so a stuck search diversifies with random perturbations instead
    """
    n = len(adj)
    rng = random.Random(seed)
    if n == 0:
        return [], 0.0

    # sensible defaults scaled to the graph size
    if L0 is None:
        L0 = max(1, n // 20)
    if tabu_tenure is None:
        tabu_tenure = max(1, n // 10)
    decay = max(1.0, max_no_improve / 8.0)

    s = MaxCutBLS(adj, rng)
    s.descend()
    best_x, best_cut = s.snapshot()

    tabu_until = [0] * n
    it = 0          # global move counter (drives tabu tenure)
    omega = 0       # consecutive non-improving local optima
    no_improve = 0  # local optima since last global improvement
    start = time.time()

    while no_improve < max_no_improve and (time.time() - start) < time_limit:
        # --- adapt jump magnitude and perturbation type to the stagnation ---
        if omega > tabu_tenure:                       # badly stuck -> jump far
            L = L0 + rng.randint(0, L0)
        else:
            L = L0
        p_directed = max(p_min, math.exp(-omega / decay))

        if rng.random() < p_directed:
            it = directed_perturbation(s, L, it, tabu_until, tabu_tenure, best_cut)
        else:
            random_perturbation(s, L)

        # --- descend back to a local optimum and bookkeep ---
        s.descend()
        if s.cut > best_cut + s.EPS:
            best_x, best_cut = s.snapshot()
            omega = 0
            no_improve = 0
            if verbose:
                print(f"  [{time.time()-start:6.2f}s] new best cut = {best_cut:g}",
                      file=sys.stderr)
        else:
            omega += 1
            no_improve += 1

    return best_x, best_cut


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="Breakout Local Search for MAX-CUT (adjacency-list input).")
    ap.add_argument("input", help="adjacency-list text file")
    ap.add_argument("-o", "--output", default="maxcut_solution.txt",
                    help="solution output file (default: maxcut_solution.txt)")
    ap.add_argument("--time", type=float, default=2.0,
                    help="time limit in seconds (default: 2.0)")
    ap.add_argument("--max-no-improve", type=int, default=2000,
                    help="stop after this many non-improving local optima")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--L0", type=int, default=None, help="base jump magnitude")
    ap.add_argument("--tabu", type=int, default=None, help="tabu tenure")
    ap.add_argument("--quiet", action="store_true", help="suppress progress")
    args = ap.parse_args()

    labels, adj, m = read_graph(args.input)
    n = len(labels)
    print(f"Loaded graph: {n} nodes, {m} edges", file=sys.stderr)

    t0 = time.time()
    best_x, best_cut = run_bls(
        adj, seed=args.seed, time_limit=args.time,
        max_no_improve=args.max_no_improve, L0=args.L0,
        tabu_tenure=args.tabu, verbose=not args.quiet)
    runtime = time.time() - t0

    write_solution(args.output, labels, best_x, best_cut, runtime, n, m)
    print(f"Best cut value: {best_cut:g}")
    print(f"Solution written to: {args.output}")


if __name__ == "__main__":
    main()