import argparse
import math
import random
import time

import numpy as np

from pigat.data.loaders import load_graph


# ============================================================
# Utilities
# ============================================================

def cut_value(g, x):
    total = 0.0
    for u, v, data in g.edges(data=True):
        if x[u] != x[v]:
            total += float(data.get("weight", 1.0))
    return total


def total_edge_weight(g):
    return sum(
        float(data.get("weight", 1.0))
        for _, _, data in g.edges(data=True)
    )


# ============================================================
# Gain computation
# ============================================================

def initial_gains(g, x):
    n = g.number_of_nodes()
    gain = np.zeros(n, dtype=np.float64)

    for v in range(n):
        gv = 0.0

        for u, data in g[v].items():
            w = float(data.get("weight", 1.0))

            if x[v] == x[u]:
                gv += w
            else:
                gv -= w

        gain[v] = gv

    return gain


def flip_vertex(g, x, gain, v):
    delta = gain[v]
    old_side = x[v]

    for u, data in g[v].items():
        w = float(data.get("weight", 1.0))

        if old_side == x[u]:
            gain[u] -= 2.0 * w
        else:
            gain[u] += 2.0 * w

    x[v] = 1 - x[v]
    gain[v] = -gain[v]

    return float(delta)


# ============================================================
# M1 / M2 / M3
# ============================================================

def best_m1(gain, tabu_until, iteration, current_cut, best_cut):
    """
    M1:
    Highest-gain single vertex move satisfying tabu/aspiration.
    """
    order = np.argsort(-gain)

    for v in order:
        v = int(v)

        aspiration = current_cut + gain[v] > best_cut + 1e-12
        allowed = iteration >= tabu_until[v]

        if allowed or aspiration:
            return v

    return int(order[0])


def pair_gain(g, x, gain, v1, v2):
    """
    Gain of simultaneously flipping v1 and v2.

    M2 chooses one vertex from each partition.
    If v1-v2 is an edge, both endpoints flip, so that edge keeps
    its cut status. The correction removes the double-counting
    from gain[v1] + gain[v2].
    """
    pg = gain[v1] + gain[v2]

    if g.has_edge(v1, v2):
        w = float(g[v1][v2].get("weight", 1.0))

        # v1 and v2 are from opposite partitions, so this edge
        # is currently cut. Individual gains count -w twice,
        # while simultaneous flip leaves it cut.
        pg += 2.0 * w

    return float(pg)


def best_vertex_in_side(
    x,
    gain,
    side,
    tabu_until,
    iteration
):
    vertices = np.where(x == side)[0]

    if len(vertices) == 0:
        return None

    eligible = [
        int(v)
        for v in vertices
        if iteration >= tabu_until[int(v)]
    ]

    if eligible:
        return max(
            eligible,
            key=lambda v: gain[v]
        )

    return int(
        vertices[np.argmax(gain[vertices])]
    )


def best_m2(
    g,
    x,
    gain,
    tabu_until,
    iteration,
    current_cut,
    best_cut
):
    """
    M2:
    Highest-gain vertex from V1 and highest-gain vertex from V2.
    """

    v0 = best_vertex_in_side(
        x,
        gain,
        0,
        tabu_until,
        iteration
    )

    v1 = best_vertex_in_side(
        x,
        gain,
        1,
        tabu_until,
        iteration
    )

    if v0 is None or v1 is None:
        return None

    pg = pair_gain(
        g,
        x,
        gain,
        v0,
        v1
    )

    # Aspiration: pair would improve global best.
    aspiration = (
        current_cut + pg
        > best_cut + 1e-12
    )

    allowed = (
        iteration >= tabu_until[v0]
        and iteration >= tabu_until[v1]
    )

    if allowed or aspiration:
        return v0, v1

    return None


# ============================================================
# Local search
# ============================================================

def local_search(
    g,
    x,
    gain,
    current_cut,
    tabu_until,
    iteration,
    rng,
    tabu_max
):
    """
    Steepest ascent using M1.

    Important:
    BLS does NOT use tabu restrictions to select local-search
    moves. Tabu history is updated, but diversification is used
    only after reaching a local optimum.
    """

    moves = 0

    while True:
        v = int(np.argmax(gain))
        best_gain = gain[v]

        if best_gain <= 1e-12:
            break

        delta = flip_vertex(
            g,
            x,
            gain,
            v
        )

        current_cut += delta

        gamma = rng.randint(
            3,
            max(3, tabu_max)
        )

        tabu_until[v] = iteration + gamma

        iteration += 1
        moves += 1

    return (
        x,
        gain,
        float(current_cut),
        iteration,
        moves
    )


# ============================================================
# Perturbation
# ============================================================

def perturb_a1(
    g,
    x,
    gain,
    current_cut,
    best_cut,
    tabu_until,
    iteration,
    L,
    rng,
    tabu_max
):
    """
    Directed perturbation A1:
    repeatedly apply best eligible M1 move.
    """

    for _ in range(L):
        v = best_m1(
            gain,
            tabu_until,
            iteration,
            current_cut,
            best_cut
        )

        delta = flip_vertex(
            g,
            x,
            gain,
            v
        )

        current_cut += delta

        gamma = rng.randint(
            3,
            max(3, tabu_max)
        )

        tabu_until[v] = iteration + gamma
        iteration += 1

    return x, gain, current_cut, iteration


def perturb_a2(
    g,
    x,
    gain,
    current_cut,
    best_cut,
    tabu_until,
    iteration,
    L,
    rng,
    tabu_max
):
    """
    Directed perturbation A2:
    repeatedly apply M2 moves.
    """

    n = len(x)

    for _ in range(L):
        pair = best_m2(
            g,
            x,
            gain,
            tabu_until,
            iteration,
            current_cut,
            best_cut
        )

        if pair is None:
            # Fallback to A1
            v = best_m1(
                gain,
                tabu_until,
                iteration,
                current_cut,
                best_cut
            )

            delta = flip_vertex(
                g,
                x,
                gain,
                v
            )

            current_cut += delta

            gamma = rng.randint(
                3,
                max(3, tabu_max)
            )

            tabu_until[v] = iteration + gamma
            iteration += 1
            continue

        v1, v2 = pair

        # Apply sequentially. Incremental gains automatically
        # produce the correct combined cut change.
        delta1 = flip_vertex(
            g,
            x,
            gain,
            v1
        )
        current_cut += delta1

        delta2 = flip_vertex(
            g,
            x,
            gain,
            v2
        )
        current_cut += delta2

        gamma1 = rng.randint(
            3,
            max(3, tabu_max)
        )

        gamma2 = rng.randint(
            3,
            max(3, tabu_max)
        )

        tabu_until[v1] = iteration + gamma1
        tabu_until[v2] = iteration + gamma2

        iteration += 1

    return x, gain, current_cut, iteration


def perturb_b(
    g,
    x,
    gain,
    current_cut,
    tabu_until,
    iteration,
    L,
    rng,
    tabu_max
):
    """
    Random perturbation B / M3.
    """

    n = len(x)

    for _ in range(L):
        v = rng.randrange(n)

        delta = flip_vertex(
            g,
            x,
            gain,
            v
        )

        current_cut += delta

        gamma = rng.randint(
            3,
            max(3, tabu_max)
        )

        tabu_until[v] = iteration + gamma
        iteration += 1

    return x, gain, current_cut, iteration


# ============================================================
# Breakout Local Search
# ============================================================

def bls(
    g,
    seed=1,
    time_limit=60.0,
    L0=None,
    T=1000,
    P0=0.8,
    Q=0.5
):
    """
    Breakout Local Search for Max-Cut following the algorithmic
    structure of Benlic & Hao (2013).

    Parameters:
        L0 = 0.01 |V|
        T  = 1000
        tabu tenure = rand[3, |V|/10]
        P0 = 0.8
        Q  = 0.5
    """

    rng = random.Random(seed)

    n = g.number_of_nodes()

    if L0 is None:
        L0 = max(
            1,
            int(round(0.01 * n))
        )

    tabu_max = max(
        3,
        int(n / 10)
    )

    # Initial random solution
    x = np.array(
        [
            rng.randint(0, 1)
            for _ in range(n)
        ],
        dtype=np.int8
    )

    current_cut = cut_value(g, x)
    gain = initial_gains(g, x)

    tabu_until = np.zeros(
        n,
        dtype=np.int64
    )

    iteration = 0

    # First descent
    (
        x,
        gain,
        current_cut,
        iteration,
        _
    ) = local_search(
        g,
        x,
        gain,
        current_cut,
        tabu_until,
        iteration,
        rng,
        tabu_max
    )

    best_x = x.copy()
    best_cut = float(current_cut)

    previous_local = x.copy()

    omega = 0
    L = L0

    local_optima = 1
    improvements = 0

    perturb_a1_count = 0
    perturb_a2_count = 0
    perturb_b_count = 0

    start = time.perf_counter()

    while (
        time.perf_counter() - start
        < time_limit
    ):
        # --------------------------------------------
        # Update best / stagnation
        # --------------------------------------------
        if current_cut > best_cut + 1e-12:
            best_cut = float(current_cut)
            best_x = x.copy()
            omega = 0
            improvements += 1
        else:
            omega += 1

        # --------------------------------------------
        # Strong stagnation detection
        # --------------------------------------------
        force_random = False

        if omega > T:
            omega = 0
            force_random = True

        # --------------------------------------------
        # Adaptive jump magnitude
        # --------------------------------------------
        if np.array_equal(
            x,
            previous_local
        ):
            L += 1
        else:
            L = L0

        previous_local = x.copy()

        # --------------------------------------------
        # Select perturbation type
        # --------------------------------------------
        if force_random:
            mode = "B"

        else:
            P = max(
                math.exp(
                    -float(omega) / float(T)
                ),
                P0
            )

            r = rng.random()

            if r < P * Q:
                mode = "A1"

            elif r < P:
                mode = "A2"

            else:
                mode = "B"

        # --------------------------------------------
        # Apply L perturbation moves
        # --------------------------------------------
        if mode == "A1":
            (
                x,
                gain,
                current_cut,
                iteration
            ) = perturb_a1(
                g,
                x,
                gain,
                current_cut,
                best_cut,
                tabu_until,
                iteration,
                L,
                rng,
                tabu_max
            )

            perturb_a1_count += 1

        elif mode == "A2":
            (
                x,
                gain,
                current_cut,
                iteration
            ) = perturb_a2(
                g,
                x,
                gain,
                current_cut,
                best_cut,
                tabu_until,
                iteration,
                L,
                rng,
                tabu_max
            )

            perturb_a2_count += 1

        else:
            (
                x,
                gain,
                current_cut,
                iteration
            ) = perturb_b(
                g,
                x,
                gain,
                current_cut,
                tabu_until,
                iteration,
                L,
                rng,
                tabu_max
            )

            perturb_b_count += 1

        # --------------------------------------------
        # Descent to next local optimum
        # --------------------------------------------
        (
            x,
            gain,
            current_cut,
            iteration,
            _
        ) = local_search(
            g,
            x,
            gain,
            current_cut,
            tabu_until,
            iteration,
            rng,
            tabu_max
        )

        local_optima += 1

    # Final best update
    if current_cut > best_cut + 1e-12:
        best_cut = float(current_cut)
        best_x = x.copy()
        improvements += 1

    runtime = (
        time.perf_counter() - start
    )

    return {
        "best_cut": float(best_cut),
        "best_x": best_x,
        "runtime_sec": runtime,
        "iterations": iteration,
        "local_optima": local_optima,
        "improvements": improvements,
        "A1": perturb_a1_count,
        "A2": perturb_a2_count,
        "B": perturb_b_count,
        "L0": L0
    }


# ============================================================
# Run graph
# ============================================================

def run_graph(args):
    print()
    print("========================================")
    print("BREAKOUT LOCAL SEARCH - MAX CUT")
    print("========================================")
    print("Graph:", args.graph)

    g, info = load_graph(args.graph)

    print()
    print(
        f"Nodes        : "
        f"{g.number_of_nodes()}"
    )
    print(
        f"Edges        : "
        f"{g.number_of_edges()}"
    )

    wsum = total_edge_weight(g)

    print(
        f"Total weight : "
        f"{wsum:.1f}"
    )

    result = bls(
        g,
        seed=args.seed,
        time_limit=args.time_limit,
        L0=args.L0,
        T=args.T,
        P0=args.P0,
        Q=args.Q
    )

    best_cut = result["best_cut"]

    verified = cut_value(
        g,
        result["best_x"]
    )

    assert abs(
        best_cut - verified
    ) < 1e-8, (
        f"BUG: tracked={best_cut}, "
        f"verified={verified}"
    )

    ratio = best_cut / wsum

    print()
    print(
        f"Tracked cut  : "
        f"{best_cut:.1f}"
    )
    print(
        f"Verified cut : "
        f"{verified:.1f}"
    )
    print(
        f"BLS cut      : "
        f"{best_cut:.1f}"
    )
    print(
        f"BLS ratio    : "
        f"{ratio * 100:.2f}%"
    )

    print()
    print(
        f"Runtime      : "
        f"{result['runtime_sec']:.3f} sec"
    )
    print(
        f"Iterations   : "
        f"{result['iterations']}"
    )
    print(
        f"Local optima : "
        f"{result['local_optima']}"
    )
    print(
        f"Improvements : "
        f"{result['improvements']}"
    )

    print()
    print(
        f"L0           : "
        f"{result['L0']}"
    )
    print(
        f"A1 perturb   : "
        f"{result['A1']}"
    )
    print(
        f"A2 perturb   : "
        f"{result['A2']}"
    )
    print(
        f"B perturb    : "
        f"{result['B']}"
    )

    print("========================================")
    print()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--graph",
        type=str,
        required=True
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1
    )

    parser.add_argument(
        "--time_limit",
        type=float,
        default=60.0
    )

    parser.add_argument(
        "--L0",
        type=int,
        default=None
    )

    parser.add_argument(
        "--T",
        type=int,
        default=1000
    )

    parser.add_argument(
        "--P0",
        type=float,
        default=0.8
    )

    parser.add_argument(
        "--Q",
        type=float,
        default=0.5
    )

    args = parser.parse_args()

    run_graph(args)


if __name__ == "__main__":
    main()
