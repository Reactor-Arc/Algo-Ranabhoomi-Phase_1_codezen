"""Optimized Autoclave Queue solver.

Strategy:
1. Build an earliest-due-date starting solution.
2. Improve it using insertion moves:
       remove one batch and insert it at another position.
3. Use several deterministic/seeded perturbations to escape local minima.
4. Submit the best solution as improvements are found.

The evaluator interface is unchanged.
"""

from adapter import Solver, cost_of
from benchkit.rng import Rng, derive_seed

import time


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

# Number of perturbation/restart attempts.
MAX_RESTARTS = 12

# Do not spend the entire time budget calculating one final move.
SAFETY_S = 0.12

# Maximum number of complete insertion-search passes per solution.
MAX_INSERTION_PASSES = 6

# Number of random perturbations used to create a new starting order.
PERTURB_MOVES_FACTOR = 0.18


# ------------------------------------------------------------
# Basic helpers
# ------------------------------------------------------------

def edd_order(instance):
    """Earliest Due Date order."""
    return sorted(
        range(instance.size),
        key=lambda b: (instance.due[b], b)
    )


def urgency_order(instance):
    """
    Alternative starting order.

    Combines due date, weight and processing time.
    This is only used as a secondary restart; exact cost_of()
    still decides whether the resulting solution is better.
    """

    def key(b):
        # Earlier due dates first.
        # Higher weights are slightly more urgent.
        # Longer jobs are given a small priority because they
        # can create more downstream tardiness.
        return (
            instance.due[b],
            -instance.weight[b],
            instance.proc[b],
            b,
        )

    return sorted(range(instance.size), key=key)


def release_due_order(instance):
    """
    Release-aware starting order.

    The machine cannot skip an unreleased batch, so using release
    time as a secondary signal can avoid some unnecessary idle time.
    """

    return sorted(
        range(instance.size),
        key=lambda b: (
            instance.due[b],
            instance.release[b],
            b
        )
    )


def family_aware_order(instance):
    """Group batches by family and rank each family by urgency."""
    families = {}
    for b in range(instance.size):
        families.setdefault(instance.fam[b], []).append(b)

    family_priority = {}
    for fam, members in families.items():
        family_priority[fam] = (
            min(instance.due[b] for b in members),
            -max(instance.weight[b] for b in members),
            -sum(instance.proc[b] for b in members),
            fam,
        )

    order = []
    for fam in sorted(families, key=lambda f: family_priority[f]):
        order.extend(sorted(
            families[fam],
            key=lambda b: (
                instance.due[b],
                -instance.weight[b],
                instance.release[b],
                instance.proc[b],
                b,
            )
        ))
    return order


def slack_order(instance):
    """Prioritise jobs with little slack and high urgency."""
    return sorted(
        range(instance.size),
        key=lambda b: (
            instance.due[b] - instance.release[b] - instance.proc[b],
            instance.due[b],
            -instance.weight[b],
            b,
        )
    )


def random_perturb(order, rng):
    """
    Make a small perturbation of an existing solution.

    We use insertion-style perturbations instead of only swaps,
    because an insertion can move a batch across many positions
    in one operation.
    """

    order = list(order)
    n = len(order)

    if n < 2:
        return order

    moves = max(2, int(n * PERTURB_MOVES_FACTOR))

    for _ in range(moves):
        i = rng.below(n)

        # Short and medium distance moves are usually safer than
        # completely random permutations.
        distance = rng.between(1, max(2, min(n - 1, 10)))

        if rng.below(2) == 0:
            j = max(0, i - distance)
        else:
            j = min(n - 1, i + distance)

        if i == j:
            continue

        item = order.pop(i)
        order.insert(j, item)

    return order


# ------------------------------------------------------------
# Insertion local search
# ------------------------------------------------------------

def insertion_descent(instance, initial_order, deadline):
    """
    Best-insertion descent.

    For each batch, try every possible insertion location and keep the single
    best improving move. This is slower than a greedy first-improvement pass,
    but it finds substantially better local optima on the larger shifted
    instances that dominate the overall score.
    """

    order = list(initial_order)
    n = len(order)

    if n <= 1:
        return order, cost_of(instance, order)

    current_cost = cost_of(instance, order)

    for _pass in range(MAX_INSERTION_PASSES):
        if time.monotonic() >= deadline:
            break

        best_move_cost = current_cost
        best_move = None

        positions = sorted(
            range(n),
            key=lambda i: (
                instance.due[order[i]],
                -instance.weight[order[i]],
                instance.release[order[i]],
                i,
            )
        )

        for original_pos in positions:
            if time.monotonic() >= deadline:
                return order, current_cost

            batch = order[original_pos]
            reduced = order[:original_pos] + order[original_pos + 1:]

            for insert_pos in range(len(reduced) + 1):
                if time.monotonic() >= deadline:
                    return order, current_cost

                candidate = reduced[:insert_pos] + [batch] + reduced[insert_pos:]
                candidate_cost = cost_of(instance, candidate)

                if candidate_cost < best_move_cost:
                    best_move_cost = candidate_cost
                    best_move = (original_pos, insert_pos, candidate)

        if best_move is None:
            break

        _, _, order = best_move
        current_cost = best_move_cost

    return order, current_cost


# ------------------------------------------------------------
# More focused insertion search
# ------------------------------------------------------------

def swap_descent(instance, initial_order, deadline):
    """First-improvement pairwise swap descent."""
    order = list(initial_order)
    n = len(order)

    if n <= 1:
        return order, cost_of(instance, order)

    current_cost = cost_of(instance, order)

    while time.monotonic() < deadline:
        improved = False

        for i in range(n - 1):
            if time.monotonic() >= deadline:
                return order, current_cost

            for j in range(i + 1, n):
                if time.monotonic() >= deadline:
                    return order, current_cost

                order[i], order[j] = order[j], order[i]
                candidate_cost = cost_of(instance, order)

                if candidate_cost < current_cost:
                    current_cost = candidate_cost
                    improved = True
                    break

                order[i], order[j] = order[j], order[i]

            if improved:
                break

        if not improved:
            break

    return order, current_cost


def targeted_insertion_search(instance, initial_order, deadline):
    """
    Stronger second-stage search.

    Focuses on batches that are likely to have a large effect:
      - high weight
      - early due date
      - long processing time

    Instead of moving every batch everywhere, we try the most
    important batches first.
    """

    order = list(initial_order)
    n = len(order)

    if n <= 1:
        return order, cost_of(instance, order)

    current_cost = cost_of(instance, order)

    # Rank batches by likely contribution to tardiness.
    important = sorted(
        range(n),
        key=lambda i: (
            -instance.weight[order[i]],
            instance.due[order[i]],
            -instance.proc[order[i]],
        )
    )

    # Limit this stage so it cannot consume the entire budget.
    max_positions_to_try = min(n, 30)

    for idx in important:

        if time.monotonic() >= deadline:
            break

        if idx >= len(order):
            continue

        batch = order[idx]

        reduced = order[:idx] + order[idx + 1:]

        # Candidate positions:
        # start/end and evenly distributed positions.
        candidates = {
            0,
            len(reduced),
        }

        if len(reduced) > 1:
            step = max(1, len(reduced) // max_positions_to_try)

            for p in range(0, len(reduced) + 1, step):
                candidates.add(p)

            # Also try positions near the current location.
            for delta in range(-8, 9):
                p = idx + delta

                if 0 <= p <= len(reduced):
                    candidates.add(p)

        for p in sorted(candidates):

            if time.monotonic() >= deadline:
                return order, current_cost

            candidate = reduced[:p] + [batch] + reduced[p:]

            candidate_cost = cost_of(instance, candidate)

            if candidate_cost < current_cost:
                order = candidate
                current_cost = candidate_cost

                # Restart because positions changed.
                break

    return order, current_cost


# ------------------------------------------------------------
# Solver
# ------------------------------------------------------------

class MySolver(Solver):

    def solve(self, instance, submit_candidate):

        n = instance.size

        if n == 0:
            return {"order": []}

        # ----------------------------------------------------
        # Deterministic RNG
        # ----------------------------------------------------

        rng = Rng(
            derive_seed(
                "autoclave-mine",
                instance.digest
            )
        )

        # ----------------------------------------------------
        # Initial candidate
        # ----------------------------------------------------

        start_time = time.monotonic()

        # Start from several strong, deterministic orderings and keep the best one.
        candidates = [
            edd_order(instance),
            urgency_order(instance),
            release_due_order(instance),
            family_aware_order(instance),
            slack_order(instance),
        ]

        best_order = min(candidates, key=lambda order: cost_of(instance, order))
        best_cost = cost_of(instance, best_order)

        receipt = submit_candidate({
            "order": best_order
        })

        # Use the evaluator's remaining time if available.
        remaining = receipt.get("remaining_s", 5.0)

        deadline = (
            start_time
            + max(0.0, remaining - SAFETY_S)
        )

        # ----------------------------------------------------
        # First insertion descent from EDD
        # ----------------------------------------------------

        if time.monotonic() < deadline:

            order, cost = insertion_descent(
                instance,
                best_order,
                deadline
            )
            if time.monotonic() < deadline:
                order, cost = swap_descent(instance, order, deadline)

            if cost < best_cost:

                best_order = order
                best_cost = cost

                receipt = submit_candidate({
                    "order": best_order
                })

                remaining = receipt.get(
                    "remaining_s",
                    max(0.0, deadline - time.monotonic())
                )

                deadline = time.monotonic() + max(
                    0.0,
                    remaining - SAFETY_S
                )

        # ----------------------------------------------------
        # Targeted improvement
        # ----------------------------------------------------

        if time.monotonic() < deadline:

            order, cost = targeted_insertion_search(
                instance,
                best_order,
                deadline
            )
            if time.monotonic() < deadline:
                order, cost = swap_descent(instance, order, deadline)

            if cost < best_cost:

                best_order = order
                best_cost = cost

                receipt = submit_candidate({
                    "order": best_order
                })

                remaining = receipt.get(
                    "remaining_s",
                    max(0.0, deadline - time.monotonic())
                )

                deadline = time.monotonic() + max(
                    0.0,
                    remaining - SAFETY_S
                )

        # ----------------------------------------------------
        # Restart search
        # ----------------------------------------------------

        restart = 0

        while (
            restart < MAX_RESTARTS
            and time.monotonic() < deadline
        ):

            # Different restart sources.
            if restart == 0:
                start_order = urgency_order(instance)

            elif restart == 1:
                start_order = release_due_order(instance)

            else:
                start_order = random_perturb(
                    best_order,
                    rng
                )

            # Perturb deterministic alternatives as well.
            if restart >= 2:
                start_order = random_perturb(
                    start_order,
                    rng
                )

            order, cost = insertion_descent(
                instance,
                start_order,
                deadline
            )
            if time.monotonic() < deadline:
                order, cost = swap_descent(instance, order, deadline)

            # Targeted improvement after insertion descent.
            if time.monotonic() < deadline:

                order, cost = targeted_insertion_search(
                    instance,
                    order,
                    deadline
                )
                if time.monotonic() < deadline:
                    order, cost = swap_descent(instance, order, deadline)

            if cost < best_cost:

                best_order = order
                best_cost = cost

                receipt = submit_candidate({
                    "order": best_order
                })

                remaining = receipt.get(
                    "remaining_s",
                    max(0.0, deadline - time.monotonic())
                )

                # Recalculate deadline from the remaining evaluator
                # time after every accepted submission.
                deadline = time.monotonic() + max(
                    0.0,
                    remaining - SAFETY_S
                )

            restart += 1

        # ----------------------------------------------------
        # Final valid candidate
        # ----------------------------------------------------

        return {
            "order": best_order
        }