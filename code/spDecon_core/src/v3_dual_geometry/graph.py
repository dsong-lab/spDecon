"""Coordinate-only graphs, local non-edges, and spatial null controls.

Every operation is section-safe and deterministic under ``random_state``.
The primary graph is the undirected union of directed 6-nearest-neighbor
queries, with Zelnik-Manor/Perona self-tuning edge affinities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, overload

import numpy as np
from scipy.spatial import cKDTree


EPS = 1e-12


@dataclass
class SpatialGraph:
    """A simple undirected graph stored once per canonical edge ``i < j``."""

    n_spots: int
    edges: np.ndarray
    distances: np.ndarray
    weights: np.ndarray
    sections: np.ndarray
    diagnostics: dict[str, object]


def _validate_coords(coords: np.ndarray) -> np.ndarray:
    values = np.asarray(coords, dtype=float)
    if (
        values.ndim != 2
        or values.shape[0] < 2
        or values.shape[1] < 2
        or np.any(~np.isfinite(values))
    ):
        raise ValueError(
            "coords must contain at least two rows and two finite columns"
        )
    return values


def _is_missing_label(label: object) -> bool:
    if label is None:
        return True
    try:
        unequal = label != label
        return isinstance(unequal, (bool, np.bool_)) and bool(unequal)
    except Exception:
        return False


def _section_groups(
    section_labels: Sequence[object] | np.ndarray | None,
    n_spots: int,
) -> tuple[np.ndarray, list[np.ndarray], list[str]]:
    if section_labels is None:
        return (
            np.zeros(n_spots, dtype=int),
            [np.arange(n_spots, dtype=int)],
            ["all"],
        )
    labels = np.asarray(section_labels, dtype=object)
    if labels.ndim != 1 or labels.shape[0] != n_spots:
        raise ValueError("section_labels must contain one label per spot")

    key_to_code: dict[tuple[str, str], int] = {}
    members: list[list[int]] = []
    names: list[str] = []
    codes = np.empty(n_spots, dtype=int)
    for index, label in enumerate(labels.tolist()):
        if _is_missing_label(label):
            raise ValueError("section_labels must not contain missing values")
        # Type is part of the key: integer 1 and string "1" are different
        # sections even though their display strings coincide.
        key = (type(label).__name__, repr(label))
        if key not in key_to_code:
            key_to_code[key] = len(members)
            members.append([])
            names.append(str(label))
        code = key_to_code[key]
        codes[index] = code
        members[code].append(index)
    return (
        codes,
        [np.asarray(indices, dtype=int) for indices in members],
        names,
    )


def _canonical_edges(
    edges: np.ndarray,
    *,
    n_spots: int | None = None,
    name: str = "edges",
) -> np.ndarray:
    raw = np.asarray(edges)
    if raw.size == 0:
        return np.empty((0, 2), dtype=int)
    if raw.ndim != 2 or raw.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n_edges, 2)")
    try:
        numeric = np.asarray(raw, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain integer spot indices") from error
    if np.any(~np.isfinite(numeric)) or np.any(numeric != np.floor(numeric)):
        raise ValueError(f"{name} must contain integer spot indices")
    result = np.sort(numeric.astype(int), axis=1)
    if np.any(result[:, 0] == result[:, 1]):
        raise ValueError(f"{name} must not contain self-loops")
    if n_spots is not None and (
        np.any(result < 0) or np.any(result >= int(n_spots))
    ):
        raise ValueError(f"{name} contains an out-of-range spot index")
    order = np.lexsort((result[:, 1], result[:, 0]))
    result = result[order]
    if result.shape[0] > 1 and np.any(np.all(result[1:] == result[:-1], axis=1)):
        raise ValueError(f"{name} must not contain duplicate undirected pairs")
    return result


def _edge_summary(values: np.ndarray) -> dict[str, float | None]:
    return {
        "min": float(np.min(values)) if values.size else None,
        "median": float(np.median(values)) if values.size else None,
        "max": float(np.max(values)) if values.size else None,
    }


def build_spatial_graph(
    coords: np.ndarray,
    *,
    section_labels: Sequence[object] | np.ndarray | None = None,
    n_neighbors: int = 6,
    local_scale_neighbor_rank: int = 7,
) -> SpatialGraph:
    """Build a section-wise symmetric kNN graph with self-tuning weights.

    The edge set is the union of directed kNN queries.  For edge ``(i, j)``,
    ``weight = exp(-distance**2 / (sigma_i * sigma_j))``, where ``sigma_i``
    is the distance to the requested local-scale neighbor (rank seven by
    default, or the furthest available neighbor in a small section).
    """

    coords = _validate_coords(coords)
    n_spots = int(coords.shape[0])
    if int(n_neighbors) != n_neighbors or n_neighbors < 1:
        raise ValueError("n_neighbors must be a positive integer")
    if (
        int(local_scale_neighbor_rank) != local_scale_neighbor_rank
        or local_scale_neighbor_rank < 1
    ):
        raise ValueError("local_scale_neighbor_rank must be a positive integer")
    n_neighbors = int(n_neighbors)
    local_scale_neighbor_rank = int(local_scale_neighbor_rank)
    section_codes, groups, names = _section_groups(section_labels, n_spots)

    edge_to_distance: dict[tuple[int, int], float] = {}
    local_scales = np.full(n_spots, np.nan, dtype=float)
    effective_neighbors: dict[str, int] = {}
    effective_scale_ranks: dict[str, int] = {}

    for section_number, (indices, name) in enumerate(zip(groups, names)):
        diagnostic_name = f"{section_number}:{name}"
        section_size = int(indices.size)
        if section_size < 2:
            effective_neighbors[diagnostic_name] = 0
            effective_scale_ranks[diagnostic_name] = 0
            local_scales[indices] = 1.0
            continue
        edge_k = min(n_neighbors, section_size - 1)
        scale_rank = min(local_scale_neighbor_rank, section_size - 1)
        effective_neighbors[diagnostic_name] = edge_k
        effective_scale_ranks[diagnostic_name] = scale_rank
        query_k = min(section_size, max(edge_k, scale_rank) + 1)
        local_coords = coords[indices, :2]
        distances, neighbors = cKDTree(local_coords).query(
            local_coords, k=query_k
        )
        distances = np.asarray(distances, dtype=float)
        neighbors = np.asarray(neighbors, dtype=int)
        if distances.ndim == 1:
            distances = distances[:, None]
            neighbors = neighbors[:, None]

        candidate_rows: list[list[tuple[float, int]]] = []
        positive_distances: list[float] = []
        for local_i in range(section_size):
            candidates = [
                (float(distance), int(local_j))
                for distance, local_j in zip(
                    distances[local_i].tolist(), neighbors[local_i].tolist()
                )
                if int(local_j) != local_i and np.isfinite(distance)
            ]
            candidates.sort(key=lambda item: (item[0], item[1]))
            candidate_rows.append(candidates)
            positive_distances.extend(
                distance for distance, _ in candidates if distance > EPS
            )

        fallback_scale = (
            float(np.median(positive_distances))
            if positive_distances
            else 1.0
        )
        minimum_scale = max(fallback_scale * 1e-6, EPS)
        for local_i, global_i in enumerate(indices.tolist()):
            candidates = candidate_rows[local_i]
            if not candidates:
                local_scales[global_i] = fallback_scale
                continue
            rank_index = min(scale_rank, len(candidates)) - 1
            local_scales[global_i] = max(
                float(candidates[rank_index][0]), minimum_scale
            )
            for distance, local_j in candidates[:edge_k]:
                global_j = int(indices[local_j])
                edge = (min(global_i, global_j), max(global_i, global_j))
                previous = edge_to_distance.get(edge)
                if previous is None or distance < previous:
                    edge_to_distance[edge] = float(distance)

    invalid_scales = ~np.isfinite(local_scales) | (local_scales <= 0.0)
    if np.any(invalid_scales):
        valid = local_scales[~invalid_scales]
        fallback = float(np.median(valid)) if valid.size else 1.0
        local_scales[invalid_scales] = max(fallback, EPS)

    edge_tuples = sorted(edge_to_distance)
    edges = (
        np.asarray(edge_tuples, dtype=int).reshape(-1, 2)
        if edge_tuples
        else np.empty((0, 2), dtype=int)
    )
    edge_distances = np.asarray(
        [edge_to_distance[edge] for edge in edge_tuples], dtype=float
    )
    if edges.size:
        denominators = np.maximum(
            local_scales[edges[:, 0]] * local_scales[edges[:, 1]], EPS
        )
        weights = np.exp(-np.square(edge_distances) / denominators)
    else:
        weights = np.empty(0, dtype=float)

    degrees = (
        np.bincount(edges.ravel(), minlength=n_spots)
        if edges.size
        else np.zeros(n_spots, dtype=int)
    )
    distance_summary = _edge_summary(edge_distances)
    weight_summary = _edge_summary(weights)
    diagnostics: dict[str, object] = {
        "construction": "sectionwise_symmetric_coordinate_knn_self_tuning",
        "n_spots": n_spots,
        "n_sections": len(groups),
        "section_sizes": [int(group.size) for group in groups],
        "section_names": names,
        "n_neighbors": n_neighbors,
        "effective_neighbors_by_section": effective_neighbors,
        "local_scale_neighbor_rank": local_scale_neighbor_rank,
        "effective_scale_rank_by_section": effective_scale_ranks,
        "local_scale_min": float(np.min(local_scales)),
        "local_scale_median": float(np.median(local_scales)),
        "local_scale_max": float(np.max(local_scales)),
        "n_edges": int(edges.shape[0]),
        "n_isolated_spots": int(np.sum(degrees == 0)),
        "degree_mean": float(np.mean(degrees)),
        "degree_max": int(np.max(degrees, initial=0)),
        "distance_min": distance_summary["min"],
        "distance_median": distance_summary["median"],
        "distance_max": distance_summary["max"],
        "weight_min": weight_summary["min"],
        "weight_median": weight_summary["median"],
        "weight_max": weight_summary["max"],
        "symmetrization": "undirected_union",
        "cross_section_edges": 0,
    }
    return SpatialGraph(
        n_spots=n_spots,
        edges=edges,
        distances=edge_distances,
        weights=weights,
        sections=section_codes,
        diagnostics=diagnostics,
    )


def _validate_graph(graph: SpatialGraph) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(graph, SpatialGraph):
        raise TypeError("graph must be a SpatialGraph")
    if int(graph.n_spots) != graph.n_spots or graph.n_spots < 2:
        raise ValueError("graph.n_spots must be an integer of at least two")
    edges = _canonical_edges(graph.edges, n_spots=graph.n_spots)
    distances = np.asarray(graph.distances, dtype=float)
    weights = np.asarray(graph.weights, dtype=float)
    if distances.shape != (edges.shape[0],) or np.any(~np.isfinite(distances)):
        raise ValueError("graph.distances must contain one finite value per edge")
    if np.any(distances < 0.0):
        raise ValueError("graph.distances must be nonnegative")
    if weights.shape != (edges.shape[0],) or np.any(~np.isfinite(weights)):
        raise ValueError("graph.weights must contain one finite value per edge")
    sections = np.asarray(graph.sections)
    if sections.ndim != 1 or sections.shape[0] != graph.n_spots:
        raise ValueError("graph.sections must contain one section code per spot")
    if edges.size and np.any(sections[edges[:, 0]] != sections[edges[:, 1]]):
        raise ValueError("graph contains a cross-section edge")
    return edges, sections


@overload
def sample_local_negative_edges(
    graph: SpatialGraph,
    *,
    ratio: float = 1.0,
    local_radius_multiplier: float = 3.0,
    random_state: int = 0,
    return_diagnostics: bool = False,
) -> np.ndarray: ...


@overload
def sample_local_negative_edges(
    graph: SpatialGraph,
    *,
    ratio: float = 1.0,
    local_radius_multiplier: float = 3.0,
    random_state: int = 0,
    return_diagnostics: bool = True,
) -> tuple[np.ndarray, dict[str, object]]: ...


def sample_local_negative_edges(
    graph: SpatialGraph,
    *,
    ratio: float = 1.0,
    local_radius_multiplier: float = 3.0,
    random_state: int = 0,
    return_diagnostics: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Sample local two-hop non-edges, with a 1:1 default class balance.

    Candidates whose shortest observed two-edge path is no longer than
    ``local_radius_multiplier`` times the section's median neighbor distance
    are preferred.  If that strict pool is too small, remaining two-hop
    non-edges are used; arbitrary global/distant pairs are never introduced.
    """

    edges, sections = _validate_graph(graph)
    ratio = float(ratio)
    local_radius_multiplier = float(local_radius_multiplier)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("ratio must be finite and positive")
    if (
        not np.isfinite(local_radius_multiplier)
        or local_radius_multiplier <= 1.0
    ):
        raise ValueError("local_radius_multiplier must be finite and above one")
    if int(random_state) != random_state or random_state < 0:
        raise ValueError("random_state must be a nonnegative integer")
    target = int(round(ratio * edges.shape[0]))
    if edges.shape[0] and target < 1:
        target = 1
    if target == 0:
        result = np.empty((0, 2), dtype=int)
        details: dict[str, object] = {
            "n_positive_edges": 0,
            "n_negative_edges": 0,
            "negative_ratio": ratio,
            "candidate_pool": "two_hop_local_non_edges",
        }
        return (result, details) if return_diagnostics else result

    adjacency: list[set[int]] = [set() for _ in range(graph.n_spots)]
    edge_lengths: dict[tuple[int, int], float] = {}
    for edge, distance in zip(edges.tolist(), np.asarray(graph.distances).tolist()):
        i, j = int(edge[0]), int(edge[1])
        adjacency[i].add(j)
        adjacency[j].add(i)
        edge_lengths[(i, j)] = float(distance)

    section_medians: dict[int, float] = {}
    for section in np.unique(sections):
        mask = sections[edges[:, 0]] == section
        section_distances = np.asarray(graph.distances, dtype=float)[mask]
        positive = section_distances[section_distances > EPS]
        section_medians[int(section)] = (
            float(np.median(positive)) if positive.size else 1.0
        )

    path_lengths: dict[tuple[int, int], float] = {}
    for i in range(graph.n_spots):
        for middle in adjacency[i]:
            first = (min(i, middle), max(i, middle))
            first_distance = edge_lengths[first]
            for j in adjacency[middle]:
                if j <= i or j == i or j in adjacency[i]:
                    continue
                if sections[i] != sections[j]:
                    continue
                second = (min(middle, j), max(middle, j))
                candidate = (i, j)
                length = first_distance + edge_lengths[second]
                previous = path_lengths.get(candidate)
                if previous is None or length < previous:
                    path_lengths[candidate] = length

    all_candidates = sorted(path_lengths)
    if len(all_candidates) < target:
        raise ValueError(
            "not enough section-safe two-hop non-edges for the requested "
            f"negative ratio: need {target}, found {len(all_candidates)}"
        )
    preferred = [
        edge
        for edge in all_candidates
        if path_lengths[edge]
        <= local_radius_multiplier * section_medians[int(sections[edge[0]])] + EPS
    ]
    rng = np.random.default_rng(int(random_state))
    if len(preferred) >= target:
        pool = preferred
        pool_kind = "radius_matched_two_hop_non_edges"
    else:
        pool = all_candidates
        pool_kind = "two_hop_non_edges"
    chosen_indices = rng.choice(len(pool), size=target, replace=False)
    selected = np.asarray([pool[int(index)] for index in chosen_indices], dtype=int)
    order = np.lexsort((selected[:, 1], selected[:, 0]))
    selected = selected[order]
    details = {
        "n_positive_edges": int(edges.shape[0]),
        "n_negative_edges": int(selected.shape[0]),
        "negative_ratio": ratio,
        "n_two_hop_candidates": int(len(all_candidates)),
        "n_radius_matched_candidates": int(len(preferred)),
        "local_radius_multiplier": local_radius_multiplier,
        "candidate_pool": pool_kind,
        "random_state": int(random_state),
        "cross_section_edges": 0,
    }
    return (selected, details) if return_diagnostics else selected


def permute_coordinates_within_sections(
    coords: np.ndarray,
    *,
    section_labels: Sequence[object] | np.ndarray | None = None,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute coordinate assignments independently inside each section."""

    coords = _validate_coords(coords)
    if int(random_state) != random_state or random_state < 0:
        raise ValueError("random_state must be a nonnegative integer")
    _, groups, _ = _section_groups(section_labels, coords.shape[0])
    rng = np.random.default_rng(int(random_state))
    source_indices = np.arange(coords.shape[0], dtype=int)
    for indices in groups:
        if indices.size < 2:
            continue
        sources = rng.permutation(indices)
        # Avoid the degenerate identity null for a whole non-singleton section.
        if np.array_equal(sources, indices):
            sources = np.roll(sources, 1)
        source_indices[indices] = sources
    return coords[source_indices].copy(), source_indices


def permuted_coordinate_graph(
    coords: np.ndarray,
    *,
    section_labels: Sequence[object] | np.ndarray | None = None,
    n_neighbors: int = 6,
    local_scale_neighbor_rank: int = 7,
    random_state: int = 0,
) -> SpatialGraph:
    """Build the coordinate-permutation spatial null graph."""

    permuted, source_indices = permute_coordinates_within_sections(
        coords,
        section_labels=section_labels,
        random_state=random_state,
    )
    graph = build_spatial_graph(
        permuted,
        section_labels=section_labels,
        n_neighbors=n_neighbors,
        local_scale_neighbor_rank=local_scale_neighbor_rank,
    )
    graph.diagnostics.update(
        {
            "null_model": "coordinate_permutation",
            "random_state": int(random_state),
            "coordinate_permutation_moved": int(
                np.sum(source_indices != np.arange(source_indices.size))
            ),
            "coordinate_source_indices": source_indices.tolist(),
        }
    )
    return graph


def _allocate_integer(total: int, sizes: list[int]) -> list[int]:
    if total == 0 or sum(sizes) == 0:
        return [0 for _ in sizes]
    raw = np.asarray(sizes, dtype=float) * (float(total) / float(sum(sizes)))
    allocated = np.floor(raw).astype(int)
    remainder = int(total - allocated.sum())
    priorities = np.argsort(-(raw - allocated), kind="stable")
    for index in priorities[:remainder]:
        allocated[int(index)] += 1
    return allocated.tolist()


def degree_preserving_randomized_graph(
    graph: SpatialGraph,
    *,
    random_state: int = 0,
    n_swaps: int | None = None,
    max_attempts: int | None = None,
) -> SpatialGraph:
    """Apply section-wise simple-graph double-edge swaps.

    Node degrees, section membership, edge count, and the empirical
    distance/weight distributions are preserved exactly.  Connectivity is
    intentionally not constrained because this is a topology null.
    """

    edges, sections = _validate_graph(graph)
    if int(random_state) != random_state or random_state < 0:
        raise ValueError("random_state must be a nonnegative integer")
    if n_swaps is not None and (int(n_swaps) != n_swaps or n_swaps < 0):
        raise ValueError("n_swaps must be a nonnegative integer or None")
    if max_attempts is not None and (
        int(max_attempts) != max_attempts or max_attempts < 1
    ):
        raise ValueError("max_attempts must be a positive integer or None")

    section_values = np.unique(sections)
    section_edge_lists: list[list[tuple[int, int]]] = []
    for section in section_values:
        mask = sections[edges[:, 0]] == section
        section_edge_lists.append(
            [tuple(map(int, edge)) for edge in edges[mask].tolist()]
        )
    sizes = [len(section_edges) for section_edges in section_edge_lists]
    total_requested = (
        10 * int(edges.shape[0]) if n_swaps is None else int(n_swaps)
    )
    targets = _allocate_integer(total_requested, sizes)
    if max_attempts is None:
        attempt_limits = [max(100, 30 * target) if target else 0 for target in targets]
    else:
        attempt_limits = _allocate_integer(int(max_attempts), sizes)

    rng = np.random.default_rng(int(random_state))
    completed_by_section: list[int] = []
    attempts_by_section: list[int] = []
    randomized_sections: list[list[tuple[int, int]]] = []
    for edge_list, target, attempt_limit in zip(
        section_edge_lists, targets, attempt_limits
    ):
        edge_set = set(edge_list)
        completed = 0
        attempts = 0
        while completed < target and attempts < attempt_limit and len(edge_list) >= 2:
            attempts += 1
            first_index, second_index = rng.choice(
                len(edge_list), size=2, replace=False
            ).tolist()
            a, b = edge_list[int(first_index)]
            c, d = edge_list[int(second_index)]
            if len({a, b, c, d}) < 4:
                continue
            if int(rng.integers(2)) == 0:
                proposed = ((a, c), (b, d))
            else:
                proposed = ((a, d), (b, c))
            new_first = tuple(sorted(proposed[0]))
            new_second = tuple(sorted(proposed[1]))
            if new_first == new_second:
                continue
            old_first = edge_list[int(first_index)]
            old_second = edge_list[int(second_index)]
            edge_set.remove(old_first)
            edge_set.remove(old_second)
            if new_first in edge_set or new_second in edge_set:
                edge_set.add(old_first)
                edge_set.add(old_second)
                continue
            edge_set.add(new_first)
            edge_set.add(new_second)
            edge_list[int(first_index)] = new_first
            edge_list[int(second_index)] = new_second
            completed += 1
        completed_by_section.append(completed)
        attempts_by_section.append(attempts)
        randomized_sections.append(edge_list)

    randomized_tuples = sorted(
        edge for section_edges in randomized_sections for edge in section_edges
    )
    randomized_edges = (
        np.asarray(randomized_tuples, dtype=int).reshape(-1, 2)
        if randomized_tuples
        else np.empty((0, 2), dtype=int)
    )
    attribute_order = rng.permutation(edges.shape[0])
    distances = np.asarray(graph.distances, dtype=float)[attribute_order].copy()
    weights = np.asarray(graph.weights, dtype=float)[attribute_order].copy()
    old_set = set(map(tuple, edges.tolist()))
    new_set = set(randomized_tuples)
    diagnostics = dict(graph.diagnostics)
    diagnostics.update(
        {
            "null_model": "degree_preserving",
            "random_state": int(random_state),
            "requested_swaps": int(total_requested),
            "completed_swaps": int(sum(completed_by_section)),
            "attempted_swaps": int(sum(attempts_by_section)),
            "completed_swaps_by_section": completed_by_section,
            "attempted_swaps_by_section": attempts_by_section,
            "edge_change_fraction": (
                float(len(old_set.symmetric_difference(new_set)) / (2 * len(old_set)))
                if old_set
                else 0.0
            ),
            "degree_preserved": True,
            "attribute_distributions_preserved": True,
            "cross_section_edges": 0,
        }
    )
    return SpatialGraph(
        n_spots=graph.n_spots,
        edges=randomized_edges,
        distances=distances,
        weights=weights,
        sections=np.asarray(graph.sections).copy(),
        diagnostics=diagnostics,
    )


def permute_edge_labels(
    positive_edges: np.ndarray,
    negative_edges: np.ndarray,
    *,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Permute binary labels over a fixed positive/negative pair pool."""

    positive = _canonical_edges(positive_edges, name="positive_edges")
    negative = _canonical_edges(negative_edges, name="negative_edges")
    positive_set = set(map(tuple, positive.tolist()))
    negative_set = set(map(tuple, negative.tolist()))
    if positive_set.intersection(negative_set):
        raise ValueError("positive_edges and negative_edges must be disjoint")
    if int(random_state) != random_state or random_state < 0:
        raise ValueError("random_state must be a nonnegative integer")
    pool = np.vstack([positive, negative])
    if pool.shape[0] == 0:
        return positive.copy(), negative.copy()
    labels = np.concatenate(
        [np.ones(positive.shape[0], dtype=bool), np.zeros(negative.shape[0], dtype=bool)]
    )
    rng = np.random.default_rng(int(random_state))
    shuffled = rng.permutation(labels)
    if positive.size and negative.size and np.array_equal(shuffled, labels):
        shuffled = np.roll(shuffled, 1)
    null_positive = pool[shuffled]
    null_negative = pool[~shuffled]
    for values in (null_positive, null_negative):
        if values.shape[0] > 1:
            values[:] = values[np.lexsort((values[:, 1], values[:, 0]))]
    return null_positive, null_negative


__all__ = [
    "EPS",
    "SpatialGraph",
    "build_spatial_graph",
    "degree_preserving_randomized_graph",
    "permute_coordinates_within_sections",
    "permute_edge_labels",
    "permuted_coordinate_graph",
    "sample_local_negative_edges",
]
