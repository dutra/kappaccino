"""Uncertain images with the sources integrated out: an exact sum over the partitions built from a library of groups.

Each uncertain image is either alone or one of a group of uncertain images that share a new source.  An image alone
is an unrelated source of its own or a new image of a trusted family F.  Given the lens Theta the likelihood of the
uncertain images is

    L(Theta) = sum_Z prod_{G in Z} w_G E_G(Theta) prod_{i alone in Z} a_i(Theta),
    a_i = pi_i0 E_i + sum_F pi_iF p(D_i | F),

the sum over the partitions Z of the uncertain images whose groups come from a library.  The terms are as follows.

- E_G is the evidence of the images of G as the images of one source.  The source position is integrated in closed
  form (``lens_evidence.family_lens_evidence``), the colors in closed form against the population mixture
  (``catalog_evidence``), and the redshift by quadrature on a grid spaced uniformly in the lensing efficiency.
- E_i is the same evidence for image i alone.
- p(D_i | F) is the position term of image i at the sampled source and redshift of family F, times the catalog
  predictive of the image given the family's members.

The prior weights follow a mixture over sources.  An image is unrelated with weight pi_i0, joins trusted family F with
pi_iF, or joins one of K new sources with pi_ik.  The weights are shared equally except for an image that a family
claims with an association prior (``inputs.association_log_weights``).  A group then has the weight
w_G = sum_k prod_{i in G} pi_ik of falling on one new source.  K is the number of groups the image catalog claims plus
the number of groups linked at the start lens (pairs whose evidence as one source beats both images alone).

Two images are connected when a library group holds both.  The sum factorizes over the connected components.  The
partitions of each component (the sets of disjoint library groups, with the other images alone) are enumerated once,
before the fit, so the likelihood is a fixed log-sum-exp over the partitions and the lens is the only parameter.  The
library holds every group of mutually compatible uncertain images at any of a few lens models.  A pair enters when the
evidence of the pair as one source comes within ``library_pair_margin`` (log) of the two images alone, and a group of
linked images when the group's evidence comes within ``library_group_margin`` of its images alone.  The library is
built at the start lens, the lens is refit with the library summed, and the library is rebuilt at every lens so far
(``library_refits`` times).  A group not compatible at any of those lenses never enters the sum.

After sampling, the probability of every partition at each posterior draw is exact.  ``GroupModel.posterior`` turns
these into the probability of every option of each image, the probability that two images share a source, and joint
draws of the partition.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.special import logsumexp as np_logsumexp

from ..core.model import ModelState
from ..io.hdf5 import save_arrays
from .inputs import (UncertainImageInputs, association_log_weights, association_priors, build_inputs, claim_columns, claimed_groups,
                     position_loglike)
from .lens_evidence import LensImages, error_scales, family_lens_evidence, grid_fields, image_terms, reference_fields

LOG_FLOOR = -1.0e30            # stands in for log 0, so that a partition with an impossible group stays finite
MAX_PARTITIONS = 20000         # partitions allowed in one component; the least supported groups are pruned beyond it
OUTPUT_FILE = "uncertain_images.h5"


def lens_terms(evaluator, params: jnp.ndarray, stats: dict[str, Any], images: LensImages, trusted_ratio: np.ndarray,
               membership: jnp.ndarray | None = None) -> tuple[dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """The lens side of the uncertain images at one lens: the per-image terms on the redshift grid (``image_terms``),
    the lens evidence of each image alone, shape (image, grid), and the position term of each image at each trusted
    family's source, shape (image, family)."""
    physical = evaluator.physical(params)
    reference = reference_fields(evaluator, physical, images, membership)
    grid = image_terms(grid_fields(reference, images), error_scales(evaluator, physical, images, images.ratio), images)
    n = images.n_images
    alone = family_lens_evidence(grid, images, jnp.arange(n)[:, None], jnp.ones((n, 1), dtype=bool))["log_evidence"]
    ratio = jnp.asarray(trusted_ratio)
    trusted_terms = image_terms(grid_fields(reference, images, ratio), error_scales(evaluator, physical, images, ratio), images)
    trusted = position_loglike(trusted_terms, stats["source_by_family"], images.parity).T
    return grid, alone, trusted


@dataclass(frozen=True, eq=False)
class GroupModel:
    """The likelihood of the uncertain images summed over the partitions of a group library (module docstring).

    Attributes
    ----------
    images : LensImages
        The uncertain images, the redshift grid and the source prior.
    labels : tuple of str
        Uncertain image labels, length (image,).
    trusted_ids : tuple of str
        Trusted family ids in ``ImageData`` order.
    trusted_ratio, trusted_catalog, singleton_catalog : numpy.ndarray
        As in ``inputs.UncertainImageInputs``.
    log_alone_weights : numpy.ndarray
        log pi_i0 and log pi_iF of each image, shape (image, 1 + family).
    groups : tuple of tuple of int
        The library: image indices of each group of two or more images.
    group_index, group_mask : numpy.ndarray
        The library padded to the largest group, shape (group, width).
    group_catalog : numpy.ndarray
        Catalog evidence of each group on the redshift grid (quadrature weight included), shape (group, grid).
    log_group_weight : numpy.ndarray
        log w_G, shape (group,).
    packing_groups : numpy.ndarray
        Which library groups each partition of each component uses, bool of shape (partition, group).
    packing_alone : numpy.ndarray
        Which images each partition leaves alone, bool of shape (partition, image).
    packing_component : numpy.ndarray
        The component of each partition, shape (partition,).
    n_components : int
    image_component : numpy.ndarray
        The component of each image, shape (image,).
    group_members : numpy.ndarray
        Which images each library group holds, float 0/1 of shape (group, image).
    """
    images: LensImages
    labels: tuple[str, ...]
    trusted_ids: tuple[str, ...]
    trusted_ratio: np.ndarray
    trusted_catalog: np.ndarray
    singleton_catalog: np.ndarray
    log_alone_weights: np.ndarray
    groups: tuple[tuple[int, ...], ...]
    group_index: np.ndarray
    group_mask: np.ndarray
    group_catalog: np.ndarray
    log_group_weight: np.ndarray
    packing_groups: np.ndarray
    packing_alone: np.ndarray
    packing_component: np.ndarray
    n_components: int
    image_component: np.ndarray
    group_members: np.ndarray

    @property
    def n_images(self) -> int:
        return len(self.labels)

    # ------------------------------------------------------------------ likelihood
    def terms(self, evaluator, params: jnp.ndarray, stats: dict[str, Any], membership: jnp.ndarray | None = None) -> dict[str, jnp.ndarray]:
        """The log weight of every option of an image alone (``options``, shape (image, 1 + family)), their sum
        (``log_alone``, log a_i), log E_G of every library group (``log_group``) and the log weight of every partition
        (``score``).  ``stats`` are the trusted images' statistics for the same ``membership`` (indicators of the
        enumerated uncertain members, or None for their expected mass)."""
        grid, alone, trusted = lens_terms(evaluator, params, stats, self.images, self.trusted_ratio, membership)
        unrelated = jax.scipy.special.logsumexp(alone + jnp.asarray(self.singleton_catalog), axis=1)
        options = jnp.asarray(self.log_alone_weights) + jnp.column_stack([unrelated, trusted + jnp.asarray(self.trusted_catalog)])
        log_alone = jax.scipy.special.logsumexp(options, axis=1)
        if self.groups:
            lens = family_lens_evidence(grid, self.images, jnp.asarray(self.group_index), jnp.asarray(self.group_mask))["log_evidence"]
            log_group = jax.scipy.special.logsumexp(lens + jnp.asarray(self.group_catalog), axis=1)
        else:
            log_group = jnp.zeros(0)
        weighted = jnp.maximum(jnp.asarray(self.log_group_weight) + log_group, LOG_FLOOR)
        score = (jnp.asarray(self.packing_groups, dtype=jnp.float64) @ weighted
                 + jnp.asarray(self.packing_alone, dtype=jnp.float64) @ jnp.maximum(log_alone, LOG_FLOOR))
        return dict(options=options, log_alone=log_alone, log_group=log_group, score=score)

    def component_logsumexp(self, score: jnp.ndarray) -> jnp.ndarray:
        """log of the sum of exp(score) over the partitions of each component, shape (component,)."""
        segment = jnp.asarray(self.packing_component)
        peak = jax.ops.segment_max(score, segment, num_segments=self.n_components)
        total = jax.ops.segment_sum(jnp.exp(score - peak[segment]), segment, num_segments=self.n_components)
        return peak + jnp.log(total)

    def log_likelihood(self, evaluator, params: jnp.ndarray, stats: dict[str, Any], membership: jnp.ndarray | None = None) -> jnp.ndarray:
        """log L(Theta): the term ``Evaluator.log_likelihood`` adds (per configuration of the enumerated uncertain
        members, inside the membership sum)."""
        return jnp.sum(self.component_logsumexp(self.terms(evaluator, params, stats, membership)["score"]))

    # ------------------------------------------------------------------ posterior
    def option_names(self) -> list[str]:
        """The options of an image: ``unrelated``, the trusted family ids, then ``group<g>`` for library group g."""
        return ["unrelated"] + list(self.trusted_ids) + [f"group{g}" for g in range(len(self.groups))]

    def _at_lens(self, options: np.ndarray, log_alone: np.ndarray, score: np.ndarray) -> dict[str, np.ndarray]:
        """Partition probabilities at one lens: of every partition, of every option of each image and of each pair of
        images sharing a source."""
        component, member = np.asarray(self.packing_component), self.group_members
        prob = np.zeros_like(score)
        for c in range(self.n_components):
            rows = component == c
            prob[rows] = np.exp(score[rows] - np_logsumexp(score[rows]))
        p_group = prob @ self.packing_groups
        p_alone = prob @ self.packing_alone
        share = np.exp(options - log_alone[:, None])                        # the options of an image given it is alone
        responsibilities = np.concatenate([p_alone[:, None] * share, (p_group[:, None] * member).T], axis=1)
        same_component = self.image_component[:, None] == self.image_component[None]
        both_alone = np.where(same_component, (self.packing_alone.T * prob) @ self.packing_alone, np.outer(p_alone, p_alone))
        pair = (member.T * p_group) @ member + both_alone * (share[:, 1:] @ share[:, 1:].T)
        return dict(prob=prob, share=share, responsibilities=responsibilities, pair=pair)

    def _draw(self, at: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
        """One joint draw of the options of the images (option index per image) at one lens."""
        n_trusted = len(self.trusted_ids)
        component = np.asarray(self.packing_component)
        labels = np.zeros(self.n_images, dtype=int)
        for c in range(self.n_components):
            rows = np.flatnonzero(component == c)
            pick = rows[rng.choice(rows.size, p=at["prob"][rows] / at["prob"][rows].sum())]
            for g in np.flatnonzero(self.packing_groups[pick]):
                labels[list(self.groups[g])] = 1 + n_trusted + g
            for i in np.flatnonzero(self.packing_alone[pick]):
                labels[i] = rng.choice(1 + n_trusted, p=at["share"][i] / at["share"][i].sum())
        return labels

    def posterior(self, evaluator, samples: np.ndarray, rng: np.random.Generator, *, draws_per_sample: int = 1) -> dict[str, np.ndarray]:
        """Option probabilities and joint draws of the partition for latent posterior draws.

        Returns ``responsibilities`` (draw, image, option), sampled ``labels`` (joint draw, image; option index, with
        ``draws_per_sample`` joint draws per posterior draw), ``probability`` (image, option; averaged over draws) and
        ``pair_probability`` (image, image; the probability that two images share a source, a group or a trusted
        family).  With enumerated uncertain members, each draw averages every member configuration, weighted by the
        configuration's posterior.
        """
        enumerated = len(evaluator.membership_enumerated_positions) > 0

        def at_lens(params):
            if not enumerated:
                out = self.terms(evaluator, params, evaluator.statistics(params))
                return jnp.zeros(1), *(out[k][None] for k in ("options", "log_alone", "score"))
            log_weight, _ = evaluator._membership_configuration_terms(params, source_jacobian=True)
            per = jax.vmap(lambda m: self.terms(evaluator, params, evaluator.statistics(params, membership=m), m))(
                evaluator.membership_configurations)
            return log_weight - jax.scipy.special.logsumexp(log_weight), per["options"], per["log_alone"], per["score"]

        at_lens = jax.jit(at_lens)
        responsibilities, pair, labels = [], 0.0, []
        samples = np.atleast_2d(samples)
        for params in samples:
            log_weight, options, log_alone, score = (np.asarray(v) for v in at_lens(jnp.asarray(params, dtype=jnp.float64)))
            weight = np.exp(log_weight)
            per = [self._at_lens(options[m], log_alone[m], score[m]) for m in range(weight.size)]
            responsibilities.append(sum(w * p["responsibilities"] for w, p in zip(weight, per)))
            pair = pair + sum(w * p["pair"] for w, p in zip(weight, per))
            for _ in range(int(draws_per_sample)):
                labels.append(self._draw(per[rng.choice(weight.size, p=weight / weight.sum())], rng))
        pair = pair / len(samples)
        np.fill_diagonal(pair, 1.0)
        responsibilities = np.stack(responsibilities)
        return dict(responsibilities=responsibilities, labels=np.stack(labels), probability=responsibilities.mean(axis=0),
                    pair_probability=pair)


# ------------------------------------------------------------------ library
def _cliques(adjacent: np.ndarray, max_size: int) -> list[tuple[int, ...]]:
    """Every complete subgraph of two to ``max_size`` nodes."""
    n = adjacent.shape[0]
    out: list[tuple[int, ...]] = []

    def extend(clique: tuple[int, ...], candidates: list[int]) -> None:
        for k, v in enumerate(candidates):
            grown = clique + (v,)
            out.append(grown)
            if len(grown) < max_size:
                extend(grown, [w for w in candidates[k + 1:] if adjacent[v, w]])

    for v in range(n):
        extend((v,), [w for w in range(v + 1, n) if adjacent[v, w]])
    return out


def _components(groups, n: int) -> list[list[int]]:
    """The connected components of n nodes joined by ``groups`` (sorted lists of nodes, every node in one)."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for group in groups:
        for i in group[1:]:
            parent[find(i)] = find(group[0])
    out: dict[int, list[int]] = {}
    for i in range(n):
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def _packings(images: list[int], groups: list[int], library: list[tuple[int, ...]], limit: int) -> list[tuple[int, ...]]:
    """Every set of disjoint library groups within one component (the other images alone), as group indices."""
    by_first = {i: [g for g in groups if library[g][0] == i] for i in images}
    order = sorted(images)
    out: list[tuple[int, ...]] = []

    def walk(position: int, covered: frozenset, chosen: tuple[int, ...]) -> None:
        if len(out) > limit:
            raise RuntimeError(f"more than {limit} partitions in a component of {len(images)} images")
        while position < len(order) and order[position] in covered:
            position += 1
        if position == len(order):
            out.append(chosen)
            return
        i = order[position]
        walk(position + 1, covered | {i}, chosen)                        # i alone
        for g in by_first[i]:                                            # i the first image of group g
            if not covered.intersection(library[g]):
                walk(position + 1, covered | set(library[g]), chosen + (g,))

    walk(0, frozenset(), ())
    return out


def _padded(groups) -> tuple[np.ndarray, np.ndarray]:
    width = max([len(g) for g in groups], default=1)
    index = np.zeros((len(groups), width), dtype=np.int32)
    mask = np.zeros((len(groups), width), dtype=bool)
    for k, g in enumerate(groups):
        index[k, :len(g)], mask[k, :len(g)] = g, True
    return index, mask


def group_log_weight(log_new: np.ndarray, index: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """log w_G = log sum_k prod_{i in G} pi_ik for padded groups, from the log weights of each image on each new source
    (image, source); shape (group,)."""
    per_source = np.where(mask[..., None], log_new[index], 0.0).sum(axis=1)
    return np_logsumexp(per_source, axis=1)


def prior_weights(config, inputs: UncertainImageInputs, linked: list[tuple[int, ...]]) -> tuple[np.ndarray, np.ndarray]:
    """log pi of each image on its options alone (image, 1 + family) and on each new source (image, source).

    There is one new source per group the image catalog claims and per linked group (minus the claimed images).  An
    unrelated source gets ``unrelated_fraction`` and the trusted families and new sources share the rest equally,
    except for images claimed with an association prior (``association_weight``)."""
    partition = config.partition
    claimed = claimed_groups(inputs.table, inputs.trusted_ids) if partition.association_weight else []
    taken = {i for g in claimed for i in g}
    sources = claimed + [g for g in (tuple(i for i in group if i not in taken) for group in linked) if len(g) > 1]
    n_trusted, n_new = len(inputs.trusted_ids), max(len(sources), 1)
    weights = np.full(1 + n_trusted + n_new, (1.0 - partition.unrelated_fraction) / (n_trusted + n_new))
    weights[0] = partition.unrelated_fraction
    log_weights = np.log(np.broadcast_to(weights, (inputs.n_images, weights.size)))
    if partition.association_weight:
        log_weights = association_log_weights(weights, claim_columns(inputs.table, inputs.trusted_ids, claimed),
                                              association_priors(inputs.table))
    return log_weights[:, :1 + n_trusted], log_weights[:, 1 + n_trusted:]


def build_library(inputs: UncertainImageInputs, lenses: list[tuple], log_alone_weights: np.ndarray, log_new: np.ndarray, *,
                  pair_margin: float, group_margin: float, max_size: int, required: list[tuple[int, ...]] = (),
                  limit: int = MAX_PARTITIONS) -> tuple[list[tuple[int, ...]], int]:
    """The group library from ``lens_terms`` at several lenses, and the number of groups pruned.

    A pair is linked when, at any of the lenses, the weighted evidence of the pair as one source exceeds the two images
    alone minus ``pair_margin``.  Every clique of linked images (up to ``max_size``) whose weighted evidence exceeds its
    images alone minus ``group_margin`` at any lens enters, as do the ``required`` groups.  A component with more than
    ``limit`` partitions loses its least supported groups until it fits.
    """
    images, n = inputs.images, inputs.n_images
    alternatives = []
    for _, alone, trusted in lenses:
        unrelated = np_logsumexp(np.asarray(alone) + inputs.singleton_catalog, axis=1)
        options = log_alone_weights + np.column_stack([unrelated, np.asarray(trusted) + inputs.trusted_catalog])
        alternatives.append(np_logsumexp(options, axis=1))

    def gains(groups) -> np.ndarray:
        """The best, over the lenses, of log w_G E_G minus the sum of log a_i over the group's images."""
        index, mask = _padded(groups)
        catalog = inputs.catalog(index, mask)
        weight = group_log_weight(log_new, index, mask)
        best = np.full(len(groups), -np.inf)
        for (grid, _, _), alternative in zip(lenses, alternatives):
            lens = np.asarray(family_lens_evidence(grid, images, jnp.asarray(index), jnp.asarray(mask))["log_evidence"])
            best = np.maximum(best, weight + np_logsumexp(lens + catalog, axis=1) - np.where(mask, alternative[index], 0.0).sum(axis=1))
        return best

    pairs = list(combinations(range(n), 2))
    adjacent = np.zeros((n, n), dtype=bool)
    if pairs:
        linked = np.asarray(pairs)[gains(pairs) > -pair_margin]
        adjacent[linked[:, 0], linked[:, 1]] = adjacent[linked[:, 1], linked[:, 0]] = True
    candidates = _cliques(adjacent, max_size)
    support = dict(zip(candidates, gains(candidates))) if candidates else {}
    library = [g for g, value in support.items() if value > -group_margin]
    required = [tuple(sorted(g)) for g in required]
    support.update({g: np.inf for g in required})
    library = sorted(set(library) | set(required))
    pruned = 0
    while True:
        too_big = None
        for members in _components(library, n):
            groups = [g for g, group in enumerate(library) if group[0] in members]
            try:
                _packings(members, groups, library, limit)
            except RuntimeError:
                too_big = groups
                break
        if too_big is None:
            return library, pruned
        optional = [g for g in too_big if support[library[g]] < np.inf]
        if not optional:
            raise RuntimeError(f"the required groups alone give more than {limit} partitions in one component")
        drop = {library[g] for g in sorted(optional, key=lambda g: support[library[g]])[:max(1, len(optional) // 10)]}
        library = [g for g in library if g not in drop]
        pruned += len(drop)


def group_model(inputs: UncertainImageInputs, library: list[tuple[int, ...]], log_alone_weights: np.ndarray, log_new: np.ndarray,
                *, limit: int = MAX_PARTITIONS) -> GroupModel:
    """The ``GroupModel`` of a library: the catalog evidence and weight of every group and the partitions of every
    component."""
    n = inputs.n_images
    index, mask = _padded(library)
    packing_groups, packing_alone, packing_component = [], [], []
    components = _components(library, n)
    for c, members in enumerate(components):
        groups = [g for g, group in enumerate(library) if group[0] in members]
        for chosen in _packings(members, groups, library, limit):
            used = np.zeros(len(library), dtype=bool)
            used[list(chosen)] = True
            alone = np.zeros(n, dtype=bool)
            alone[members] = True
            for g in chosen:
                alone[list(library[g])] = False
            packing_groups.append(used)
            packing_alone.append(alone)
            packing_component.append(c)
    image_component = np.zeros(n, dtype=int)
    for c, members in enumerate(components):
        image_component[members] = c
    group_members = np.zeros((len(library), n))
    for g, group in enumerate(library):
        group_members[g, list(group)] = 1.0
    return GroupModel(images=inputs.images, labels=inputs.labels, trusted_ids=inputs.trusted_ids, trusted_ratio=inputs.trusted_ratio,
                      trusted_catalog=inputs.trusted_catalog, singleton_catalog=inputs.singleton_catalog,
                      log_alone_weights=log_alone_weights, groups=tuple(library), group_index=index, group_mask=mask,
                      group_catalog=inputs.catalog(index, mask) if library else np.zeros((0, inputs.images.z_grid.size)),
                      log_group_weight=group_log_weight(log_new, index, mask) if library else np.zeros(0),
                      packing_groups=np.asarray(packing_groups, dtype=bool).reshape(len(packing_component), len(library)),
                      packing_alone=np.asarray(packing_alone, dtype=bool),
                      packing_component=np.asarray(packing_component, dtype=np.int32), n_components=len(components),
                      image_component=image_component, group_members=group_members)


def linked_groups(inputs: UncertainImageInputs, lens: tuple) -> list[tuple[int, ...]]:
    """Groups of uncertain images linked at one lens: connected by pairs whose evidence as one source beats the best
    alternative of each image (an unrelated source or its best trusted family), unweighted."""
    n = inputs.n_images
    pairs = list(combinations(range(n), 2))
    if not pairs:
        return []
    grid, alone, trusted = lens
    unrelated = np_logsumexp(np.asarray(alone) + inputs.singleton_catalog, axis=1)
    best = np.maximum(unrelated, np.max(np.asarray(trusted) + inputs.trusted_catalog, axis=1, initial=-np.inf))
    index, mask = _padded(pairs)
    evidence = np.asarray(family_lens_evidence(grid, inputs.images, jnp.asarray(index), jnp.asarray(mask))["log_evidence"])
    value = np_logsumexp(evidence + inputs.catalog(index, mask), axis=1)
    linked = [pair for pair, gain in zip(pairs, value - best[index].sum(axis=1)) if gain > 0.0]
    return [tuple(g) for g in _components(linked, n) if len(g) > 1]


def add_uncertain_images(config, state: ModelState, evaluator, theta: np.ndarray, *,
                         refit: Callable[[ModelState, Any, np.ndarray], np.ndarray] | None = None,
                         log=print) -> tuple[ModelState, Any, np.ndarray]:
    """The state with the uncertain images of ``config.partition``, its evaluator, and the lens to start from.

    The library is built at ``theta`` (the trusted-family start).  With ``refit`` (a function of the state, evaluator
    and start returning the best lens), the lens is refit with the library summed and the library rebuilt at every
    lens so far, ``partition.library_refits`` times; the last lens is returned (``theta`` without refits).  No
    parameters are added to the lens model.  Deterministic given the inputs, so a resumed final stage rebuilds the same
    library.
    """
    from ..likelihood.evaluator import Evaluator
    partition = config.partition
    inputs = build_inputs(config, state, log=log)
    at = jax.jit(lambda params: lens_terms(evaluator, params, evaluator.statistics(params), inputs.images, inputs.trusted_ratio))
    lenses = [np.asarray(theta, dtype=float)]
    terms = [at(jnp.asarray(lenses[0], dtype=jnp.float64))]
    log_alone_weights, log_new = prior_weights(config, inputs, linked_groups(inputs, terms[0]))
    claimed = claimed_groups(inputs.table, inputs.trusted_ids) if partition.association_weight else []

    def build() -> tuple[ModelState, Any]:
        library, pruned = build_library(inputs, terms, log_alone_weights, log_new, pair_margin=float(partition.library_pair_margin),
                                        group_margin=float(partition.library_group_margin), max_size=int(partition.library_max_group),
                                        required=claimed)
        model = group_model(inputs, library, log_alone_weights, log_new)
        sizes = sorted(((int(np.sum(model.image_component == c)), int(np.sum(model.packing_component == c)))
                        for c in range(model.n_components)), key=lambda s: -s[1])[:3]
        log(f"[uncertain] library of {len(library)} groups at {len(terms)} lens(es) ({pruned} pruned); {log_new.shape[1]} new "
            f"source(s) in the prior; {model.n_components} components, {len(model.packing_component)} partitions; largest "
            f"components (images, partitions) {sizes}")
        extended = replace(state, uncertain_images=model)
        return extended, Evaluator(extended, evaluator.likelihood_mode)

    for _ in range(int(partition.library_refits) if refit is not None else 0):
        extended, extended_evaluator = build()
        lenses.append(np.asarray(refit(extended, extended_evaluator, lenses[-1]), dtype=float))
        terms.append(at(jnp.asarray(lenses[-1], dtype=jnp.float64)))
    extended, extended_evaluator = build()
    return extended, extended_evaluator, lenses[-1]


# ------------------------------------------------------------------ outputs
def write_partition_outputs(state: ModelState, evaluator, samples: np.ndarray, stage, *, draws: int, seed: int,
                            draws_per_sample: int = 1, log=print) -> dict[str, Any]:
    """Option probabilities and joint draws of the partition of the uncertain images from posterior draws.

    Uses up to ``draws`` of ``samples`` (evenly spaced), with ``draws_per_sample`` joint draws of the partition at each.
    Writes ``<stage>/artifacts/uncertain_images.h5`` (``options``, ``image_labels``, the library ``groups`` as
    comma-separated labels, per-draw ``responsibilities``, sampled ``assignment``, ``probability`` and
    ``pair_probability``), ``<stage>/tables/uncertain_images.csv`` (per image: the probability of every option and the
    most probable one) and ``<stage>/tables/uncertain_pairs.csv`` (image pairs sharing a source with probability above
    0.01).
    """
    model: GroupModel = state.uncertain_images
    samples = np.atleast_2d(np.asarray(samples, dtype=float))
    samples = samples[np.isfinite(samples).all(axis=1)]
    chosen = samples[np.unique(np.linspace(0, len(samples) - 1, min(int(draws), len(samples))).astype(int))]
    result = model.posterior(evaluator, chosen, np.random.default_rng(int(seed)), draws_per_sample=draws_per_sample)
    options = model.option_names()
    labels = np.asarray(model.labels)
    save_arrays(stage.artifacts / OUTPUT_FILE, options=np.asarray(options), image_labels=labels,
                groups=np.asarray([",".join(labels[list(g)]) for g in model.groups], dtype=str), assignment=result["labels"],
                responsibilities=result["responsibilities"], probability=result["probability"], pair_probability=result["pair_probability"])
    table = pd.DataFrame(result["probability"], columns=[f"p_{o}" for o in options])
    best = np.argmax(result["probability"], axis=1)
    table.insert(0, "image_label", list(labels))
    table.insert(1, "most_probable", [options[k] for k in best])
    table.insert(2, "p_most_probable", result["probability"][np.arange(len(best)), best])
    table.to_csv(stage.tables / "uncertain_images.csv", index=False)
    i, j = np.triu_indices(len(labels), k=1)
    p = result["pair_probability"][i, j]
    keep = p > 0.01
    pairs = pd.DataFrame({"image_a": labels[i[keep]], "image_b": labels[j[keep]], "p_same_source": p[keep]})
    pairs.sort_values("p_same_source", ascending=False).to_csv(stage.tables / "uncertain_pairs.csv", index=False)
    for label, option, probability in zip(labels, table["most_probable"], table["p_most_probable"]):
        log(f"[uncertain] {label}: {option} (p = {probability:.2f})")
    return result
