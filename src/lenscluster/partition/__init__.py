"""Uncertain lensed images assigned to sources jointly with the lens.

``slots`` adds the uncertain images to the position likelihood as a mixture over source slots (trusted families, new
sampled sources, or an unrelated source), with each image's assignment summed out; ``lens_evidence`` and
``catalog_evidence`` give the per-image terms and the evidence of an image as its own source.  See
``docs/uncertain-images.md``.
"""
from .slots import SlotModel, add_slots, extend_theta

__all__ = ["SlotModel", "add_slots", "extend_theta"]
