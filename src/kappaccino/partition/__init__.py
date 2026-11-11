"""Uncertain lensed images assigned to sources jointly with the lens.

``groups`` adds the uncertain images to the likelihood as an exact sum over their partitions into groups that share a
new source, with each image alone either unrelated or a new image of a trusted family.  The sources are integrated
out, so no parameters are added.  ``inputs`` reads the images and computes the lens-independent terms, and
``lens_evidence`` and ``catalog_evidence`` give the evidence of a group of images as one source.  See
``docs/likelihoods.md`` (Uncertain images).
"""
from .groups import GroupModel, add_uncertain_images, write_partition_outputs

__all__ = ["GroupModel", "add_uncertain_images", "write_partition_outputs"]
