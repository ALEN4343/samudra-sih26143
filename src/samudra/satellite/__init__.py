"""Satellite operations — layer 1 ingest with provenance, and layer 3 inference.

The console this backs exists to answer one question honestly: where did this
image come from, and did the model really produce that mask? See sources.py for
the provenance rules and inference.py for the ground-truth guard.
"""

from samudra.satellite import inference, pipeline, sources

__all__ = ["inference", "pipeline", "sources"]
