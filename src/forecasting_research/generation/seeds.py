"""Deterministic seed policy.

seed = SeedSequence([base_seed, blake2b32(canonical_key), replication])

Seeds depend on a combination's *identity* (not its position in the design), so editing
the config lists never changes the data of combinations that remain.
"""

from __future__ import annotations

import hashlib

import numpy as np


def key_hash32(canonical_key: str) -> int:
    return int.from_bytes(hashlib.blake2b(canonical_key.encode("utf-8"), digest_size=4).digest(), "big")


def derive_seed(base_seed: int, canonical_key: str, replication: int) -> int:
    """Return a single 64-bit integer seed for one dataset replication."""
    ss = np.random.SeedSequence([int(base_seed), key_hash32(canonical_key), int(replication)])
    return int(ss.generate_state(1, dtype=np.uint64)[0])
