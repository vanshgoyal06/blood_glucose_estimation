# Recommended Correction (Not Yet Official Methodology)

The current 250-row feature generator groups by `Age`. The audit found 701 contiguous blocks in the raw order, but the project has not established what those blocks represent.

If a trustworthy session/recording identifier becomes available, use that identifier as the rolling-history boundary. `src/features/block_aware.py` contains a helper for this case.

Do **not** substitute the derived contiguous block number automatically. A block is currently only a structural diagnostic, not a confirmed participant/session ID.

Once the correct boundary is established, regenerate the history features and repeat all evaluation. The resulting numbers should be treated as a new experiment.
