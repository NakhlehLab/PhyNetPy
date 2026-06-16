"""Remove MSNC DP code duplicated in _mcmc_gt.py (now lives in _msnc_density.py)."""
from pathlib import Path

path = Path(__file__).resolve().parents[1] / "src/_mcmc_gt.py"
lines = path.read_text(encoding="utf-8").splitlines(keepends=True)

# 0-based inclusive ranges to DELETE (in original file; recompute after each pass)
# We'll delete from bottom to top to preserve indices.
# 0-based half-open intervals [start, end)
ranges = [
    (1853, 2094),  # msnc DP section through _frontier_acc
    (1654, 1852),  # _msc_log_prob_tree_int through cython bind block
    (1349, 1366),  # _logsumexp
    (986, 1097),   # engine _fact_range through _log_denom
    (479, 883),    # _GeneTreeIndex through _NetworkIndex
]

for start, end in sorted(ranges, reverse=True):
    del lines[start:end]

path.write_text("".join(lines), encoding="utf-8")
print(f"stripped {path}, now {len(lines)} lines")
