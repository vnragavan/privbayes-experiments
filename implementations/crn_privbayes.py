from __future__ import annotations

import hashlib
import os
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd
from pandas.api.types import is_integer_dtype, is_numeric_dtype, is_bool_dtype

SMOOTH: float = 1e-8  # additive smoothing for probabilities


def _constraint_aware_clip_bounds(pb_lo, pb_hi, col_spec, column_type):
    """
    Compute (lo, hi) for clipping so output satisfies schema column_constraints.
    Used by schema-native sample() to enforce min_exclusive, min, max, max_exclusive.
    """
    lo, hi = pb_lo, pb_hi
    if col_spec:
        if col_spec.get("min_exclusive") is not None:
            x = col_spec["min_exclusive"]
            if column_type == "integer":
                lo = max(lo, x + 1) if lo is not None else x + 1
            else:
                eps = 1e-9 if hi is None or hi > x + 1 else (float(hi) - x) / 2
                lo = max(lo, x + eps) if lo is not None else x + eps
        if col_spec.get("min") is not None:
            x = col_spec["min"]
            lo = max(lo, x) if lo is not None else x
        if col_spec.get("max") is not None:
            x = col_spec["max"]
            hi = min(hi, x) if hi is not None else x
        if col_spec.get("max_exclusive") is not None:
            x = col_spec["max_exclusive"]
            if column_type == "integer":
                hi = min(hi, x - 1) if hi is not None else x - 1
            else:
                hi = min(hi, x - 1e-9) if hi is not None else x - 1e-9
    return lo, hi


# ---- Minimal "register" shim for compatibility ----

def register(*args, **kwargs):
    def decorator(cls_or_func):
        return cls_or_func
    return decorator

# ======================== Auto-tuning (utility ↑ with ε) ========================

@dataclass
class PBTune:
    eps_split: Dict[str, float]
    eps_disc: float
    bins_per_numeric: int
    max_parents: int
    cat_buckets: int
    cat_topk: int
    dp_bounds_mode: str  # "public" or "smooth"
    dp_quantile_alpha: float  # e.g., 0.01 => [1%, 99%] bounds

def auto_tune_for_epsilon(
    epsilon: float,
    n: int,
    d: int,
    *,
    have_public_bounds: bool,
    target_high_utility: bool = True
) -> PBTune:
    """
    Heuristic tuning schedule aimed at monotonic utility w.r.t ε.
    - More ε to CPTs as ε grows (structure still gets a meaningful slice).
    - bins_per_numeric increases slowly with ε (caps to avoid CPT blow-ups).
    - max_parents increases at higher ε (2 -> 3).
    - Small metadata budget; use "smooth" DP bounds if no public coarse bounds.
    """
    eps = float(max(epsilon, 1e-6))
    # Structure/CPT split: favor CPTs slightly as ε grows
    s_frac = 0.35 if eps < 0.5 else (0.30 if eps < 2 else 0.25)
    c_frac = 1.0 - s_frac
    # Reserve a thin slice for metadata (bounds/domains). Use less as ε grows.
    disc_frac = 0.12 if eps < 0.5 else (0.08 if eps < 2 else 0.05)
    disc_frac = min(disc_frac, 0.15)
    # Numeric discretization granularity grows slowly with ε (cap to 64)
    base_bins = 8
    extra = int(np.floor(np.log2(1 + eps * 10.0)))
    bins_per_numeric = int(np.clip(base_bins + extra, 8, 64))
    # Parent width: keep small at low ε; allow 3 at higher ε if d is large.
    max_parents = 2 if eps < 1.5 else (3 if d >= 16 else 2)
    # DP categorical via hash buckets: keep domain bounded & stable
    cat_buckets = 64 if eps < 1.0 else (96 if eps < 2.0 else 128)
    cat_topk = 24 if eps < 1.0 else (28 if eps < 2.0 else 32)
    dp_bounds_mode = "public" if have_public_bounds else "smooth"
    dp_quantile_alpha = 0.01  # [1%, 99%] clipping for smooth DP bounds
    eps_disc = float(np.clip(disc_frac * eps, 0.0, eps))
    eps_split = {"structure": s_frac, "cpt": c_frac}
    return PBTune(
        eps_split=eps_split,
        eps_disc=eps_disc,
        bins_per_numeric=bins_per_numeric,
        max_parents=max_parents,
        cat_buckets=cat_buckets,
        cat_topk=cat_topk,
        dp_bounds_mode=dp_bounds_mode,
        dp_quantile_alpha=dp_quantile_alpha,
    )

# ========================== Helpers for DP metadata ===========================

def _blake_bucket(s: str, m: int) -> int:
    """Hash string to bucket index using BLAKE2b.
    
    Deterministic mapping for DP categorical heavy hitters. Returns integer
    in range [0, m-1] for bucket assignment.
    """
    h = hashlib.blake2b(s.encode("utf-8", errors="ignore"), digest_size=16)
    return int.from_bytes(h.digest(), "little") % int(m)

def _quantile_indices(n: int, q: float) -> int:
    """Compute array index for quantile q in sorted array of length n.
    
    Returns 0-based index. Uses ceiling to handle edge cases consistently.
    """
    q = float(np.clip(q, 0.0, 1.0))
    # Safe upper bound even when n == 0 (callers guard x.size == 0 before use).
    return int(np.clip(int(np.ceil(q * n)) - 1, 0, max(n - 1, 0)))

def _smooth_sensitivity_quantile(
    x: np.ndarray,
    q: float,
    eps: float,
    delta: float,
    rng: np.random.Generator,
    beta_scale: float = 1.0,
) -> float:
    """
    Approximate smooth sensitivity mechanism for a quantile (Nissim–Raskhodnikova–Smith'07).
    Produces an (ε, δ)-DP noisy quantile without public bounds.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    x.sort()
    n = x.size
    i = _quantile_indices(n, q)
    delta = float(np.clip(delta, 1e-15, 1.0 - 1e-12))
    eps = float(max(eps, 1e-12))
    # NRS'07 calibration
    beta = beta_scale * (eps / (2.0 * np.log(1.0 / delta)))
    max_s = 0.0
    k_max = min(n - 1, int(np.ceil(4.0 * np.sqrt(n + 1))))
    for k in range(0, k_max + 1):
        l = max(i - k, 0)
        r = min(i + k, n - 1)
        ls = float(x[r] - x[l])
        ss = np.exp(-beta * k) * ls
        if ss > max_s:
            max_s = ss
    # Correct scale factor: 2 * S* / ε
    scale = (2.0 * max_s) / eps
    noise = rng.laplace(0.0, scale)
    y = float(x[i] + noise)
    return y

def _dp_numeric_bounds_public(
    col: pd.Series,
    eps_min: float,
    eps_max: float,
    coarse_bounds: Tuple[float, float],
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """
    Pure ε-DP: with public coarse [L,U], add Laplace noise to min/max of data clipped to [L,U].
    """
    L, U = coarse_bounds
    x = pd.to_numeric(col, errors="coerce").to_numpy()
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(L), float(U)
    xc = np.clip(x, L, U)
    sens = float(max(U - L, 0.0))
    lo = float(np.min(xc) + rng.laplace(0.0, sens / max(eps_min, 1e-12)))
    hi = float(np.max(xc) + rng.laplace(0.0, sens / max(eps_max, 1e-12)))
    if not np.isfinite(hi) or hi <= lo:
        hi = lo + (U - L) / 100.0
    lo = float(np.clip(lo, L, U))
    hi = float(np.clip(hi, L, U))
    return lo, hi

def _dp_numeric_bounds_smooth(
    col: pd.Series,
    eps_total: float,
    delta_total: float,
    alpha: float,
    rng: np.random.Generator,
    public_coarse: Optional[Tuple[float, float]] = None,
) -> Tuple[float, float]:
    """
    (ε,δ)-DP bounds via smooth-sensitivity quantiles.
    """
    x = pd.to_numeric(col, errors="coerce").to_numpy()
    x = x[np.isfinite(x)]
    if x.size == 0:
        if public_coarse is not None:
            return float(public_coarse[0]), float(public_coarse[1])
        return 0.0, 1.0
    eps_each = max(eps_total, 1e-12) * 0.5
    delta_each = max(delta_total, 1e-15) * 0.5
    qL = _smooth_sensitivity_quantile(x, alpha, eps_each, delta_each, rng)
    qU = _smooth_sensitivity_quantile(x, 1.0 - alpha, eps_each, delta_each, rng)
    if public_coarse is not None:
        Lc, Uc = public_coarse
        qL = float(np.clip(qL, Lc, Uc))
        qU = float(np.clip(qU, Lc, Uc))
    if not np.isfinite(qU) or qU <= qL:
        if public_coarse is not None:
            span = (public_coarse[1] - public_coarse[0]) / 100.0
            qU = float(qL + max(span, 1.0))
        else:
            qU = float(qL + (np.nanmax(x) - np.nanmin(x) + 1.0) / 100.0)
    return float(qL), float(qU)


def _validate_schema_numeric_column(
    orig_col: pd.Series,
    x_num: np.ndarray,
    *,
    colname: str,
    stype: str,
    int_stypes: set,
    missing_string_sentinels: Optional[set] = None,
    reveal_row_index_in_errors: bool = False,
) -> None:
    """
    Validate that schema-authoritative numeric column coercion did not silently
    turn non-missing tokens into NaN, and that values satisfy stype constraints.

    - orig_col: original column (pre-to_numeric)
    - x_num: result of pd.to_numeric(orig_col, errors="coerce").to_numpy(float)
    """
    if missing_string_sentinels is None:
        missing_string_sentinels = {"", "na", "n/a", "null", "none", "nan"}

    # Identify entries that became NaN after numeric coercion
    nan_after = np.isnan(x_num)
    if np.any(nan_after):
        s = orig_col.astype("string")
        # "true missing" according to pandas
        missing_pandas = s.isna()

        # Strings that should be treated as missing
        s_norm = s.fillna("").str.strip().str.lower()
        missing_sentinel = s_norm.isin(list(missing_string_sentinels))

        # Bad means: turned into NaN but was not missing by our rules
        bad = nan_after & ~(missing_pandas.to_numpy() | missing_sentinel.to_numpy())
        if np.any(bad):
            n_bad = int(np.sum(bad))
            if reveal_row_index_in_errors:
                i0 = int(np.flatnonzero(bad)[0])
                raise ValueError(
                    f"Column '{colname}' (schema_type={stype!r}) has non-numeric value at row {i0} "
                    f"(n_bad={n_bad})."
                )
            # Default path avoids both row indices and raw tokens; only counts are revealed.
            raise ValueError(
                f"Column '{colname}' (schema_type={stype!r}) contains {n_bad} non-numeric value(s)."
            )

    # Now enforce stype-specific constraints on finite values
    finite = np.isfinite(x_num)

    if stype == "binary":
        # Treat any non-NaN non-finite value (e.g., ±inf) as invalid in binary columns.
        nonfinite = ~np.isfinite(x_num)
        bad_nonfinite = nonfinite & ~np.isnan(x_num)
        if np.any(bad_nonfinite):
            raise ValueError(
                f"Column '{colname}' is declared binary but contains non-finite values."
            )
        ok = finite & ((x_num == 0.0) | (x_num == 1.0))
        if not np.all(ok | ~finite):
            raise ValueError(
                f"Column '{colname}' is declared binary but contains values other than 0 and 1."
            )

    elif stype in int_stypes:
        if np.any(finite):
            xf = x_num[finite]
            # Allow values that are exactly integers in float representation
            if not np.all(xf == np.floor(xf)):
                raise ValueError(
                    f"Column '{colname}' (schema_type={stype!r}) contains non-integer numeric values."
                )

    else:
        # continuous: forbid inf/-inf (NaN already handled above)
        if np.any(~np.isfinite(x_num) & ~np.isnan(x_num)):
            raise ValueError(
                f"Column '{colname}' (schema_type={stype!r}) contains non-finite values (inf)."
            )


# ============================ Model internals ============================

@dataclass
class _ColMeta:
    kind: str  # "numeric" or "categorical"
    k: int
    bins: Optional[np.ndarray] = None
    cats: Optional[List[str]] = None
    is_int: bool = False
    bounds: Optional[Tuple[float, float]] = None
    binary_numeric: bool = False
    original_dtype: Optional[np.dtype] = None
    all_nan: bool = False
    # DP hashed-categorical flags
    hashed_cats: bool = False
    hash_m: Optional[int] = None
    schema_type: Optional[str] = None
    explicit_bin_edges: Optional[np.ndarray] = None

@register("model", "privbayes")
class PrivBayesSynthesizerEnhanced:
    """Enhanced Differentially Private PrivBayes with QI-linkage reduction.
    
    Supported data types:
    - Numeric: int, float, decimal (discretized into bins)
    - Categorical: string/varchar (uses DP heavy hitters when domain unknown)
    - Boolean: binary numeric or categorical
    - Datetime/timedelta: converted to nanoseconds since epoch
      * Handles datetime64[ns] and string-formatted dates from CSV
      * Formats: '2023-01-15 10:30:00', '2023-01-15T10:30:00', etc.
    - Object columns: auto-detected as datetime (if 95%+ parseable), then numeric (if 95%+ convertible), else categorical
    
    __UNK__ tokens and how to avoid them:
    
    Without public categories, categoricals use DP heavy hitters:
    - Values hashed into buckets (B000, B001, etc.) for privacy
    - Only top-K buckets kept in vocabulary
    - Values in non-top-K buckets become __UNK__
    
    Strategies to reduce/avoid __UNK__:
    
    1. Provide public_categories (best - no UNK, no DP cost):
       For public domains (US states, ISO codes, etc.), provide all values.
       Example: public_categories={'state': ['CA', 'NY', 'TX', ...]}
    
    2. Increase cat_topk (DP-safe, uses more epsilon):
       Keeps more top-K buckets. Trade-off: more epsilon for discovery, less UNK.
       Use cat_topk_overrides for per-column control.
    
    3. Increase cat_buckets (DP-safe, may help):
       More hash buckets can capture more categories, but UNK still occurs if not in top-K.
    
    4. Use label_columns (best for target variables):
       Label columns never use hashing, never get UNK. Example: label_columns=['income']
    
    5. Allocate more epsilon to categorical discovery:
       Increase eps_disc to learn more categories. Trade-off: less epsilon for structure/CPT.
    
    6. Use cat_keep_all_nonzero=True (universal strategy):
       Keeps all observed buckets instead of just top-K. Captures all training categories,
       minimizing __UNK__ to near-zero. DP-safe, but uses more memory. Default in adapter.
    
    Additional features:
      • temperature: flatten CPTs at sampling (p -> p^(1/temperature), temperature>=1)
      • forbid_as_parent: columns never allowed as parents (e.g., QIs)
      • parent_blacklist: {child: [parents_not_allowed]} for fine-grained edge bans
      • numeric_bins_overrides: {col: k} to coarsen discretization per column
      • integer_decode_mode: 'round' | 'stochastic' | 'granular'
      • numeric_granularity: {col: step} snaps floats to bands on decode
      • cat_topk_overrides / cat_buckets_overrides: per-column tail compression
    """

    # ------------------------------------------------------------------ #
    #  SCHEMA TYPE → behaviour table                                       #
    #                                                                      #
    #  schema_type  | public_bounds req | public_categories req | output   #
    #  ------------ | ----------------- | -------------------- | -------- #
    #  continuous   | yes               | —                    | float    #
    #  integer      | yes               | —                    | int64    #
    #  binary       | no (fixed [0,1])  | no (fixed {0,1})     | 0 or 1   #
    #  ordinal      | yes [1..K]        | —                    | int64    #
    #  categorical  | —                 | yes                  | string   #
    #  datetime     | yes (ns epoch)    | —                    | int64    #
    #  timedelta    | yes (ns duration) | —                    | int64    #
    # ------------------------------------------------------------------ #
    _NUMERIC_STYPES = frozenset({"continuous", "integer", "binary", "ordinal", "datetime", "timedelta"})
    _INT_STYPES     = frozenset({"integer", "ordinal", "datetime", "timedelta"})
    _KNOWN_STYPES   = _NUMERIC_STYPES | frozenset({"categorical"})

    def __init__(
        self,
        *,
        epsilon: float,
        delta: float = 1e-6,
        # Seed controlling *sampling* only (non-DP randomness). DP noise uses
        # its own RNG seeded from os.urandom and is never user-controlled.
        seed: Optional[int] = None,
        # tuning / privacy split
        eps_split: Optional[Dict[str, float]] = None,
        eps_disc: Optional[float] = None,
        max_parents: int = 2,
        bins_per_numeric: int = 16,
        adjacency: str = "unbounded",
        # DP metadata strategy
        dp_bounds_mode: str = "smooth",
        dp_quantile_alpha: float = 0.01,
        public_bounds: Optional[Dict[str, List[float]]] = None,
        public_categories: Optional[Dict[str, List[str]]] = None,
        public_binary_numeric: Optional[Dict[str, bool]] = None,
        original_data_bounds: Optional[Dict[str, List[float]]] = None,  # Original data min/max for clipping
        # DP heavy hitters for categoricals when domain private
        cat_buckets: int = 64,
        cat_topk: int = 28,
        # Universal strategy: if cat_topk is None or -1, keep all non-zero buckets
        cat_keep_all_nonzero: bool = False,  # If True, keep all buckets with noisy_count > 0
        # decoding
        decode_binary_as_bool: bool = False,
        cpt_dtype: str = "float64",
        # misc
        require_public: bool = False,
        strict_dp: bool = True,
        # ===== New knobs (all optional) =====
        temperature: float = 1.0,
        cpt_smoothing: float = 1.5,  # Pseudo-counts added after DP noise (post-processing, DP-safe)
        forbid_as_parent: Optional[List[str]] = None,
        parent_blacklist: Optional[Dict[str, List[str]]] = None,
        numeric_bins_overrides: Optional[Dict[str, int]] = None,
        integer_decode_mode: str = "round",
        integer_granularity: Optional[Dict[str, int]] = None,
        numeric_granularity: Optional[Dict[str, float]] = None,
        cat_topk_overrides: Optional[Dict[str, int]] = None,
        cat_buckets_overrides: Optional[Dict[str, int]] = None,
        # categorical unknown handling
        unknown_token: str = "__UNK__",
        categorical_unknown_to_nan: bool = False,
        # label columns (no hashing, no UNK)
        label_columns: Optional[List[str]] = None,
        schema: Optional[Dict[str, Any]] = None,
        # schema numeric validation (optional)
        missing_string_sentinels: Optional[set] = None,
        reveal_row_index_in_errors: bool = False,
        **kwargs: Any,
    ) -> None:
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        # DP noise RNG: always seeded from os.urandom so that fit-time Laplace
        # draws are unpredictable even if the caller knows or controls `seed`.
        dp_seed = int.from_bytes(os.urandom(8), "little")
        self._rng = np.random.default_rng(dp_seed)
        # Sampling RNG: user-controllable for reproducibility when generating
        # synthetic data from already-DP CPTs.
        self.seed = int(seed) if seed is not None else None

        if cpt_dtype not in ("float32", "float64"):
            raise ValueError("cpt_dtype must be 'float32' or 'float64'")
        self.cpt_dtype = cpt_dtype

        self.adjacency = str(adjacency).lower()
        if self.adjacency not in {"unbounded", "bounded"}:
            raise ValueError("adjacency must be 'unbounded' or 'bounded'")
        self._sens_count = 1.0 if self.adjacency == "unbounded" else 2.0

        # metadata / strategy
        self.require_public = bool(require_public)
        self.dp_bounds_mode = str(dp_bounds_mode).lower()
        if self.dp_bounds_mode not in {"public", "smooth"}:
            raise ValueError("dp_bounds_mode must be 'public' or 'smooth'")
        self.dp_quantile_alpha = float(dp_quantile_alpha)
        self.strict_dp = bool(strict_dp)

        # Schema numeric validation knobs (optional)
        # None => use default sentinel set in _validate_schema_numeric_column.
        # reveal_row_index_in_errors defaults to False to avoid leaking row ids
        # through exception messages unless the caller explicitly opts in.
        self.missing_string_sentinels: Optional[set] = missing_string_sentinels
        self.reveal_row_index_in_errors: bool = bool(reveal_row_index_in_errors)

        # public hints
        self.public_bounds: Dict[str, List[float]] = dict(public_bounds or {})
        self.public_categories: Dict[str, List[str]] = {k: list(v or []) for k, v in (public_categories or {}).items()}
        self.public_binary_numeric: Dict[str, bool] = dict(public_binary_numeric or {})
        self.original_data_bounds: Dict[str, List[float]] = dict(original_data_bounds or {})

        # DP heavy hitters defaults
        self.cat_buckets = int(cat_buckets)
        self.cat_topk = int(cat_topk) if cat_topk is not None and cat_topk > 0 else None
        self.cat_keep_all_nonzero = bool(cat_keep_all_nonzero)

        # main knobs
        self.max_parents = int(max_parents)
        self.bins_per_numeric = int(bins_per_numeric)

        # Default DP metadata budget
        if eps_disc is None:
            self.eps_disc = float(min(max(0.10 * self.epsilon, 1e-6), 0.15 * self.epsilon))
        else:
            self.eps_disc = float(eps_disc)
        self._eps_split_cfg = dict(eps_split or {"structure": 0.3, "cpt": 0.7})
        self._recompute_budget()

        # store decode flags
        self.decode_binary_as_bool = bool(decode_binary_as_bool)

        # ===== store new knobs =====
        self.temperature = float(temperature)
        if not np.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.cpt_smoothing = float(cpt_smoothing)
        if self.cpt_smoothing < 0:
            raise ValueError("cpt_smoothing must be >= 0")
        self.forbid_as_parent_set = set(forbid_as_parent or [])
        self.parent_blacklist = {k: set(v or []) for k, v in (parent_blacklist or {}).items()}
        self.numeric_bins_overrides: Dict[str, int] = dict(numeric_bins_overrides or {})
        self.integer_decode_mode = str(integer_decode_mode).lower()
        if self.integer_decode_mode not in {"round", "stochastic", "granular"}:
            raise ValueError("integer_decode_mode must be 'round', 'stochastic', or 'granular'")
        self.integer_granularity: Dict[str, int] = dict(integer_granularity or {})
        self.numeric_granularity: Dict[str, float] = dict(numeric_granularity or {})
        self.cat_topk_overrides: Dict[str, int] = dict(cat_topk_overrides or {})
        self.cat_buckets_overrides: Dict[str, int] = dict(cat_buckets_overrides or {})
        
        # categorical unknown handling
        self.unknown_token = str(unknown_token)
        self.categorical_unknown_to_nan = bool(categorical_unknown_to_nan)
        
        # label columns (no hashing, no UNK)
        self.label_columns = set(label_columns or [])

        # learned state
        self._meta: Dict[str, _ColMeta] = {}
        self._order: List[str] = []
        self._cpt: Dict[str, Dict[str, Any]] = {}

        # book-keeping
        self._dp_metadata_used_bounds: set[str] = set()
        self._dp_metadata_used_cats: set[str] = set()
        self._dp_metadata_delta_used: float = 0.0
        self._dp_metadata_eps_spent: float = 0.0
        self._dp_metadata_cols_bounds: int = 0
        self._dp_metadata_cols_cats: int = 0
        self._all_nan_columns: int = 0

        # Schema-native state — populated by load_schema()
        self._schema: Optional[Dict] = None
        self._schema_col_types: Dict[str, str] = {}
        self._schema_bin_edges: Dict[str, list] = {}
        self._schema_constraints: Dict = {}
        self._datetime_cols: set = set()
        self._n_declared: Optional[int] = None
        self._schema_provenance: Dict = {}
        self._event_col: Optional[str] = None
        self._duration_col: Optional[str] = None
        self._tau: Optional[float] = None
        self._n_fit: Optional[int] = None
        self._n_source: Optional[str] = None
        self._n_observed: Optional[int] = None
        self._schema_public_category_cols: set[str] = set()

        if schema is not None:
            self.load_schema(schema)

    def _recompute_budget(self) -> None:
        """Recompute epsilon allocation between structure, CPTs, and metadata."""
        es = getattr(self, "_eps_split_cfg", {"structure": 0.3, "cpt": 0.7})
        s = max(0.0, float(es.get("structure", 0.3)))
        c = max(0.0, float(es.get("cpt", 0.7)))
        if s + c == 0:
            s, c = 0.3, 0.7
        z = s + c

        # In schema-authoritative mode, metadata budget is disabled
        disc = float(max(self.eps_disc, 0.0))
        if self.require_public:
            disc = 0.0
        self.eps_disc = disc

        main_eps = max(self.epsilon - disc, 0.0)
        self._eps_main = main_eps
        self._eps_struct = main_eps * (s / z)
        self._eps_cpt = main_eps * (c / z)

    def load_schema(self, schema: dict) -> None:
        """
        Load a schema-generator JSON and configure the model
        to be fully schema-authoritative.
        """
        self._schema = schema
        raw_pb = schema.get("public_bounds", {})
        self.public_bounds = {}
        self._schema_bin_edges = {}
        for col, v in raw_pb.items():
            if isinstance(v, dict):
                lo = v.get("min")
                hi = v.get("max")
                if lo is not None and hi is not None:
                    self.public_bounds[col] = [float(lo), float(hi)]
                if "bins" in v and v["bins"] is not None:
                    self._schema_bin_edges[col] = list(v["bins"])
            elif isinstance(v, (list, tuple)) and len(v) == 2:
                self.public_bounds[col] = [float(v[0]), float(v[1])]

        pc = schema.get("public_categories", {})
        self.public_categories = {col: list(vals) for col, vals in pc.items()}
        self._schema_public_category_cols = set(pc.keys())
        self._schema_col_types = dict(schema.get("column_types", {}))
        for _col, _t in self._schema_col_types.items():
            if _t not in self._KNOWN_STYPES:
                raise ValueError(
                    f"Unknown schema column type '{_t}' for column '{_col}' "
                    f"in load_schema(). Known types: {sorted(self._KNOWN_STYPES)}."
                )

        ts = schema.get("target_spec", {})
        targets = ts.get("targets", [])
        primary = ts.get("primary_target")
        self._event_col = primary
        self._duration_col = next(
            (t for t in targets if t != primary), None)
        self._tau = ts.get("tau")
        auto_labels = [c for c in [primary, self._duration_col] if c is not None]
        self.label_columns = set(self.label_columns) | set(auto_labels)

        label_domain = schema.get("label_domain", [])
        if primary and label_domain:
            # Prefer explicit label_domain for the primary target. If it conflicts
            # with an existing public_categories entry, override it but emit a
            # warning rather than silently dropping one of them.
            if primary in self.public_categories:
                existing = list(self.public_categories.get(primary, []))
                if existing != list(label_domain):
                    warnings.warn(
                        f"Schema label_domain for primary target '{primary}' "
                        f"overrides public_categories['{primary}'] "
                        f"(existing={existing}, label_domain={list(label_domain)}).",
                        UserWarning,
                        stacklevel=2,
                    )
            self.public_categories[primary] = list(label_domain)

        self._datetime_cols = set(schema.get("datetime_spec", {}).keys())
        self._schema_constraints = schema.get("constraints", {})

        dataset = schema.get("dataset", {})
        # Prefer a dict-style dataset with explicit n_records; fall back to
        # an optional legacy dataset_info block when dataset is a string.
        if isinstance(dataset, dict):
            self._n_declared = dataset.get("n_records")
        else:
            info = schema.get("dataset_info", {})
            self._n_declared = info.get("n_records") if isinstance(info, dict) else None
        self._schema_provenance = schema.get("provenance", {})

        self.require_public = True
        self.dp_bounds_mode = "public"
        self.eps_disc = 0.0
        self._recompute_budget()

        print(
            f"[CRNPrivBayes] Schema loaded: "
            f"{len(self._schema_col_types)} typed cols, "
            f"{len(self.public_bounds)} bounded, "
            f"{len(self.public_categories)} categorical, "
            f"{len(self._schema_bin_edges)} with explicit bin edges. "
            f"label_columns={sorted(self.label_columns)}"
        )

    def _lap(self, eps: float, shape: Any, *, sens: Optional[float] = None) -> np.ndarray:
        """Generate Laplace noise for differential privacy.
        
        Noise scale is sensitivity/epsilon. Uses adjacency mode to determine
        default sensitivity (1.0 for unbounded, 2.0 for bounded).
        """
        base_sens = float(self._sens_count) if sens is None else float(sens)
        scale = base_sens / max(float(eps), 1e-12)
        return self._rng.laplace(0.0, scale, size=shape)

    def _build_meta(self, df: pd.DataFrame) -> None:
        """Build column metadata from schema (require_public=True) or via DP
        discovery (require_public=False).

        Schema-authoritative path (require_public=True)
        ------------------------------------------------
        Every column MUST appear in _schema_col_types.  No dtype inspection,
        no pd.unique, no pd.to_datetime heuristics — everything comes from
        the schema JSON.  Private data is only touched for quantization in
        _discretize(), never for type or domain inference here.

        Non-schema path (require_public=False)
        ---------------------------------------
        Legacy behaviour: dtype inference + DP heavy-hitters for bounds and
        categorical vocabularies.
        """
        self._dp_metadata_used_bounds.clear()
        self._dp_metadata_used_cats.clear()
        self._dp_metadata_delta_used = 0.0
        self._dp_metadata_eps_spent = 0.0
        self._dp_metadata_cols_bounds = 0
        self._dp_metadata_cols_cats = 0
        self._all_nan_columns = 0

        if self.require_public:
            self._build_meta_from_schema(df)
        else:
            self._build_meta_inferred(df)

    # ------------------------------------------------------------------ #
    #  Schema-authoritative path — zero private-data scanning             #
    # ------------------------------------------------------------------ #

    def _build_meta_from_schema(self, df: pd.DataFrame) -> None:
        """Build metadata purely from schema.  No dtype/value inspection."""
        pb  = dict(self.public_bounds or {})
        pbn = dict(self.public_binary_numeric or {})
        meta: Dict[str, _ColMeta] = {}
        unk = self.unknown_token

        # Every column in the dataframe must be declared in the schema.
        undeclared = [c for c in df.columns if c not in self._schema_col_types]
        if undeclared:
            raise ValueError(
                f"Columns {undeclared} are not declared in schema column_types. "
                "Every column must be explicitly typed when require_public=True."
            )

        for c in df.columns:
            stype = self._schema_col_types[c]  # validated in load_schema()

            # ---- datetime / timedelta: convert column in-place (caller safe: fit() uses df.copy()) ----
            if stype == "datetime":
                if pd.api.types.is_datetime64_any_dtype(df[c]):
                    df[c] = df[c].view("int64")
                else:
                    # Accept already-numeric representations, but for string-like
                    # datetime columns make an explicit parse attempt so we do not
                    # silently turn all values into NaN in _discretize().
                    if is_integer_dtype(df[c]) or is_numeric_dtype(df[c]):
                        # Already numeric (e.g., epoch seconds); leave as-is.
                        pass
                    else:
                        try:
                            parsed = pd.to_datetime(df[c])
                            df[c] = parsed.view("int64")
                        except Exception as e:
                            raise ValueError(
                                f"Column '{c}' (schema_type='datetime') could not be parsed: {e}"
                            ) from e
            elif stype == "timedelta":
                if pd.api.types.is_timedelta64_dtype(df[c]):
                    df[c] = df[c].view("int64")
                else:
                    # As with datetime, try an explicit parse for string-like timedeltas
                    # to avoid silently converting everything to NaN.
                    if is_integer_dtype(df[c]) or is_numeric_dtype(df[c]):
                        # Already numeric duration; leave as-is.
                        pass
                    else:
                        try:
                            parsed = pd.to_timedelta(df[c])
                            df[c] = parsed.view("int64")
                        except Exception as e:
                            raise ValueError(
                                f"Column '{c}' (schema_type='timedelta') could not be parsed: {e}"
                            ) from e

            # ---- NUMERIC branch: continuous / integer / binary / ordinal /
            #                      datetime / timedelta ----
            if stype in self._NUMERIC_STYPES:

                if stype == "binary":
                    L, U = 0.0, 1.0
                    pb[c] = [L, U]
                    pbn[c] = True
                else:
                    if c not in pb:
                        raise ValueError(
                            f"Column '{c}' (schema_type='{stype}') requires "
                            "public_bounds in the schema when require_public=True."
                        )
                    L, U = float(pb[c][0]), float(pb[c][1])
                    if not (np.isfinite(L) and np.isfinite(U) and U > L):
                        raise ValueError(
                            f"public_bounds for '{c}' must satisfy lo < hi and both finite "
                            f"(got [{L}, {U}])."
                        )

                binary_numeric = (stype == "binary")
                is_int         = (stype in self._INT_STYPES)
                original_dtype = np.dtype("int64") if is_int else None

                if binary_numeric:
                    k    = 2
                    bins = np.array([0.0, 0.5, 1.0], dtype=float)
                elif c in self._schema_bin_edges:
                    raw_edges = np.array(self._schema_bin_edges[c], dtype=float)
                    norm = np.clip((raw_edges - L) / max(U - L, 1e-12), 0.0, 1.0)
                    norm = np.unique(np.concatenate([[0.0], norm, [1.0]]))
                    k    = len(norm) - 1
                    bins = norm
                else:
                    k_override = self.numeric_bins_overrides.get(c)
                    k = max(2, int(k_override)) if k_override is not None else max(2, int(self.bins_per_numeric))
                    bins = np.linspace(0.0, 1.0, k + 1)

                meta[c] = _ColMeta(
                    kind="numeric",
                    k=k,
                    bins=bins,
                    cats=None,
                    is_int=is_int,
                    bounds=(L, U),
                    binary_numeric=binary_numeric,
                    original_dtype=original_dtype,
                    all_nan=False,
                    hashed_cats=False,
                    hash_m=None,
                    schema_type=stype,
                    explicit_bin_edges=(
                        np.array(self._schema_bin_edges[c], dtype=float)
                        if c in self._schema_bin_edges else None),
                )

            # ---- CATEGORICAL branch: categorical ----
            else:  # stype == "categorical"

                if c in self.label_columns:
                    pub = list(self.public_categories.get(c, []) or [])
                    if not pub:
                        raise ValueError(
                            f"Label column '{c}' requires public_categories['{c}'] "
                            "in the schema (require_public=True)."
                        )
                    cats = [x for x in pub if x != unk]
                    if len(cats) < 2:
                        warnings.warn(
                            f"Label column '{c}' has <2 classes after filtering.",
                            stacklevel=1,
                        )
                    meta[c] = _ColMeta(
                        kind="categorical", k=len(cats), cats=cats,
                        hashed_cats=False, hash_m=None,
                        schema_type=stype, explicit_bin_edges=None,
                    )
                    self.public_categories[c] = cats
                    continue

                if c not in self.public_categories:
                    raise ValueError(
                        f"Categorical column '{c}' requires public_categories['{c}'] "
                        "in the schema (require_public=True)."
                    )
                pub = list(self.public_categories[c] or [])
                if not pub:
                    raise ValueError(
                        f"public_categories['{c}'] is empty — provide at least one value."
                    )

                cats = [x for x in pub if x != unk]
                if not cats:
                    cats = list(pub)

                meta[c] = _ColMeta(
                    kind="categorical", k=len(cats), cats=cats,
                    hashed_cats=False, hash_m=None,
                    schema_type=stype, explicit_bin_edges=None,
                )
                self.public_categories[c] = cats

        self._meta              = meta
        self.public_bounds      = pb
        self.public_binary_numeric = pbn

    # ------------------------------------------------------------------ #
    #  Non-schema path — legacy dtype-inference + DP discovery            #
    # ------------------------------------------------------------------ #

    def _build_meta_inferred(self, df: pd.DataFrame) -> None:
        """Legacy path when require_public=False: infer types from dtype and
        use DP heavy-hitters for unknown bounds / categorical domains."""
        pb  = dict(self.public_bounds or {})
        pc  = {k: list(v or []) for k, v in (self.public_categories or {}).items()}
        pbn = dict(self.public_binary_numeric or {})

        m_cols_need_bounds: List[str] = []
        m_cols_need_cats: List[str] = []

        if self.eps_disc > 0.0:
            for c in df.columns:
                if is_numeric_dtype(df[c]) and c not in pb:
                    m_cols_need_bounds.append(c)
                elif (not is_numeric_dtype(df[c])) and not pc.get(c):
                    m_cols_need_cats.append(c)

        m_total = len(m_cols_need_bounds) + len(m_cols_need_cats)

        if self.strict_dp and m_total > 0 and self.eps_disc <= 0.0:
            raise ValueError(
                "DP metadata required, but eps_disc=0. "
                "Provide public bounds/categories or set a positive eps_disc."
            )

        eps_disc_per_col   = (self.eps_disc / m_total) if m_total > 0 else 0.0
        smooth_cols        = [c for c in m_cols_need_bounds
                              if eps_disc_per_col > 0.0 and self.dp_bounds_mode == "smooth"]
        n_smooth           = len(smooth_cols)
        delta_per_smooth   = (self.delta / n_smooth) if n_smooth > 0 else 0.0

        # Ledger: effective eps/delta spent on metadata (sum over columns)
        self._dp_metadata_eps_spent = eps_disc_per_col * (len(m_cols_need_bounds) + len(m_cols_need_cats))
        self._dp_metadata_cols_bounds = len(m_cols_need_bounds)
        self._dp_metadata_cols_cats = len(m_cols_need_cats)
        # Delta: we pass delta_total=delta_per_smooth per smooth column; total = n_smooth * delta_per_smooth
        self._dp_metadata_delta_used = n_smooth * max(delta_per_smooth, 1e-15)

        meta: Dict[str, _ColMeta] = {}
        unk = self.unknown_token

        for c in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[c]):
                df[c] = df[c].view("int64")
            elif pd.api.types.is_timedelta64_dtype(df[c]):
                df[c] = df[c].view("int64")
            elif df[c].dtype == "object":
                try:
                    dt_parsed = pd.to_datetime(df[c], errors="coerce")
                    if dt_parsed.notna().mean() >= 0.95:
                        df[c] = dt_parsed.astype("int64")
                except (ValueError, TypeError, OverflowError):
                    pass

            _stype = getattr(self, "_schema_col_types", {}).get(c)
            is_bool_col  = is_bool_dtype(df[c])
            is_num_col   = is_numeric_dtype(df[c])

            if is_bool_col and c not in pb:
                pb[c]  = [0.0, 1.0]
                pbn[c] = True

            if is_num_col or is_bool_col or (c in pb):
                raw          = pd.to_numeric(df[c], errors="coerce").to_numpy()
                all_nan_flag = False

                if c in pb and isinstance(pb[c], (list, tuple)) and len(pb[c]) == 2:
                    L, U = pb[c]
                    if not (np.isfinite(L) and np.isfinite(U) and U > L):
                        x = raw[np.isfinite(raw)]
                        if x.size == 0:
                            L, U = 0.0, 1.0
                            self._all_nan_columns += 1
                            all_nan_flag = True
                        else:
                            L, U = float(np.min(x)), float(np.max(x))
                        pb[c] = [L, U]
                else:
                    if eps_disc_per_col > 0.0:
                        coarse = None
                        for key in ("*", "__all__", "__global__"):
                            if key in pb and isinstance(pb[key], (list, tuple)) and len(pb[key]) == 2:
                                L, U = pb[key]
                                if np.isfinite(L) and np.isfinite(U) and U > L:
                                    coarse = (float(L), float(U))
                                    break
                        if self.dp_bounds_mode == "public" and coarse is not None:
                            L, U = _dp_numeric_bounds_public(
                                pd.Series(raw),
                                eps_min=eps_disc_per_col * 0.5,
                                eps_max=eps_disc_per_col * 0.5,
                                coarse_bounds=coarse,
                                rng=self._rng
                            )
                            self._dp_metadata_used_bounds.add(c)
                        else:
                            L, U = _dp_numeric_bounds_smooth(
                                pd.Series(raw),
                                eps_total=eps_disc_per_col,
                                delta_total=max(delta_per_smooth, 1e-15),
                                alpha=self.dp_quantile_alpha,
                                rng=self._rng,
                                public_coarse=coarse
                            )
                            self._dp_metadata_used_bounds.add(c)
                        pb[c] = [float(L), float(U)]
                    else:
                        if self.strict_dp:
                            raise ValueError(
                                f"DP bounds required for column '{c}' but eps_disc_per_col=0 under strict_dp."
                            )
                        x = raw[np.isfinite(raw)]
                        if x.size == 0:
                            L, U = 0.0, 1.0
                            self._all_nan_columns += 1
                            all_nan_flag = True
                        else:
                            L, U = float(np.min(x)), float(np.max(x))
                        pb[c] = [float(L), float(U)]

                L, U  = float(pb[c][0]), float(pb[c][1])
                is_int = (_stype == "integer" or _stype == "ordinal") or is_integer_dtype(df[c])
                original_dtype = None
                if is_int:
                    try:
                        original_dtype = df[c].to_numpy(copy=False).dtype
                    except Exception:
                        original_dtype = np.dtype("int64")

                vals_num = pd.to_numeric(df[c], errors="coerce")
                u        = pd.unique(vals_num.dropna())
                try:
                    binary_numeric = (
                        len(u) <= 2 and
                        set([0.0, 1.0]).issuperset(set(pd.Series(u).astype(float))))
                except Exception:
                    binary_numeric = False
                binary_numeric = binary_numeric or is_bool_col

                if binary_numeric:
                    k = 2
                    bins = np.array([0.0, 0.5, 1.0], dtype=float)
                else:
                    k_override = self.numeric_bins_overrides.get(c)
                    k = max(2, int(k_override)) if k_override is not None else max(2, int(self.bins_per_numeric))
                    if c in getattr(self, "_schema_bin_edges", {}):
                        raw_edges = np.array(self._schema_bin_edges[c], dtype=float)
                        norm = np.clip((raw_edges - L) / max(U - L, 1e-12), 0.0, 1.0)
                        norm = np.unique(np.concatenate([[0.0], norm, [1.0]]))
                        k = len(norm) - 1
                        bins = norm
                    else:
                        bins = np.linspace(0.0, 1.0, k + 1)

                meta[c] = _ColMeta(
                    kind="numeric",
                    k=k,
                    bins=bins,
                    cats=None,
                    is_int=bool(is_int),
                    bounds=(L, U),
                    binary_numeric=bool(binary_numeric),
                    original_dtype=original_dtype,
                    all_nan=all_nan_flag,
                    hashed_cats=False,
                    hash_m=None,
                    schema_type=_stype,
                    explicit_bin_edges=(
                        np.array(self._schema_bin_edges[c], dtype=float)
                        if c in getattr(self, "_schema_bin_edges", {}) else None),
                )
            else:
                if c in self.label_columns:
                    pub = list(self.public_categories.get(c, []) or [])
                    if not pub:
                        if self.strict_dp:
                            raise ValueError(
                                f"Label column '{c}' requires public_categories['{c}'] under strict_dp."
                            )
                        vals_s = pd.Series(df[c], copy=False).astype("string").dropna().unique().tolist()
                        pub = sorted([str(v) for v in vals_s])
                    cats = [x for x in pub if x != unk]
                    if len(cats) < 2:
                        warnings.warn(f"Label column '{c}' has <2 classes.", stacklevel=1)
                    meta[c] = _ColMeta(
                        kind="categorical", k=len(cats), cats=cats,
                        hashed_cats=False, hash_m=None,
                        schema_type=_stype, explicit_bin_edges=None,
                    )
                    self.public_categories[c] = cats
                    continue

                pub = list(self.public_categories.get(c, []) or [])
                hashed = False
                hash_m = None
                if pub:
                    # Always inject __UNK__ for non-label categoricals so unseen maps somewhere safe.
                    cats = [unk] + [x for x in pub if x != unk]
                    if len(cats) == 1 and len(pub) > 0:
                        cats = list(pub)  # pub was only unk; keep as-is
                elif eps_disc_per_col > 0.0:
                    ser = pd.Series(df[c], copy=False).astype("string")
                    m_sz = max(8, int(self.cat_buckets_overrides.get(c, self.cat_buckets)))
                    buckets = ser.fillna(unk).map(lambda v: f"B{_blake_bucket(str(v), m_sz):03d}")
                    counts = buckets.value_counts(dropna=False).to_dict()
                    # Ensure the DP universe covers all hash buckets B000..B(m_sz-1),
                    # not just those observed in the private data.
                    for idx in range(m_sz):
                        key = f"B{idx:03d}"
                        counts.setdefault(key, 0.0)
                    eps_col = max(eps_disc_per_col, 1e-12)
                    noisy = {b: float(cnt) + float(self._lap(eps_col, (), sens=None)) for b, cnt in counts.items()}
                    order = sorted(noisy.keys(), key=lambda t: noisy[t], reverse=True)
                    if self.cat_keep_all_nonzero:
                        # Threshold noisy buckets by a scale-aware tau so that
                        # empty buckets are unlikely to pass when epsilon is small.
                        sens = float(self._sens_count)
                        tau = (sens / eps_col) * float(np.log(max(m_sz, 2)))
                        topk = [b for b in order if noisy[b] > tau]
                    else:
                        k_def = int(self.cat_topk) if self.cat_topk is not None else 28
                        K = max(8, int(min(self.cat_topk_overrides.get(c, k_def), len(order))))
                        topk = order[:K]
                    cats = [unk] + topk
                    hashed = True
                    hash_m = m_sz
                    self._dp_metadata_used_cats.add(c)
                else:
                    if self.strict_dp:
                        raise ValueError(
                            f"DP categorical discovery required for '{c}' but eps_disc_per_col=0 under strict_dp."
                        )
                    warnings.warn(
                        "Non-DP categorical discovery used due to eps_disc=0 and strict_dp=False. "
                        "This is NOT differentially private.",
                        stacklevel=1,
                    )
                    vals_s = pd.Series(df[c], copy=False).astype("string").dropna().unique().tolist()
                    cats = ([unk] if unk not in pub else []) + [x for x in vals_s if x != unk]

                meta[c] = _ColMeta(
                    kind="categorical",
                    k=len(cats),
                    cats=cats,
                    hashed_cats=hashed,
                    hash_m=hash_m,
                    schema_type=_stype,
                    explicit_bin_edges=None,
                )
                self.public_categories[c] = cats

        self._meta                 = meta
        self.public_bounds         = pb
        self.public_binary_numeric = pbn

    def _discretize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert dataframe to integer codes using learned metadata.
        
        Numeric columns: normalize to [0,1], bin using learned bins, map to integer codes.
        Categorical columns: map values to category indices. For hashed categoricals,
        apply hash bucket mapping first. Label columns skip hashing and unknown tokens.
        """
        out: Dict[str, np.ndarray] = {}
        for c, m in self._meta.items():
            if m.kind == "numeric":
                lo, hi = m.bounds if m.bounds is not None else (0.0, 1.0)
                x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
                stype = getattr(m, "schema_type", None)

                if self.require_public and stype:
                    _validate_schema_numeric_column(
                        orig_col=df[c],
                        x_num=x,
                        colname=c,
                        stype=stype,
                        int_stypes=set(self._INT_STYPES),
                        missing_string_sentinels=getattr(self, "missing_string_sentinels", None),
                        reveal_row_index_in_errors=getattr(self, "reveal_row_index_in_errors", False),
                    )

                z = (x - lo) / max(hi - lo, 1e-12)
                z = np.where(np.isfinite(z), z, 0.5)
                z = np.clip(z, 0.0, 1.0)
                idx = np.digitize(z, m.bins, right=False) - 1
                idx = np.clip(idx, 0, m.k - 1)
                out[c] = idx.astype(int, copy=False)
            else:
                # Categorical: keep unknowns as token (no NaNs)
                unk = getattr(self, "unknown_token", "__UNK__")
                cats = list(m.cats or [])

                col = df[c].astype("string").fillna(
                    unk if c not in self.label_columns else (cats[0] if cats else unk)
                )

                # Never hash labels
                if getattr(m, "hashed_cats", False) and m.hash_m and (c not in self.label_columns):
                    msize = int(m.hash_m)
                    col = col.map(lambda v: unk if v == unk else f"B{_blake_bucket(str(v), msize):03d}")

                cat = pd.Categorical(col, categories=cats, ordered=False)
                codes = np.asarray(cat.codes, dtype=int)

                if self.require_public and c not in self.label_columns:
                    # Closed vocabulary from schema: unseen values must either map
                    # to an explicit __UNK__ token or trigger an error.
                    if np.any(codes < 0):
                        bad_mask = codes < 0
                        n_bad = int(np.sum(bad_mask))
                        if unk in cats:
                            unk_idx = cats.index(unk)
                            codes = np.where(bad_mask, unk_idx, codes)
                        else:
                            raise ValueError(
                                f"Column '{c}': some values are not in schema public_categories "
                                f"and no __UNK__ token is defined (n_unseen={n_bad})."
                            )
                else:
                    # Non-schema path: map unseen to __UNK__ index when present,
                    # else strict_dp → error, else legacy fallback to index 0 (assumes cats[0]==unk).
                    if np.any(codes < 0):
                        bad_mask = codes < 0
                        n_bad = int(np.sum(bad_mask))
                        if unk in cats:
                            unk_idx = cats.index(unk)
                            codes = np.where(bad_mask, unk_idx, codes)
                        elif self.strict_dp:
                            raise ValueError(
                                f"Column '{c}': unseen values not in categories and no __UNK__ present "
                                f"(n_unseen={n_bad})."
                            )
                        else:
                            # Legacy: assume cats[0] is __UNK__ when present
                            codes = np.where(bad_mask, 0, codes)

                out[c] = codes
        return pd.DataFrame(out, index=df.index)

    def fit(self, df: pd.DataFrame, *,
            schema: Optional[Union[Dict, str]] = None,
            config: Optional[Dict[str, Any]] = None) -> None:
        """Fit PrivBayes model: learn structure and conditional probability tables.
        
        Process: (1) build metadata, (2) discretize data, (3) compute DP mutual information
        scores for all pairs, (4) select parents greedily, (5) estimate noisy CPTs with
        Laplace mechanism. Epsilon split between structure learning and CPT estimation.
        """
        if schema is not None:
            if isinstance(schema, dict):
                self.load_schema(schema)
            elif isinstance(schema, str):
                import json as _json
                with open(schema) as _f:
                    self.load_schema(_json.load(_f))
        cfg = config or {}
        kw = dict(cfg.get("kwargs", {}))

        if "eps_split" in kw:
            self._eps_split_cfg = dict(kw["eps_split"] or {})

        if "bins_per_numeric" in kw:
            self.bins_per_numeric = int(kw["bins_per_numeric"])
        if "require_public" in kw:
            self.require_public = bool(kw["require_public"])
        if "strict_dp" in kw:
            self.strict_dp = bool(kw["strict_dp"])

        if self.require_public:
            # Schema-authoritative mode never spends epsilon on metadata
            self.eps_disc = 0.0
        self._recompute_budget()

        # In schema-authoritative mode, guard against leaking private min/max
        # through original_data_bounds. Either it must be empty or exactly
        # match the schema's public_bounds for the same columns.
        if self.require_public and self.original_data_bounds:
            for col, ob in self.original_data_bounds.items():
                if col not in self.public_bounds:
                    raise ValueError(
                        f"original_data_bounds specifies column '{col}' "
                        "which is not present in schema public_bounds "
                        "when require_public=True."
                    )
                if not isinstance(ob, (list, tuple)) or len(ob) != 2:
                    raise ValueError(
                        f"original_data_bounds[{col!r}] must be a [lo, hi] pair."
                    )
                orig_lo, orig_hi = float(ob[0]), float(ob[1])
                sch_lo, sch_hi = map(float, self.public_bounds[col])
                if not (np.isclose(orig_lo, sch_lo, rtol=1e-9, atol=1e-9)
                        and np.isclose(orig_hi, sch_hi, rtol=1e-9, atol=1e-9)):
                    raise ValueError(
                        "original_data_bounds may encode private min/max; "
                        f"for schema-authoritative mode, bounds for column '{col}' "
                        "must either be omitted or exactly match schema public_bounds."
                    )

        # In schema-authoritative mode, a schema must be loaded up front.
        if self.require_public and self._schema is None:
            raise ValueError(
                "Schema must be loaded (schema=... or load_schema()) "
                "before fit() when require_public=True."
            )

        # Work on a defensive copy so that in-place type coercions (e.g., datetime→int)
        # do not mutate the caller's dataframe.
        df = df.copy()
        n_observed = len(df)
        # In schema-authoritative mode we avoid retaining the private observed
        # count on the instance; it is only used locally for the tolerance
        # check below. In non-schema mode we record it for reporting.
        if self.require_public:
            self._n_observed = None
        else:
            self._n_observed = int(n_observed)

        if self.require_public:
            # In schema-authoritative mode, dataset size is treated as public metadata
            # and must match the declared schema. Observed len(df) is not exported
            # to avoid unintended disclosure.
            if self._n_declared is None:
                raise ValueError("Schema must declare n_records when require_public=True.")
            # Allow small relative differences plus at least 1-row slack to avoid
            # brittleness on small datasets.
            tol = max(0.001 * self._n_declared, 1.0)
            diff = abs(n_observed - self._n_declared)
            if diff > tol:
                raise ValueError(
                    "Schema n_records does not match provided dataframe length "
                    "(mismatch exceeds tolerance; check schema n_records declaration)."
                )
            self._n_fit = int(self._n_declared)
            self._n_source = "schema"
        else:
            self._n_fit = int(n_observed)
            self._n_source = "observed"

        self._build_meta(df)
        disc = self._discretize(df)
        cols = list(disc.columns)
        self._order = cols[:]

        parents: Dict[str, List[str]] = {c: [] for c in cols}
        dp_mi_scores: Dict[Tuple[str, str], float] = {}

        if self._eps_struct > 0:
            pair_info = []
            for j, c in enumerate(cols):
                for p in cols[:j]:
                    pair_info.append((c, p))
            n_pairs = len(pair_info)
            if n_pairs > 0:
                eps_per_pair = self._eps_struct / n_pairs
                for c, p in pair_info:
                    x = disc[c].to_numpy()
                    y = disc[p].to_numpy()
                    kx = self._meta[c].k
                    ky = self._meta[p].k
                    joint = np.zeros((kx, ky), dtype=float)
                    np.add.at(joint, (x, y), 1.0)
                    joint += self._lap(eps_per_pair, joint.shape, sens=None)
                    joint = np.maximum(joint, 0.0) + SMOOTH
                    pxy = joint / joint.sum()
                    px = pxy.sum(axis=1, keepdims=True)
                    py = pxy.sum(axis=0, keepdims=True)
                    denom = (px @ py)
                    ratio = np.divide(pxy, denom, out=np.ones_like(pxy), where=denom > 0)
                    mi = float(max(0.0, (pxy * np.log(ratio)).sum()))
                    dp_mi_scores[(c, p)] = mi
                    dp_mi_scores[(p, c)] = mi

            for j, c in enumerate(cols):
                cand = [p for p in cols[:j] if p not in self.forbid_as_parent_set
                        and p not in self.parent_blacklist.get(c, set())]
                if not cand:
                    continue
                scores = [(dp_mi_scores.get((c, p), 0.0), p) for p in cand]
                scores.sort(key=lambda t: (-t[0], t[1]))
                parents[c] = [p for _, p in scores[: self.max_parents]]

        self._cpt = {}
        n_vars = len(cols)
        eps_per_var = (self._eps_cpt / n_vars) if (self._eps_cpt > 0 and n_vars > 0) else 0.0

        for c in cols:
            k_child = self._meta[c].k
            pa = parents[c]
            if len(pa) == 0:
                counts = np.bincount(disc[c].to_numpy(), minlength=k_child).astype(float)
                if eps_per_var > 0:
                    counts += self._lap(eps_per_var, counts.shape, sens=None)
                # Apply smoothing to DP-noisy counts
                counts = np.maximum(counts, 0.0)
                counts += self.cpt_smoothing
                probs = (counts / counts.sum().clip(min=1e-12)).reshape(1, k_child).astype(self.cpt_dtype)
                self._cpt[c] = {"parents": [], "parent_card": [], "probs": probs}
            else:
                par_ks = [self._meta[p].k for p in pa]
                S = int(np.prod(par_ks, dtype=object))
                max_cells = int(2_000_000)
                while S * k_child > max_cells and len(pa) > 0:
                    pa = pa[:-1]
                    par_ks = [self._meta[p].k for p in pa]
                    S = int(np.prod(par_ks, dtype=object))
                if S * k_child > max_cells:
                    raise MemoryError(f"CPT for {c} too large after pruning.")
                if len(pa) == 0:
                    counts = np.bincount(disc[c].to_numpy(), minlength=k_child).astype(float)
                    if eps_per_var > 0:
                        counts += self._lap(eps_per_var, counts.shape, sens=None)
                    # Apply smoothing to DP-noisy counts
                    counts = np.maximum(counts, 0.0)
                    counts += self.cpt_smoothing
                    probs = (counts / counts.sum().clip(min=1e-12)).reshape(1, k_child).astype(self.cpt_dtype)
                    self._cpt[c] = {"parents": [], "parent_card": [], "probs": probs}
                    continue
                counts = np.zeros((S, k_child), dtype=float)
                pa_codes = np.stack([disc[p].to_numpy(dtype=np.int64, copy=False) for p in pa], axis=0)
                keys = np.ravel_multi_index(pa_codes, dims=tuple(par_ks), mode="raise")
                child = disc[c].to_numpy()
                np.add.at(counts, (keys, child), 1.0)
                if eps_per_var > 0:
                    counts += self._lap(eps_per_var, counts.shape, sens=None)
                row_sums = counts.sum(axis=1, keepdims=True)
                deg = (row_sums <= 1e-12).flatten()
                if np.any(deg):
                    counts[deg, :] = 1.0
                # Apply smoothing to DP-noisy counts
                counts = np.maximum(counts, 0.0)
                counts += self.cpt_smoothing
                probs = (counts / counts.sum(axis=1, keepdims=True).clip(min=1e-12)).astype(self.cpt_dtype)
                self._cpt[c] = {"parents": pa, "parent_card": par_ks, "probs": probs}

    def _apply_temperature(self, probs: np.ndarray) -> np.ndarray:
        """Flatten probability distribution using temperature scaling.
        
        Temperature > 1 makes distribution more uniform, reducing linkage risk.
        Formula: p' = p^(1/T) / Z where Z is normalization constant.
        """
        T = float(self.temperature)
        if not np.isfinite(T) or T <= 0:
            T = 1.0
        if abs(T - 1.0) < 1e-12:
            return probs
        p = np.clip(probs, 1e-12, 1.0) ** (1.0 / T)
        Z = p.sum(axis=1, keepdims=True)
        out = np.divide(p, Z, out=np.full_like(p, 1.0 / p.shape[1]), where=(Z > 0))
        return out

    def _sample_categorical_rows(self, probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Sample categorical values from probability matrix using inverse CDF.
        
        Applies temperature scaling before sampling. Uses cumulative distribution
        function with uniform random draws for each row.
        """
        n, _ = probs.shape
        probs = np.clip(probs, 1e-12, 1.0)
        probs = probs / probs.sum(axis=1, keepdims=True)
        probs = self._apply_temperature(probs)
        cdf = np.cumsum(probs, axis=1)
        r = np.minimum(rng.random(n), np.nextafter(1.0, 0.0))
        return (cdf >= r[:, None]).argmax(axis=1).astype(int, copy=False)

    def sample(self, n: int, seed: Optional[int] = None) -> pd.DataFrame:
        """Generate synthetic data by ancestral sampling from Bayesian network.
        
        Samples variables in topological order, conditioning on parent values.
        Returns decoded dataframe with original dtypes restored.
        """
        # Sampling uses a separate RNG from DP noise. If `seed` is provided
        # here, it takes precedence; otherwise we fall back to the constructor
        # seed, and finally to an OS-random seed. This affects only synthetic
        # sampling, not the DP guarantees of fit().
        if seed is not None:
            eff_seed = int(seed)
        elif getattr(self, "seed", None) is not None:
            eff_seed = int(self.seed)  # constructor-provided sampling seed
        else:
            eff_seed = None
        rng = np.random.default_rng(eff_seed) if eff_seed is not None else np.random.default_rng()
        if not self._cpt or not self._meta or not self._order:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        codes: Dict[str, np.ndarray] = {}
        for c in self._order:
            info = self._cpt[c]
            pa = info["parents"]
            probs = info["probs"]
            if len(pa) == 0:
                row_probs = np.repeat(probs, n, axis=0)
                picks = self._sample_categorical_rows(row_probs, rng)
            else:
                par_ks = info["parent_card"]
                pa_mat = np.stack([codes[p].astype(np.int64, copy=False) for p in pa], axis=0)
                keys = np.ravel_multi_index(pa_mat, dims=tuple(par_ks), mode="raise")
                row_probs = probs[keys]
                picks = self._sample_categorical_rows(row_probs, rng)
            codes[c] = picks
        df = self._decode(codes, n, rng)
        return self._clip_to_schema_constraints(df)

    def _clip_to_schema_constraints(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enforce schema column_constraints (min_exclusive, min, max, max_exclusive) on numeric columns."""
        schema = getattr(self, "_schema", None)
        if schema is None:
            return df
        pb = schema.get("public_bounds", {})
        ct = schema.get("column_types", {})
        col_c = (getattr(self, "_schema_constraints") or {}).get("column_constraints", {})
        for col in list(df.columns):
            if ct.get(col) not in ("continuous", "integer"):
                continue
            bv = pb.get(col)
            if bv is None:
                continue
            pb_lo = bv.get("min") if isinstance(bv, dict) else (bv[0] if isinstance(bv, (list, tuple)) else None)
            pb_hi = bv.get("max") if isinstance(bv, dict) else (bv[1] if isinstance(bv, (list, tuple)) else None)
            col_spec = col_c.get(col)
            lo, hi = _constraint_aware_clip_bounds(pb_lo, pb_hi, col_spec, ct.get(col))
            if lo is not None and hi is not None and lo <= hi:
                df[col] = pd.to_numeric(df[col], errors="coerce").clip(float(lo), float(hi))
        return df

    def _decode(self, codes: Dict[str, np.ndarray], n: int, rng: np.random.Generator) -> pd.DataFrame:
        """Convert integer codes back to original data types and value ranges.
        
        Numeric: uniform random within bins, scaled to original range. Integers use
        decode mode (round/stochastic/granular). Datetime columns decoded as numeric.
        Categorical: map codes to category strings. Binary numerics threshold at midpoint.
        """
        out: Dict[str, np.ndarray] = {}
        for c in self._order:
            m = self._meta[c]
            z = codes[c]
            if m.kind == "numeric":
                lo, hi = m.bounds if m.bounds is not None else (0.0, 1.0)
                left = m.bins[z]
                right = m.bins[np.minimum(z + 1, m.k)]
                u = rng.random(n)
                val01 = left + (right - left) * u
                val = lo + val01 * (hi - lo)
                # Clip to original data bounds if provided
                # WARNING: original_data_bounds reveals exact data range and is NOT DP-compliant
                # Only use if bounds are public knowledge (e.g., age is always 0-120)
                # For DP compliance, set original_data_bounds=None and let DP bounds handle it
                if c in (self.original_data_bounds or {}):
                    orig_lo, orig_hi = self.original_data_bounds[c]
                    if orig_lo is not None and orig_hi is not None:
                        val = np.clip(val, orig_lo, orig_hi)
                if m.binary_numeric:
                    if getattr(self, "decode_binary_as_bool", False):
                        val = (val >= (lo + (hi - lo) * 0.5))
                    else:
                        val = (val >= (lo + (hi - lo) * 0.5)).astype(int)
                elif m.is_int:
                    mode = self.integer_decode_mode
                    if mode == "stochastic":
                        base = np.floor(val)
                        frac = val - base
                        draw = rng.random(n)
                        val = base + (draw < frac).astype(float)
                    elif mode == "granular":
                        g = int(self.integer_granularity.get(c, 1))
                        g = max(1, g)
                        val = np.round(val / g) * g
                    else:
                        val = np.rint(val)
                    if m.original_dtype is not None:
                        info = np.iinfo(m.original_dtype) if m.original_dtype.kind in ("i", "u") else None
                        if info is not None:
                            val = np.clip(val, info.min, info.max)
                        val = val.astype(m.original_dtype)
                    else:
                        val = val.astype(int)
                else:
                    step = float(self.numeric_granularity.get(c, 0.0)) if hasattr(self, "numeric_granularity") else 0.0
                    if np.isfinite(step) and step > 0:
                        val = np.round(val / step) * step
                    val = val.astype(float)
                out[c] = val
            else:
                # Categorical: keep unknowns as token (no NaNs)
                unk = getattr(self, "unknown_token", "__UNK__")
                cats = m.cats or [unk]
                z = np.clip(z, 0, len(cats) - 1)
                vals = np.array(cats, dtype=object)[z]
                
                # Legacy behavior (off by default)
                if getattr(self, "categorical_unknown_to_nan", False):
                    vals = np.where(vals == unk, np.nan, vals)
                
                out[c] = vals
        return pd.DataFrame(out, columns=self._order)

    @property
    def parents_(self) -> Dict[str, List[str]]:
        """Return learned Bayesian network structure: mapping from child to parent columns."""
        if not self._cpt:
            raise RuntimeError("Model is not fitted.")
        return {c: list(self._cpt[c]["parents"]) for c in self._order}

    def validate_output(self, synth_df) -> dict:
        """Check synthetic output against schema constraints."""
        if not self._schema_constraints:
            return {
                "schema_loaded": self._schema is not None,
                "overall_violation_rate": None,
                "n_records_total": len(synth_df),
                "n_records_with_any_violation": None,
                "column_constraints": {},
                "cross_column_constraints": {},
            }
        n = len(synth_df)
        violations = pd.Series(False, index=synth_df.index)
        col_results = {}
        cross_results = {}

        for col, spec in self._schema_constraints.get(
                "column_constraints", {}).items():
            if col not in synth_df.columns:
                continue
            mask = pd.Series(False, index=synth_df.index)
            vals = pd.to_numeric(synth_df[col], errors="coerce")
            if "min_exclusive" in spec and spec["min_exclusive"] is not None:
                mask |= vals <= spec["min_exclusive"]
            if "min" in spec and spec["min"] is not None:
                mask |= vals < spec["min"]
            if "max" in spec and spec["max"] is not None:
                mask |= vals > spec["max"]
            col_results[col] = float(mask.mean())
            violations |= mask

        for constraint in self._schema_constraints.get(
                "cross_column_constraints", []):
            ctype = constraint.get("type")
            name  = constraint.get("name", ctype)
            if ctype == "survival_pair":
                ec = constraint.get("event_col")
                tc = constraint.get("time_col")
                allowed = constraint.get("event_allowed_values", [0, 1])
                if ec in synth_df.columns and tc in synth_df.columns:
                    mask = (
                        ~synth_df[ec].isin(allowed) |
                        (pd.to_numeric(synth_df[tc],
                                       errors="coerce") <= 0))
                    cross_results[name] = float(mask.mean())
                    violations |= mask

        return {
            "schema_loaded": True,
            "overall_violation_rate": float(violations.mean()),
            "n_records_total": n,
            "n_records_with_any_violation": int(violations.sum()),
            "column_constraints": col_results,
            "cross_column_constraints": cross_results,
        }

    def privacy_report(self) -> Dict[str, Any]:
        """Return privacy accounting: epsilon allocation, mechanism type, metadata usage.
        
        Reports actual epsilon consumption across structure learning, CPT estimation,
        and metadata generation. Indicates whether (ε,δ)-DP was used for bounds.
        """
        eps_struct = float(getattr(self, "_eps_struct", 0.0))
        eps_cpt = float(getattr(self, "_eps_cpt", 0.0))
        eps_main = float(getattr(self, "_eps_main", 0.0))
        eps_disc_cfg = float(getattr(self, "eps_disc", 0.0))
        schema_authoritative = bool(self.require_public and self._schema is not None)
        used_bounds = len(self._dp_metadata_used_bounds) > 0
        used_cats = len(self._dp_metadata_used_cats) > 0
        metadata_dp_used = bool(used_bounds or used_cats)
        # Effective eps spent on metadata (0 in schema mode; else ledger value)
        eps_disc_used = 0.0 if schema_authoritative else float(getattr(self, "_dp_metadata_eps_spent", 0.0))
        eps_disc_effective = 0.0 if schema_authoritative else eps_disc_used
        eps_actual = eps_struct + eps_cpt + eps_disc_effective
        mech = "pure"
        delta_used = float(self._dp_metadata_delta_used) if used_bounds and self.dp_bounds_mode == "smooth" else 0.0
        if used_bounds and self.dp_bounds_mode == "smooth":
            mech = "(ε,δ)-DP"
        report = {
            "mechanism": mech,
            "schema_authoritative": schema_authoritative,
            "epsilon": float(self.epsilon),
            "delta": float(self.delta),
            "eps_main": eps_main,
            "eps_struct": eps_struct,
            "eps_cpt": eps_cpt,
            # Report the theoretical number of variable pairs only when structure
            # learning actually used a positive epsilon budget; otherwise no MI
            # queries were made.
            "n_pairs": int((len(self._order) * (len(self._order) - 1)) // 2) if eps_struct > 0.0 else 0,
            "n_vars": int(len(self._order)),
            "eps_disc_configured": eps_disc_cfg,
            "eps_disc_used": eps_disc_used,
            "eps_disc_effective": eps_disc_effective,
            "eps_disc": eps_disc_effective,  # for ledger: 0.0 in schema-authoritative, else spent
            "epsilon_total_configured": float(self.epsilon),
            "epsilon_total_actual": eps_actual,
            "delta_used": delta_used,
            "count_sensitivity_used_in_fit": float(self._sens_count),
            "laplace_default_sensitivity": float(self._sens_count),
            "laplace_default_adjacency": self.adjacency,
            # Histogram-style mechanisms that use the Laplace count sensitivity:
            # - DP MI histograms for structure learning
            # - DP CPT histograms for conditional probabilities
            # - DP categorical heavy hitters for private domains
            "laplace_sensitivity_applies_to": [
                "structure_mi_histograms",
                "cpt_histograms",
                "categorical_heavy_hitters",
            ],
            "metadata_dp": metadata_dp_used,
            "metadata_mode": ("public" if self.require_public else ("dp_bounds_" + self.dp_bounds_mode)),
            "max_parents": int(self.max_parents),
            "temperature": float(self.temperature),
        }
        n_declared = self._n_declared
        n_fit = (
            n_declared
            if getattr(self, "_n_source", None) == "schema"
            else self._n_fit
        )
        # In schema-authoritative mode, do not report the private observed count.
        if schema_authoritative:
            n_observed = None
        else:
            n_observed = getattr(self, "_n_observed", None)
        # Reuse the same tolerance rule as fit(): max(0.1%·n, 1 row) when both
        # declared and observed are available; otherwise fall back to exact match.
        if schema_authoritative:
            # Tolerance has already been enforced in fit() and we intentionally
            # hide the private observed count. For reporting, treat the match
            # status as True whenever schema mode is active.
            n_match = True
        else:
            if n_declared is not None and n_observed is not None:
                tol_n = max(0.001 * n_declared, 1.0)
                diff_n = abs(n_observed - n_declared)
                n_match = bool(diff_n <= tol_n)
            elif n_declared is not None and n_fit is not None:
                n_match = bool(n_declared == n_fit)
            else:
                n_match = None

        report["schema"] = {
            "loaded":               self._schema is not None,
            "n_declared":           n_declared,
            "n_fit":                n_fit,
            "n_observed":           n_observed,
            "n_source":             getattr(self, "_n_source", None),
            "n_match":              n_match,
            "schema_driven_cols":   sum(
                1 for m in self._meta.values()
                if m.schema_type is not None),
            "inferred_cols":        sum(
                1 for m in self._meta.values()
                if m.schema_type is None),
            "explicit_bin_edge_cols": [
                c for c, m in self._meta.items()
                if m.explicit_bin_edges is not None],
            "event_col":            self._event_col,
            "duration_col":         self._duration_col,
            "tau":                  self._tau,
            "constraints_defined":  (
                len(self._schema_constraints.get(
                    "column_constraints", {})) +
                len(self._schema_constraints.get(
                    "cross_column_constraints", []))),
            "provenance":           self._schema_provenance,
        }
        # Invariant checks: warn if we exceed configured budget
        tiny_tol = 1e-9
        if eps_actual > float(self.epsilon) + tiny_tol:
            warnings.warn(
                f"epsilon_total_actual ({eps_actual}) > epsilon_total_configured ({self.epsilon}) + tol",
                UserWarning,
                stacklevel=2,
            )
        if delta_used > float(self.delta) + tiny_tol:
            warnings.warn(
                f"delta_used ({delta_used}) > delta ({self.delta}) + tol",
                UserWarning,
                stacklevel=2,
            )
        return report

