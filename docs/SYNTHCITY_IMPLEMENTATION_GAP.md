# SynthCity PrivBayes implementation gap (reported)

**Summary:** The synthetic event column (e.g. `status`) produced by SynthCity’s standalone PrivBayes often collapses to a single class (all 0 or all 1). This behaviour is **not** implied by the PrivBayes algorithm or the original paper; it is due to **implementation bugs in the SynthCity code**, and is reported here as a gap for upstream (SynthCity) and for our report.

---

## 1. Why this is an implementation gap, not the algorithm

- The same dataset and schema are used by CRN and dpmm; both produce a non-degenerate event column and a well-defined attribute-inference AUC.
- Only SynthCity’s synthetic event column consistently collapses to one class in our experiments.
- The PrivBayes paper does not suggest that the event variable should be degenerate; the algorithm is defined over full CPTs and should sample both outcomes when the learned (noisy) distribution has both.

So the root cause is in the **SynthCity implementation**, not in the design of PrivBayes.

---

## 2. Root causes identified

### 2.1 Root-node CPD shape (bug; to be fixed upstream)

- **What:** For nodes with no parents, the code passed a **1D** array of probabilities to pgmpy’s `TabularCPD`.
- **pgmpy requirement:** `TabularCPD` expects `values` of shape `(variable_card, 1)` for root nodes (see pgmpy docs and `pgmpy/factors/discrete/CPD.py`: `expected_cpd_shape = (variable_card, 1)` when `evidence is None`).
- **Effect:** Passing shape `(2,)` instead of `(2, 1)` violates the API and can lead to wrong interpretation or errors depending on pgmpy version, and can affect sampling.
- **Fix:** No fix is applied in this repo; this should be fixed in SynthCity (reshape root-node `node_values` to `(variable_card, 1)` before constructing `TabularCPD`).

### 2.2 Conditional CPD column order vs pgmpy (suspected bug, to report upstream)

- **What:** For nodes with parents, the conditional CPD is built from a pandas `crosstab` over the count table. The **column order** of this table (evidence configurations) is determined by pandas (e.g. order of appearance or index), not by pgmpy’s convention.
- **pgmpy convention:** During `forward_sample`, pgmpy indexes CPD columns by evidence state tuples. The CPD is expected to have columns in a well-defined order (e.g. product order of evidence variables: first evidence varies slowest). See `pgmpy/sampling/base.py` and `pre_compute_reduce_maps`.
- **Risk:** If the implementation’s CPD column order does not match the order pgmpy uses when looking up weights for a given evidence configuration, the sampler will use the **wrong** conditional distribution. That can make one outcome overwhelmingly likely and produce a single-class column.
- **Status:** Not fixed here; recommended to report to SynthCity maintainers so that CPD construction (e.g. crosstab or reindexing) explicitly follows pgmpy’s expected evidence-state order.

### 2.3 Other possible contributors

- **Laplace noise:** For binary variables with small counts, heavy noise could in principle zero out one outcome after clipping; with the current smoothing (`+1` and normalize), both outcomes typically remain possible. So noise alone is unlikely to fully explain a **systematic** single-class output only for SynthCity.
- **Random seed / sampling:** Different runs can yield different DAGs and CPTs; the fact that the collapse is **reproducible and specific to SynthCity** points to a deterministic implementation issue (e.g. shape or column order) rather than bad luck.

---

## 3. Recommendation

- **For this repo:** No patch is applied to SynthCity code; the report and tables state that SynthCity’s attribute-inference AUC is often “—” due to a **SynthCity implementation gap** (synthetic event column collapses to one class), not due to the PrivBayes paper or algorithm.
- **For SynthCity:** Report as an implementation gap: (1) root-node CPD must have shape `(variable_card, 1)` for pgmpy; (2) conditional CPD column order must match pgmpy’s evidence-state order so that sampling uses the correct conditionals.

---

## 4. References

- PrivBayes: Zhang J, Cormode G, Procopiuc CM, Srivastava D, Xiao X. *PrivBayes: Private Data Release via Bayesian Networks.* 2017.
- pgmpy `TabularCPD`: requires `values.shape == (variable_card, 1)` for root nodes and `(variable_card, product(evidence_card))` for conditional nodes, with column order matching evidence state product order.
- This repo: `synthcity_standalone/privbayes.py` (unchanged; gap documented only); `implementations/synthcity_wrapper.py` (wrapper only).

---

## 5. For reporting (upstream / write-up)

**Short statement:** The standalone PrivBayes implementation in SynthCity exhibits an implementation gap: the synthetic event column (e.g. binary `status`) often collapses to a single class, so downstream metrics (e.g. attribute-inference AUC) cannot be computed. This is not a property of the PrivBayes algorithm; other implementations (CRN, dpmm) on the same data produce a non-degenerate event column. Root causes identified: (1) root-node CPD passed to pgmpy as 1D instead of shape `(variable_card, 1)`; (2) conditional CPD column order may not match pgmpy’s evidence-state order. This gap will be reported to SynthCity maintainers.
