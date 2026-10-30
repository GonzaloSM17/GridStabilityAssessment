# Extreme cases

This independent workflow analyses the three deterministic controller settings
stored as a NET model plus `signals.npy` result matrix. It does not alter the
normal Monte Carlo or single-run workflows.

The NET file is used only to validate catalogued `*_MTR` and `PQ_NC_*` meters.
Numerical values are read only from `signals.npy`; `signal_index.json` is the
native mapping from each signal name to its raw NPY column. The published
batch flow evaluates `KIMAL_220` as its selected voltage bus, all available
active-power transfers, and high-frequency observations. Spatial analysis is
out of scope. The playground requests an Assets workbook to evaluate angular
stability using the inertia- and nominal-power-weighted centre of inertia.

Run the analysis with:

```powershell
python -m extreme_cases.playground
```

It validates that all cases have the same ordered NET meter catalog and NPY
matrix layout. The complete local `model/signal_index.json` supplies the
native one-based names for all 2,114 raw NPY columns.

Initialization is reported as diagnostic evidence only in this workflow. Every
extreme case is evaluated against the KIMAL voltage-recovery criteria even if
its initialization assessment reports a warning.
