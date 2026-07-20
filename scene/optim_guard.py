"""Optimizer-binding invariant (PERFORMANCE_PLAN T1 / Plan 2 M2.3).

Adam keys its state by parameter object identity. Code that rebinds a
param group's tensor without migrating state (or that recreates the
optimizer after a prune) silently discards momentum. This check catches
both: every state entry must belong to a currently-bound parameter, and —
once any state exists — every bound parameter of a group that has stepped
must still own its entry.
"""


def optimizer_binding_ok(optimizer) -> bool:
    """True iff optimizer state (if any) is keyed by currently-bound params.

    A fresh optimizer (no steps yet, empty state) is OK. After steps, any
    state entry keyed by an object that is no longer in a param group means
    a rebind lost the momentum migration.
    """
    if optimizer is None:
        return True
    state = getattr(optimizer, "state", None)
    if not state:
        return True
    bound = {id(p) for g in optimizer.param_groups for p in g["params"]}
    return all(id(k) in bound for k in state.keys())
