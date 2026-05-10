class SkipGSGate:
    """View-adaptive backward gating for late-phase training.

    For each training view, maintains an EMA of observed loss. A view's
    backward is skipped when its current loss is at or below its EMA baseline.
    Budget control (rho_min) ensures backward fires often enough to prevent
    quality regression.

    Reference: Li, Lee, Fan. SkipGS (arXiv:2603.08997, 2026).
    """

    def __init__(self, warmup=500, beta=0.95, eps=1e-8, rho_lo=0.5):
        self.warmup = warmup
        self.beta = beta
        self.eps = eps
        self.rho_lo = rho_lo
        self._ema = {}                  # cam_id -> float EMA
        self._step = 0                  # outer steps since enable
        self._backward_count = 0        # outer steps where backward fired
        self._warmup_would_fire = 0     # warmup iters where s > 1 would have fired
        self._warmup_evaluable = 0      # warmup iters where view had EMA history
        self._rho_min = None            # set once at end of warmup

    def deviation(self, cam_id: int, loss_value: float) -> float:
        """Ratio of current loss to EMA baseline (s in the paper)."""
        if cam_id not in self._ema:
            return float('inf')         # no history → treat as high deviation
        return loss_value / (self._ema[cam_id] + self.eps)

    def update_ema(self, cam_id: int, loss_value: float):
        """Update per-view EMA with current observation."""
        prev = self._ema.get(cam_id)
        if prev is None:
            self._ema[cam_id] = loss_value
        else:
            self._ema[cam_id] = self.beta * prev + (1 - self.beta) * loss_value

    def decide(self, deviation_scores: list) -> tuple:
        """Returns (per_view_gate, will_backward).

        per_view_gate[k] = True means view k contributes to the gradient sum.
        will_backward = True means backward should fire (any view gates on, or
        budget floor forces it).
        """
        self._step += 1
        proposals = [s > 1.0 for s in deviation_scores]

        # Warmup: backward always fires; record prospective stats for calibration
        if self._step <= self.warmup:
            for s, p in zip(deviation_scores, proposals):
                if s != float('inf'):
                    self._warmup_evaluable += 1
                    if p:
                        self._warmup_would_fire += 1
            return [True] * len(proposals), True

        # First post-warmup step: calibrate rho_min once
        if self._rho_min is None:
            rho_hat_w = self._warmup_would_fire / max(self._warmup_evaluable, 1)
            self._rho_min = self.rho_lo + (1 - self.rho_lo) * rho_hat_w

        # Budget check (use ratio BEFORE this step's decision)
        rho_cum = self._backward_count / max(self._step - 1, 1)
        if rho_cum < self._rho_min:
            return [True] * len(proposals), True

        # Normal gating: include views with s > 1
        gate = [p for p in proposals]
        return gate, any(gate)

    def record_backward(self, fired: bool):
        """Call after each outer step to track the backward ratio."""
        if fired:
            self._backward_count += 1
