class SkipGSGate:
    """View-adaptive backward gating for late-phase training.

    For each training view, maintains an EMA of observed loss. A view's
    backward is skipped when its current loss is at or below its EMA baseline.
    Budget control (rho_min) ensures backward fires often enough to prevent
    quality regression.

    Enabled once monitor.state()=="converged" has fired at least once since
    the last monitor.reset().  Warmup is measured in consecutive "improving"
    monitor samples after enable, not in iteration counts.

    Reference: Li, Lee, Fan. SkipGS (arXiv:2603.08997, 2026).
    """

    def __init__(self, warmup_steady_samples=50, beta=0.95, eps=1e-8, rho_lo=0.5):
        self.warmup_steady_samples = warmup_steady_samples
        self.beta = beta
        self.eps = eps
        self.rho_lo = rho_lo
        self._ema = {}
        self._step = 0
        self._backward_count = 0
        self._warmup_would_fire = 0
        self._warmup_evaluable = 0
        self._rho_min = None
        # Converged-once gate: set True when monitor fires "converged" after enable
        self._enabled = False
        # Consecutive "improving" samples seen after _enabled became True
        self._steady_count = 0

    def notify_monitor_state(self, state: str):
        """Call each iteration with the current monitor state string.

        Drives the two-stage enable: first "converged" unlocks _enabled;
        then warmup_steady_samples consecutive "improving" readings complete warmup.
        """
        if not self._enabled:
            if state == "converged":
                self._enabled = True
                self._steady_count = 0
        else:
            if self._rho_min is None:
                # Still in warmup: count consecutive improving samples
                if state == "improving":
                    self._steady_count += 1
                else:
                    self._steady_count = 0

    def is_ready(self) -> bool:
        """True once enabled and warmup samples have been observed."""
        return self._enabled and self._steady_count >= self.warmup_steady_samples

    def deviation(self, cam_id: int, loss_value: float) -> float:
        if cam_id not in self._ema:
            return float('inf')
        return loss_value / (self._ema[cam_id] + self.eps)

    def update_ema(self, cam_id: int, loss_value: float):
        prev = self._ema.get(cam_id)
        if prev is None:
            self._ema[cam_id] = loss_value
        else:
            self._ema[cam_id] = self.beta * prev + (1 - self.beta) * loss_value

    def decide(self, deviation_scores: list) -> tuple:
        """Returns (per_view_gate, will_backward).

        Before is_ready(), always returns all-True gates (normal training).
        """
        self._step += 1

        if not self.is_ready():
            for s, _ in zip(deviation_scores, range(len(deviation_scores))):
                if s != float('inf'):
                    self._warmup_evaluable += 1
                    if s > 1.0:
                        self._warmup_would_fire += 1
            return [True] * len(deviation_scores), True

        if self._rho_min is None:
            rho_hat_w = self._warmup_would_fire / max(self._warmup_evaluable, 1)
            self._rho_min = self.rho_lo + (1 - self.rho_lo) * rho_hat_w

        rho_cum = self._backward_count / max(self._step - 1, 1)
        if rho_cum < self._rho_min:
            return [True] * len(deviation_scores), True

        gate = [s > 1.0 for s in deviation_scores]
        return gate, any(gate)

    def record_backward(self, fired: bool):
        if fired:
            self._backward_count += 1
