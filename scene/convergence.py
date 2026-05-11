from collections import deque


class ConvergenceMonitor:
    def __init__(self, loss_window=200, densify_window=10):
        self.loss_history = deque(maxlen=loss_window)
        self.densify_history = deque(maxlen=densify_window)
        # Bumped on every reset(); used by callers that want to fire "once
        # per convergence cycle" (e.g. SG axis cull).
        self.cycle = 0

    def update_loss(self, ema_loss: float):
        self.loss_history.append(ema_loss)

    def update_densify(self, n_added: int, n_total: int):
        self.densify_history.append(n_added / max(n_total, 1))

    def reset(self, fraction_changed: float = 1.0):
        """Clear loss history after a structural scene change.

        fraction_changed: fraction of Gaussians added or removed (0–1).
        The minimum entries required before convergence can fire again is
        scaled by this fraction — a 70% prune needs ~140 fresh entries; a
        5% trim only needs ~10, so the system can re-detect quickly.
        """
        self.loss_history.clear()
        self._min_entries = max(10, int(self.loss_history.maxlen * fraction_changed))
        self.cycle += 1

    def relative_slope(self) -> float:
        min_entries = getattr(self, '_min_entries', self.loss_history.maxlen)
        if len(self.loss_history) < min_entries:
            return float('-inf')  # not enough data → assume improving
        # Reset the threshold once we have enough data
        self._min_entries = self.loss_history.maxlen
        recent = list(self.loss_history)
        slope = (recent[-1] - recent[0]) / len(recent)
        return slope / max(recent[-1], 1e-8)

    def densify_saturation(self) -> float:
        if not self.densify_history:
            return 0.0
        return max(self.densify_history)

    def state(self,
              converged_slope=-1e-4,
              active_densify=0.05,
              active_slope=-1e-3) -> str:
        s = self.relative_slope()
        d = self.densify_saturation()
        if s > converged_slope and d < 0.01:
            return "converged"
        if d > active_densify and s < active_slope:
            return "wants_capacity"
        if s < active_slope:
            return "improving"
        return "stalled"
