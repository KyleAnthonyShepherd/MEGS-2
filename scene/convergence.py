from collections import deque


class ConvergenceMonitor:
    def __init__(self, loss_window=200, densify_window=10):
        self.loss_history = deque(maxlen=loss_window)
        self.densify_history = deque(maxlen=densify_window)

    def update_loss(self, ema_loss: float):
        self.loss_history.append(ema_loss)

    def update_densify(self, n_added: int, n_total: int):
        self.densify_history.append(n_added / max(n_total, 1))

    def relative_slope(self) -> float:
        if len(self.loss_history) < self.loss_history.maxlen:
            return float('-inf')  # not enough data → assume improving
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
