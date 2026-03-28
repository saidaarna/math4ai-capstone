import numpy as np

class SGD:
    """Standard Stochastic Gradient Descent optimizer."""

    def __init__(self, lr=0.01):
        self.lr = lr  # Learning rate: how big a step we take

    def update(self, params, grads):
        # Update rule: W = W - lr * gradient
        # We use a list comprehension to update all parameters (W and b) at once
        return [p - self.lr * g for p, g in zip(params, grads)]


class Momentum:
    """SGD with Momentum to speed up learning and reduce oscillation."""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.mu = momentum
        self.v = None  # Velocity vector

    def update(self, params, grads):
        # Initialize velocity with zeros if it's the first step
        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]

        # v = mu * v - lr * grad (Building up 'speed' in the right direction)
        self.v = [self.mu * vi - self.lr * g for vi, g in zip(self.v, grads)]

        # New parameters = old parameters + velocity
        return [p + vi for p, vi in zip(params, self.v)]


class Adam:
    """Adaptive Moment Estimation (Adam) optimizer.
    Combines ideas from Momentum and RMSProp."""

    def __init__(self, lr=0.001, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = None, None, 0

    def update(self, params, grads):
        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]  # 1st moment (mean)
            self.v = [np.zeros_like(p) for p in params]  # 2nd moment (variance)

        self.t += 1  # Increment timestep for bias correction

        # Update biased first moment estimate
        self.m = [self.b1 * mi + (1 - self.b1) * g for mi, g in zip(self.m, grads)]
        # Update biased second raw moment estimate
        self.v = [self.b2 * vi + (1 - self.b2) * (g ** 2) for vi, g in zip(self.v, grads)]

        # Compute bias-corrected estimates
        m_hat = [mi / (1 - self.b1 ** self.t) for mi in self.m]
        v_hat = [vi / (1 - self.b2 ** self.t) for vi in self.v]

        # Update parameters: p = p - lr * m_hat / (sqrt(v_hat) + epsilon)
        return [p - self.lr * mh / (np.sqrt(vh) + self.eps) for p, mh, vh in zip(params, m_hat, v_hat)]