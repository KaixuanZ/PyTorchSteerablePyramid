import torch
from itertools import product


def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None) for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None) for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def batch_fftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(1, len(real.size())):  #(C,H,W) only shift H&W
        n_shift = real.size(dim)//2
        if real.size(dim) % 2 != 0:
            n_shift += 1  # for odd-sized images
        real = roll_n(real, axis=dim, n=n_shift)
        imag = roll_n(imag, axis=dim, n=n_shift)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

def batch_ifftshift2d(x):
    real, imag = torch.unbind(x, -1)
    for dim in range(len(real.size()) - 1, 0, -1):
        real = roll_n(real, axis=dim, n=real.size(dim)//2)
        imag = roll_n(imag, axis=dim, n=imag.size(dim)//2)
    return torch.stack((real, imag), -1)  # last dim=2 (real&imag)

class RegularGridInterpolator:

    def __init__(self, points, values):
        self.points = points
        self.values = values

        assert isinstance(self.points, tuple) or isinstance(self.points, list)
        assert isinstance(self.values, torch.Tensor)

        self.ms = list(self.values.shape)
        self.n = len(self.points)

        assert len(self.ms) == self.n

        for i, p in enumerate(self.points):
            assert isinstance(p, torch.Tensor)
            assert p.shape[0] == self.values.shape[i]

    def __call__(self, points_to_interp):
        assert self.points is not None
        assert self.values is not None

        assert len(points_to_interp) == len(self.points)
        K = points_to_interp[0].shape[0]
        for x in points_to_interp:
            assert x.shape[0] == K

        idxs = []
        dists = []
        overalls = []
        for p, x in zip(self.points, points_to_interp):
            idx_right = torch.bucketize(x, p)
            idx_right[idx_right >= p.shape[0]] = p.shape[0] - 1
            idx_left = (idx_right - 1).clamp(0, p.shape[0] - 1)
            dist_left = x - p[idx_left]
            dist_right = p[idx_right] - x
            dist_left[dist_left < 0] = 0.
            dist_right[dist_right < 0] = 0.
            both_zero = (dist_left == 0) & (dist_right == 0)
            dist_left[both_zero] = dist_right[both_zero] = 1.

            idxs.append((idx_left, idx_right))
            dists.append((dist_left, dist_right))
            overalls.append(dist_left + dist_right)

        numerator = 0.
        for indexer in product([0, 1], repeat=self.n):
            as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
            bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
            numerator += self.values[as_s] * \
                torch.prod(torch.stack(bs_s), dim=0)
        denominator = torch.prod(torch.stack(overalls), dim=0)
        return numerator / denominator