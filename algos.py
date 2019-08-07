from skimage.feature import corner_harris, peak_local_max
import numpy as np


def dist2(x, c):
    # Borrowed from https://inst.eecs.berkeley.edu/~cs194-26/fa18/
    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, 'Data dimension does not match dimension of centers'

    return (np.ones((ncenters, 1)) * np.sum((x ** 2).T, axis=0)).T + \
           np.ones((ndata, 1)) * np.sum((c ** 2).T, axis=0) - \
           2 * np.inner(x, c)


def harris(im, maxPeaks=300, sigma=5, edge_discard=21):
    # Borrowed / modified from https://inst.eecs.berkeley.edu/~cs194-26/fa18/
    assert edge_discard >= 16

    # find harris corners
    h = corner_harris(im, method='eps', sigma=sigma)

    if maxPeaks == -1:
        coords = peak_local_max(h, min_distance=1, indices=True)
    else:
        coords = peak_local_max(h, min_distance=1, indices=True, num_peaks=maxPeaks)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask]

    return h, coords[:, ::-1]


def anms(h, c, cmax=20):
    c = c[:, ::-1]

    n = c.shape[0]
    x = np.empty((n, 3))
    fj = h[c[:, 0], c[:, 1]]

    for i in range(n):
        xi = c[i].reshape(2, 1)
        fi = h[xi[0], xi[1]]
        xj = c[fi < 0.9 * fj]
        r = np.linalg.norm(xi - xj.T, axis=0).min(initial=np.inf)
        x[i] = *xi, r

    x = x[x[:, 2].argsort()][::-1][:cmax]

    return x[:, :-1][:, ::-1].astype(int)


def makeHomography(Sxy, Txy):
    """
    Compute a homography between at least 4 coplanar source and target points.

    The homography is computed from Ah = b with the extra constraint that h33 = 1

    :param Sxy: (4+, 2) xy-coordinates for source points.
    :param Txy: (4+, 2) xy-coordinates for target points.
    :return: (3,3) a homography that maps Ps to Pt
    """

    assert len(Sxy) >= 4, 'at least 4 source points required.'
    assert len(Txy) >= 4, 'at least 4 target points required.'
    assert len(Sxy) == len(Txy), 'same number of source and target points (at least 4) required.'

    n = len(Sxy)
    b = np.zeros(2 * n + 1)
    # this coupled with constraint 3 makes h33 = 1
    b[-1] = 1

    i = np.arange(n)
    A = np.zeros((2 * n + 1, 9))

    # 1. x constraints:
    # -sx*h11 - sy*h12  - h13 + 0 + 0 + 0 + sx*tx*h31 + sy*tx*h32 + tx*h33
    A[2 * i] = np.array([
        -Sxy[i, 0], -Sxy[i, 1], [-1]*n,
        [0]*n, [0]*n, [0]*n,
        Sxy[i, 0] * Txy[i, 0], Sxy[i, 1] * Txy[i, 0], Txy[i, 0]
    ]).T

    # 2. y constraints:
    # 0 + 0 + 0 -sx*h21 - sy*h22 - h23 + sx*tx*h31 + sy*tx*h32 + tx*h33
    A[2 * i + 1] = np.array([
        [0]*n, [0]*n, [0]*n,
        -Sxy[i, 0], -Sxy[i, 1], [-1]*n,
        Sxy[i, 0] * Txy[i, 1], Sxy[i, 1] * Txy[i, 1], Txy[i, 1]
    ]).T

    # 3. constant scale constraint:
    # 0h11 + 0h12 + 0h13 + 0h21 + 0h22 + 0h23 + 0h31 + 0h32 + h33
    A[-1, -1] = 1

    H = np.linalg.lstsq(A, b, rcond=None)[0].reshape(3, 3)
    H_ = H / H[-1, -1]
    return H_


def binterp(X, x, y):
    """
    Bilinearly interpolate over the 4 neighbors of a subpixel coordinate.

    If the image has multiple channels (i.e.: RGB), the interpolation is done for each channel.

    :param X: (h, w, d). An image to to lookup pixel values from.
    :param x: (n, ) float. Subpixel x-coordinates.
    :param y: (n, ) float. Subpixel y-coordinates.
    :return: (n,) X.dtype. The interpolated pixel values.
    """
    x = np.array([x, y])
    t11 = np.round(x + .5)
    t11 = t11.astype(np.int)
    t01 = t11 - np.array([[1], [0]])
    t00 = t01 - np.array([[0], [1]])
    t10 = t11 - np.array([[0], [1]])

    dxy = (x - t00)[..., None]
    a = (1 - dxy[0, :]) * X[t00[1, :], t00[0, :]] + dxy[0, :] * X[t10[1, :], t10[0, :]]
    b = (1 - dxy[0, :]) * X[t01[1, :], t01[0, :]] + dxy[0, :] * X[t11[1, :], t11[0, :]]
    return (1 - dxy[1, :]) * a + dxy[1, :] * b


def makeDescriptors(A, Axy):
    D = np.empty((9, 9, Axy.shape[0]))
    # Sample from a 45x45 patch with stride 5 (you get a 9x9 descriptor)
    i, j = 5 * np.mgrid[-4:5, -4:5]

    for k, _ in enumerate(Axy):
        d = A[i + Axy[k, 1], j + Axy[k, 0]]
        d = (d - np.mean(d)) / np.std(d)
        D[..., k] = d

    return D


def matchDescriptors(Da, Db, nnMax=0.5):
    Da = Da.reshape(-1, Da.shape[-1]).T
    Db = Db.reshape(-1, Db.shape[-1]).T

    M = dist2(Da, Db)
    idx = np.argmin(M, axis=1)

    M[:].sort()
    ratios = M[:, 0] / M[:, 1]

    M = np.vstack((np.arange(Da.shape[0]), idx, ratios)).T
    nnInf = (np.inf, -1)
    nnMax = (nnMax, -1)
    best = {}

    for m in M:
        k = m[1]
        v = (m[2], m[0])
        if best.get(k, nnInf) > v and v < nnMax:
            best[k] = v

    M = np.empty((len(best), 2), dtype=np.intp)

    for i, (k, v) in enumerate(best.items()):
        M[i] = v[1], k

    return M


def ransac(Sxy, Txy, st, e, n):
    best = np.inf, None, []

    for _ in range(n):
        sample = np.random.choice(st.shape[0], (4,), replace=False)
        st_ = np.ones(st.shape[0], bool)
        st_[sample] = False
        st_ = st[st_]

        H = makeHomography(Sxy[st[sample, 0]], Txy[st[sample, 1]])
        Sxy_ = np.column_stack((Sxy[st_[:, 0]], np.ones(st_.shape[0])))
        Txy_ = H @ Sxy_.T
        Txy1 = (Txy_ / Txy_[-1])[:-1]
        Txy2 = Txy[st_[:, 1]].T

        l = np.linalg.norm(Txy1 - Txy2, axis=0)
        inliers = np.where(l < e)[0]

        if len(inliers) > 4:
            sample = np.hstack((sample, inliers))

            H = makeHomography(Sxy[st[sample, 0]], Txy[st[sample, 1]])
            Sxy_ = np.column_stack((Sxy[st[sample, 0]], np.ones(sample.shape[0])))
            Txy_ = H @ Sxy_.T
            Txy1 = (Txy_ / Txy_[-1])[:-1]
            Txy2 = Txy[st[sample, 1]].T

            l = np.mean(np.linalg.norm(Txy1 - Txy2, axis=0))

            if l <= e and l < best[0]: #and len(sample) >= len(best[2]):
                best = l, H, sample

    return best[1:]

