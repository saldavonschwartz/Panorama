import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def plotImages(S, T, Sxy, Txy):
    S_ = np.zeros_like(S)
    S_[Sxy[:, 1].astype(np.int), Sxy[:, 0].astype(np.int)] = (1, 0, 0)
    S_ = cv.dilate(S_, None)
    S_ = cv.dilate(S_, None)
    S_ = cv.dilate(S_, None)
    S_ = np.where(S_ == 0, S[..., 0][..., None], S_)

    T_ = np.zeros_like(T)
    T_[Txy[:, 1].astype(np.int), Txy[:, 0].astype(np.int)] = (1, 0, 0)
    T_ = cv.dilate(T_, None)
    T_ = cv.dilate(T_, None)
    T_ = cv.dilate(T_, None)
    T_ = np.where(T_ == 0, T[..., 0][..., None], T_)
    return S_, T_


def plotDescriptors(A, Axy, size):
    cmap = plt.cm.rainbow(np.linspace(0, 1, len(Axy)))
    A_ = np.zeros_like(A)
    A_[..., 0] = A_[..., 1] = A_[..., 2] = A[..., 0]

    for k in range(len(Axy)):
        i = (Axy[k, 0] - size, Axy[k, 1] - size)
        j = (Axy[k, 0] + size, Axy[k, 1] + size)
        cv.rectangle(A_, i, j, cmap[k], 2)
        cv.circle(A_, (Axy[k, 0], Axy[k, 1]), 3, cmap[k], -1)

    return A_


def plotMatches(S, T, Sxy, Txy):
    try:
        f, ax = plt.subplots(1, 1, figsize=(12, 8))
        ax.set_xticks([])
        ax.set_yticks([])

        out = np.hstack([S, T])

        offset = S.shape[1]
        cmap = plt.cm.rainbow(np.linspace(0, 1, len(Sxy)))
        y1, x1 = Sxy[:, 1].astype(np.int), Sxy[:, 0].astype(np.int)
        y2, x2 = Txy[:, 1].astype(np.int), Txy[:, 0].astype(np.int) + offset

        for i in range(len(cmap)):
            ax.plot([x1[i], x2[i]], [y1[i], y2[i]], marker=None, color=cmap[i], linewidth=.7, markersize=3,
                    antialiased=True)

        ax.imshow(out, cmap="Greys_r", aspect='equal')
        return f

    except:
        return None
