import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import json
import os


class Profiler:
    def __init__(self, **kwargs):
        self.logger = kwargs.get('logger', print)
        self.label = kwargs.get('label', '')
        self.info = [None] * 3

    def __enter__(self):
        self.info[0] = datetime.now()
        return self.info

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.info[1] = datetime.now()
        self.info[2] = self.info[1] - self.info[0]

        if self.logger:
            label = ': {}'.format(self.label) if self.label else ''
            self.logger('[profiler{}] elapsed: {}'.format(label, self.info[2]))


class Image:
    @staticmethod
    def load(*paths, **kwargs):
        imgs = []

        for path in paths:
            if not os.path.exists(path):
                print('file not found: {}'.format(path))
                continue

            X = plt.imread(path)

            if len(X.shape) is 3 and X.shape[2] is 4:
                X = X[..., :-1]

            if X.dtype.kind in 'ui' and kwargs.get('float', False):
                X = X / 255

            imgs.append(X)

        return imgs[0] if len(imgs) is 1 else imgs

    @staticmethod
    def show(*imgs):
        f, ax = plt.subplots(1, len(imgs), figsize=(12, 8))

        if len(imgs) is 1:
            ax.imshow(imgs[0], aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])

        else:
            for i in range(len(imgs)):
                ax[i].imshow(imgs[i], aspect='equal')
                ax[i].set_xticks([])
                ax[i].set_yticks([])

        plt.waitforbuttonpress()
        plt.close(f)

    @staticmethod
    def save(*imgsAndPaths):
        for X, path in imgsAndPaths:
            plt.imsave(path, X)


class Features:
    @staticmethod
    def capture(S, T, mode, save, overwrite):
        if save and not overwrite and os.path.exists(save):
            Sxy, Txy = Features.load(save)
        else:
            assert mode is 1 or 2

            if mode is 1:
                Sxy, Txy = Features.__capture1__(S, T)
            else:
                Sxy, Txy = Features.__capture2__(S, T)

            if save:
                with open(save, 'wt') as file:
                    json.dump({'Sxy': Sxy.tolist(), 'Txy': Txy.tolist()}, file, indent=2)

        return Sxy, Txy

    @staticmethod
    def __capture1__(S, T):
        done = False

        def onKey(e):
            nonlocal done

            if e.key == 'escape':
                done = True

        layout, count = ((7, 7), 110)

        f = plt.figure(figsize=layout)
        e1 = f.canvas.mpl_connect('key_press_event', onKey)

        ax = f.add_subplot(count + 1)
        ax.set_xticks([])
        ax.set_yticks([])
        image = ax.imshow((T + 0.3) * 0.4, interpolation=None, aspect='equal')

        c = plt.get_cmap('hsv')
        Sxy, Txy = [], []

        while not done:
            xy = plt.ginput(n=4, timeout=0)

            if len(xy) == 4:
                Txy.append(xy)
                d = np.array(Txy[-1])
                ax.scatter(d[:, 0], d[:, 1], c=[c(np.random.randint(0, 256))] * 4, marker='.')
                plt.draw()

        Sxy = np.array([[0, 0], [S.shape[1] - 1, 0], [S.shape[1] - 1, S.shape[0] - 1], [0, S.shape[0] - 1]])
        Txy = np.array(Txy)

        f.canvas.mpl_disconnect(e1)
        return Sxy, Txy

    @staticmethod
    def __capture2__(S, T):
        done, ax, xy, axplot, axc = False, None, [], None, None

        def onKey(e):
            nonlocal done

            if e.key == 'escape':
                done = len(Sxy) == len(Txy) and len(Sxy) >= 4
            elif e.key == 'backspace' and xy:
                i = xy.pop()
                i[1].remove()
                updatePlot()

        def onAxisEnter(e):
            nonlocal ax, xy, axplot, axc

            if e.inaxes is ax1:
                ax, xy, axplot, axc = (ax1, Sxy, splot, 'r')
            else:
                ax, xy, axplot, axc = (ax2, Txy, tplot, 'b')

        def updatePlot():
            axplot.set_data([c[0][0] for c in xy], [c[0][1] for c in xy])

            for i, c in enumerate(xy):
                c[1].set_text(str(i))
                c[1].set_position((c[0][0] + 5, c[0][1] - 5))

            f.canvas.draw()

        f = plt.figure(figsize=(10, 5))
        f.subplots_adjust(hspace=0.1, wspace=0.1)
        e1 = f.canvas.mpl_connect('key_press_event', onKey)
        e2 = f.canvas.mpl_connect('axes_enter_event', onAxisEnter)

        ax1 = f.add_subplot(1, 2, 1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.imshow((S + 0.3) * 0.4, interpolation=None, aspect='equal')
        splot, = ax1.plot([], [], color='r', marker='.', ms=5, linestyle='')

        ax2 = f.add_subplot(1, 2, 2)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow((T + 0.3) * 0.4, interpolation=None, aspect='equal')
        tplot, = ax2.plot([], [], color='b', marker='.', ms=5, linestyle='')

        Sxy, Txy = [], []

        while not done:
            p = plt.ginput(n=1, timeout=0, show_clicks=False)

            if not p:
                continue

            xy.append((p[0], ax.text(0, 0, '', color=axc, fontsize=7)))
            updatePlot()

        Sxy = np.array([i[0] for i in Sxy])
        Txy = np.array([i[0] for i in Txy])

        f.canvas.mpl_disconnect(e1)
        f.canvas.mpl_disconnect(e2)
        return Sxy, Txy

    @staticmethod
    def __capture2__(S, T):
        done, ax, xy, axplot, axc = False, None, [], None, None

        def onKey(e):
            nonlocal done

            if e.key == 'escape':
                done = len(Sxy) == len(Txy) and len(Sxy) >= 4
            elif e.key == 'backspace' and xy:
                i = xy.pop()
                i[1].remove()
                updatePlot()

        def onAxisEnter(e):
            nonlocal ax, xy, axplot, axc

            if e.inaxes is ax1:
                ax, xy, axplot, axc = (ax1, Sxy, splot, 'r')
            else:
                ax, xy, axplot, axc = (ax2, Txy, tplot, 'b')

        def updatePlot():
            axplot.set_data([c[0][0] for c in xy], [c[0][1] for c in xy])

            for i, c in enumerate(xy):
                c[1].set_text(str(i))
                c[1].set_position((c[0][0] + 5, c[0][1] - 5))

            f.canvas.draw()

        f = plt.figure(figsize=(10, 5))
        f.subplots_adjust(hspace=0.1, wspace=0.1)
        e1 = f.canvas.mpl_connect('key_press_event', onKey)
        e2 = f.canvas.mpl_connect('axes_enter_event', onAxisEnter)

        ax1 = f.add_subplot(1, 2, 1)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.imshow((S + 0.3) * 0.4, interpolation=None, aspect='equal')
        splot, = ax1.plot([], [], color='r', marker='.', ms=5, linestyle='')

        ax2 = f.add_subplot(1, 2, 2)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow((T + 0.3) * 0.4, interpolation=None, aspect='equal')
        tplot, = ax2.plot([], [], color='b', marker='.', ms=5, linestyle='')

        Sxy, Txy = [], []

        while not done:
            p = plt.ginput(n=1, timeout=0, show_clicks=False)

            if not p:
                continue

            xy.append((p[0], ax.text(0, 0, '', color=axc, fontsize=7)))
            updatePlot()

        Sxy = np.array([i[0] for i in Sxy])
        Txy = np.array([i[0] for i in Txy])

        f.canvas.mpl_disconnect(e1)
        f.canvas.mpl_disconnect(e2)
        return Sxy, Txy

    @staticmethod
    def load(path):
        with open(path, 'rt') as file:
            data = json.load(file)
            return np.array(data['Sxy']), np.array(data['Txy'])


