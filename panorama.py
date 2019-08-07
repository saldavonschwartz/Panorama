import numpy as np
import algos
import utils
import viz
import threading


def cornerDetectionAndSuppression(I, Imask, anms, cmax, out):
    if not anms:
        _, Ixy = algos.harris(I, maxPeaks=cmax)
        out.append(Ixy)
        Id = algos.makeDescriptors(I, Ixy)
        out.append(Id)
    else:
        Ih, Ixy = algos.harris(Imask if Imask is not None else I, maxPeaks=-1)
        Ixy_ = algos.anms(Ih, Ixy, cmax=cmax)
        out.append(Ixy)
        out.append(Ixy_)
        Id = algos.makeDescriptors(I, Ixy_)
        out.append(Id)


def stitch(S, T, Tpre, anms, cmax, maskpow=1., intermediates=None):
    # 1. Operate on grayscale images (red channel chosen arbitrarily):
    S_, T_ = S[..., 0], T[..., 0]

    # 2. Corner Detection + Non-Maximal Suppression:
    Tmask = np.where(Tpre != 0, T, 0)[..., 0] if anms else None

    out = [[], []]
    tasks = [
        threading.Thread(target=cornerDetectionAndSuppression, args=(S_, None, anms, cmax, out[0])),
        threading.Thread(target=cornerDetectionAndSuppression, args=(T_, Tmask, anms, cmax, out[1]))
    ]

    [t.start() for t in tasks]
    [t.join() for t in tasks]

    # All detected corners + descriptors
    Sxy_, Txy_ = out[0][0], out[1][0]
    Sd, Td = out[0][-1], out[1][-1]

    if not anms:
        # Keep lower of N most prominent between S and T
        hmin = min(Sxy_.shape[0], Txy_.shape[0])
        Sxy, Txy = Sxy_[:hmin], Txy_[:hmin]
        Sd, Td = Sd[..., :hmin], Td[..., :hmin]

    else:
        # ANMS already dropped some
        Sxy, Txy = out[0][1], out[1][1]

    print('[total corners]:\t\t\t\tS: {} | T: {}'.format(len(Sxy_), len(Txy_)))
    print('[after suppression ({})]:\t\t\t\tS: {} | T: {}'.format('ANMS' if anms else 'rank+min', len(Sxy), len(Txy)))

    if intermediates is not None:
        # plot all corners found
        S1_, T1_ = viz.plotImages(S, T, Sxy_, Txy_)
        intermediates.append(S1_)
        intermediates.append(T1_)

        # plot corners left after suppression
        S1_, T1_ = viz.plotImages(S, T, Sxy, Txy)
        intermediates.append(S1_)
        intermediates.append(T1_)

    # 3. Match 9x9 descriptors out of detected corners:
    idx = algos.matchDescriptors(Sd, Td, nnMax=0.55)
    print('[matched descriptors]:\t\t{}'.format(len(idx)))

    if intermediates is not None:
        # plot matched descriptors:
        S1_ = viz.plotDescriptors(S, Sxy[idx[:, 0], :], size=9)
        T1_ = viz.plotDescriptors(T, Txy[idx[:, 1], :], size=9)
        intermediates.append(S1_)
        intermediates.append(T1_)

    # 4. Create homography from source to target, based on the best
    # set of descriptors computed via RANSAC:
    H, c = algos.ransac(Sxy, Txy, idx, e=6, n=1000)
    print('[RANSAC set length]:\t\t{}'.format(len(c)))

    if H is None:
        print('skip')
        return T, T

    if intermediates is not None:
        # plot best matched descriptors after RANSAC:
        S1_ = viz.plotDescriptors(S, Sxy[idx[c, 0], :], size=9)
        T1_ = viz.plotDescriptors(T, Txy[idx[c, 1], :], size=9)
        f = viz.plotMatches(S1_, T1_, Sxy[idx[c, 0], :], Txy[idx[c, 1], :])

        if f:
            intermediates.append(f)
        else:
            intermediates.append(S1_)
            intermediates.append(T1_)

    th, tw = T.shape[0], T.shape[1]
    sh, sw = S.shape[0], S.shape[1]

    # 5. Forward warp source corners onto target space to compute final composite size:
    Sc_ = np.column_stack([(0, 0, 1), (sw - 1, 0, 1), (sw - 1, sh - 1, 1), (0, sh - 1, 1)])
    Tc_ = H @ Sc_
    Tc = (Tc_ / Tc_[-1])[:-1]

    if (Tc_[:2, 0] < Sc_[:2, 0]).any():
        maskRange = (0., 1.)
    else:
        maskRange = (1., 0.)

    cmin = np.minimum(np.amin(Tc, axis=1), (0, 0))
    cmax = np.maximum(np.amax(Tc, axis=1), (tw - 1, th - 1))
    csize = np.ceil((cmax - cmin) + 1).astype(np.int)[::-1]

    if len(T.shape) is 3:
        csize = (*csize, T.shape[2])

    # 6. Copy target to new size:
    T_ = np.zeros(csize)
    cmin = np.abs(cmin).astype(np.int)
    T_[cmin[1]: cmin[1] + th, cmin[0]: cmin[0] + tw] = T

    # 7. Inverse warp target onto source space (accounting for offset in new target size):
    i = np.meshgrid(np.arange(csize[1]), np.arange(csize[0]))
    Txy_ = np.vstack((i[0].flatten(), i[1].flatten(), np.ones(csize[0] * csize[1]))).astype(np.int)
    cmin_ = np.row_stack((*cmin, 0))

    H_ = np.linalg.inv(H)
    Sxy_ = H_ @ (Txy_ - cmin_)
    Sxy = (Sxy_ / Sxy_[-1])[:-1]
    Txy = Txy_[:-1]

    # 8. Copy source to new size (from points in source space range to target space).
    S_ = np.zeros(csize)
    i = ((Sxy.T >= (0, 0)) & (Sxy.T <= (sw - 1, sh - 1))).all(axis=1).nonzero()[0]
    Txy = Txy[:, i]
    Sxy = Sxy[:, i]
    S_[Txy[1], Txy[0]] = algos.binterp(S, Sxy[0], Sxy[1])

    # 9. Final composite (a quick alpha blending):
    m = np.where((S_ != 0) & (T_ != 0))
    mvals = np.interp(m[1], (m[1].min(), m[1].max()), maskRange) ** maskpow
    C = np.where(S_ != 0, S_, T_)
    C[m] = (1.-mvals)*S_[m] + mvals*T_[m]

    if intermediates is not None:
        S1_ = S_.copy()
        T1_ = T_.copy()
        S1_[m] = (1. - mvals) * S1_[m]
        T1_[m] = mvals * T1_[m]

        intermediates.append(S_)
        intermediates.append(T_)
        intermediates.append(S1_)
        intermediates.append(T1_)

    return C, T_


def testPanorama(example, outprefix, anms, cmax, intermediates=False):
    if example == 1:
        # example 1: living room
        outpath = './data/panorama/livingroom/processed/'
        paths = [
            './data/panorama/livingroom/lr-l.jpg',
            './data/panorama/livingroom/lr-c.jpg',
            './data/panorama/livingroom/lr-r.jpg'
        ]
    else:
        # example 2: balcony
        outpath = './data/panorama/balcony/processed/'
        paths = [
            './data/panorama/balcony/IMG_4189.jpg',
            './data/panorama/balcony/IMG_4190.jpg',
            './data/panorama/balcony/IMG_4191.jpg',
            './data/panorama/balcony/IMG_4188.jpg',
            './data/panorama/balcony/IMG_4192.jpg',
            './data/panorama/balcony/IMG_4187.jpg',
            './data/panorama/balcony/IMG_4193.jpg',
            './data/panorama/balcony/IMG_4186.jpg',
            './data/panorama/balcony/IMG_4194.jpg',
            './data/panorama/balcony/IMG_4185.jpg',
            './data/panorama/balcony/IMG_4195.jpg'
        ]

    imgs = []
    np.random.seed(12);
    S, T = paths[:2]

    with utils.Profiler():
        print(paths[0], paths[1])

        try:
            S, T = utils.Image.load(S, T, float=True)
            with utils.Profiler():
                T, T_ = stitch(S, T, T, anms, cmax, maskpow=.2, intermediates=imgs if intermediates else None)
            imgs.append(T)

        except Exception as e:
            print(e)
            print('error processing: ', paths[0], paths[1], ' skip')

        for path in paths[2:]:
            print(path)

            try:
                S = utils.Image.load(path, float=True)
                with utils.Profiler():
                    T, T_ = stitch(S, T, T_, anms, cmax, maskpow=6., intermediates=imgs if intermediates else None)
                imgs.append(T)

            except Exception as e:
                print(e)
                print('error processing: ', path, ' skip.')

        print('done')

    print('saving images...')

    if not intermediates:
        imgs = imgs[-1:]

    for i, img in enumerate(imgs):
        if type(img) is np.ndarray:
            utils.Image.save((
                img, outpath + outprefix + str(i) + '.jpg'
            ))
        else:
            img.savefig(
                outpath + outprefix + str(i) + '.svg',
                dpi=1200, transparent=True, bbox_inches = 'tight', pad_inches=0
            )

        print(i+1, ' saved...')


# testPanorama(1, 'livingroom-', anms=False, cmax=300, intermediates=False)
# testPanorama(1, 'anms/livingroom-anms-', anms=True, cmax=300, intermediates=False)

# testPanorama(2, 'balcony-', anms=False, cmax=300)
# testPanorama(2, 'balcony-anms-', anms=True, cmax=300)
