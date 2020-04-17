Automatic Panorama Stitcher
===========================

This project implements a pipeline for the automatic stitching of images into panoramic compositions via reprojection of the images onto a common plane. In order to compute a homography to reproject the images, common features between pairs of images are detected and matched automatically. The project is an application of image registration and is based on the work by `Brown, Szeliski and Winder <https://inst.eecs.berkeley.edu/~cs194-26/fa18/Papers/MOPS.pdf>`_.

For a detailed writeup on the project see `this post <https://0xfede.io/2019/03/09/panorama.html>`_.

Dependencies:
=============
* `NumPy <https://numpy.org/>`_
* `scikit-image <https://scikit-image.org/>`_ (for Harris corner detector)
* `OpenCV <https://opencv.org/>`_ (only if visualizing intermediate results)
* `matplotlib <https://matplotlib.org/>`_ (only if visualizing intermediate results)


Installation:
=============
After downloading or cloning the repo:

:code:`pip install -r requirements.txt`



`LinkedIn <https://www.linkedin.com/in/federicosaldarini>`_ |
`0xfede.io <https://0xfede.io>`_ | `GitHub <https://github.com/saldavonschwartz>`_
