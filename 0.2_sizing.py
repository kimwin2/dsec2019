import os
import cv2
import argparse
import numpy as np


def asStride(arr, sub_shape, stride):
    '''Get a strided sub-matrices view of an ndarray.
    See also skimage.util.shape.view_as_windows()
    '''
    s0, s1 = arr.strides[:2]
    m1, n1 = arr.shape[:2]
    m2, n2 = sub_shape
    view_shape = (1 + (m1 - m2) // stride[0], 1 + (n1 - n2) // stride[1], m2, n2) + arr.shape[2:]
    strides = (stride[0] * s0, stride[1] * s1, s0, s1) + arr.strides[2:]
    subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides=strides)
    return subs


def poolingOverlap(mat, ksize, stride=None, pad=False):
    '''Overlapping pooling on 2D or 3D data.

    <mat>: ndarray, input array to pool.
    <ksize>: tuple of 2, kernel size in (ky, kx).
    <stride>: tuple of 2 or None, stride of pooling window.
              If None, same as <ksize> (non-overlapping pooling).
    <pad>: bool, pad <mat> or not. If no pad, output has size
           (n-f)//s+1, n being <mat> size, f being kernel size, s stride.
           if pad, output has size ceil(n/s).

    Return <result>: pooled matrix.
    '''
    m, n = mat.shape[:2]
    ky, kx = ksize
    if stride is None:
        stride = (ky, kx)
    sy, sx = stride

    _ceil = lambda x, y: int(np.ceil(x / float(y)))

    if pad:
        ny = _ceil(m, sy)
        nx = _ceil(n, sx)
        size = ((ny - 1) * sy + ky, (nx - 1) * sx + kx) + mat.shape[2:]
        mat_pad = np.full(size, np.nan)
        mat_pad[:m, :n, ...] = mat
    else:
        mat_pad = mat[:(m - ky) // sy * sy + ky, :(n - kx) // sx * sx + kx, ...]

    view = asStride(mat_pad, ksize, stride)

    result = np.nanmax(view, axis=(2, 3))

    return result


if __name__ == '__main__':

    assert (os.path.isdir('dataset')), 'Please download dataset'

    hmin = 1000000000
    hmax = 0
    wmin = 1000000000
    wmax = 0

    for phase in ['train', 'valid']:
        for data in ['data', 'label', 'binary']:
            root_dir = '/home/ymkim/dsec/190808/dataset/%s/%s' % (phase, data)
            for folder in os.listdir(root_dir):
                label_folder = '/home/ymkim/dsec/190808/dataset/%s/label/%s' % (phase, folder)
                binary_folder = '/home/ymkim/dsec/190808/dataset/%s/binary/%s' % (phase, folder)

                total_fn = len(os.listdir(label_folder))
                print("total_fn", total_fn)
                for fn in range(total_fn):
                    filename = '%s/%04d.jpg' % (label_folder, fn+1)
                    print("addr",filename)
                    img = cv2.imread('%s' % (filename), cv2.IMREAD_COLOR)
                    h, w, ch = img.shape
                    if h < hmin:
                        hmin = h
                    if h > hmax:
                        hmax = h
                    if w < wmin:
                        wmin = w
                    if w > wmax:
                        wmax = w

    print("hmin hmax", hmin, hmax)
    print("wmin wmax", wmin, wmax)
    dim = (wmax, hmax)
    for phase in ['train', 'valid']:
        for data in ['data', 'label', 'binary']:
            root_dir = 'dataset/%s/%s' % (phase, data)
            for folder in os.listdir(root_dir):
                folder = 'dataset/%s/%s/%s' % (phase, data, folder)

                total_fn = len(os.listdir(folder))

                for fn in range(total_fn):
                    filename = '%s/%04d.jpg' % (folder, fn+1)
                    img = cv2.imread('%s' % (filename), cv2.IMREAD_COLOR)
                    print(filename)
                    dim = (256, 35)
                    resized_img = cv2.resize(img, dim)
                    cv2.imwrite('%s/resized_%04d.jpg' % (folder, fn+1), resized_img)

                    # 폴더 추가 필요