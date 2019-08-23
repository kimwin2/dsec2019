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

    for phase in ['train', 'valid']:

        if not os.path.isdir('dataset/%s/binary' % (phase)):
            os.mkdir('dataset/%s/binary' % (phase))

        for data in ['label']:

            if not os.path.isdir('dataset/%s/binary' % (phase)):
                os.mkdir('dataset/%s/binary' % (phase))

            root_dir = 'dataset/%s/%s' % (phase, data)

            for folder in os.listdir(root_dir):

                if not os.path.isdir('dataset/%s/binary/%s' % (phase, folder)):
                    os.mkdir('dataset/%s/binary/%s' % (phase, folder))

                label_folder = 'dataset/%s/label/%s' % (phase, folder)
                binary_folder = 'dataset/%s/binary/%s' % (phase, folder)

                total_fn = len(os.listdir(label_folder))

                for fn in range(total_fn):
                    filename = '%s/%04d.jpg' % (label_folder, fn+1)
                    img = cv2.imread('%s' % (filename), cv2.IMREAD_COLOR)

                    mask = cv2.inRange(img, np.array([0, 80, 0]), np.array([30, 255, 30]))
                    mask = poolingOverlap(mask, (8, 8))
                    cv2.imwrite('%s/%04d.jpg' % (binary_folder, fn+1), mask)
                    print(binary_folder)
                    print(fn)


