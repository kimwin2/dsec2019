import os
import cv2 
import argparse 
import numpy as np

def asStride(arr, sub_shape, stride):
	'''Get a strided sub-matrices view of an ndarray.
	See also skimage.util.shape.view_as_windows()
	'''
	s0, s1=arr.strides[:2]
	m1, n1=arr.shape[:2]
	m2, n2=sub_shape
	view_shape=(1+(m1-m2)//stride[0], 1+(n1-n2)//stride[1], m2, n2)+arr.shape[2:]
	strides=(stride[0]*s0, stride[1]*s1, s0, s1)+arr.strides[2:]
	subs=np.lib.stride_tricks.as_strided(arr,view_shape,strides=strides)
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
	ky,kx=ksize
	if stride is None:
		stride=(ky,kx)
	sy,sx=stride
	
	_ceil=lambda x,y: int(np.ceil(x/float(y)))
	
	if pad:
		ny=_ceil(m,sy)
		nx=_ceil(n,sx)
		size=((ny-1)*sy+ky, (nx-1)*sx+kx) + mat.shape[2:]
		mat_pad=np.full(size,np.nan)
		mat_pad[:m,:n,...]=mat
	else:
		mat_pad=mat[:(m-ky)//sy*sy+ky, :(n-kx)//sx*sx+kx, ...]
	
	view=asStride(mat_pad,ksize,stride)
	
	result=np.nanmax(view,axis=(2,3))
	
	return result


if __name__ == '__main__': 
	
	resize_ratio = 10

	assert (os.path.isdir('raw_dataset')), 'Please download dataset' 

	if not os.path.isdir('dataset'): 
		os.mkdir('dataset') 

	for phase in ['train', 'valid']: 

		data_folder = 'raw_dataset/%s/data'%(phase) 
		label_folder = 'raw_dataset/%s/label'%(phase) 

		if not os.path.isdir('dataset/%s'%(phase)): 
			os.mkdir('dataset/%s'%(phase)) 

		for data in ['data', 'label']: 
			folder = 'raw_dataset/%s/%s'%(phase, data) 

			if not os.path.isdir('dataset/%s/%s'%(phase, data)): 
				os.mkdir('dataset/%s/%s'%(phase, data)) 

			for f in os.listdir(folder):
	
				vidcap = cv2.VideoCapture('%s/%s'%(folder, f)) 

				save_folder = 'dataset/%s/%s/%s'%(phase,data, f.split('.')[0]) 
				if not os.path.isdir(save_folder): 
					os.mkdir(save_folder) 

				idx = 0 

				while True: 
					success, image = vidcap.read() 
					if not success: 
						break 

					sz = (160, 90) 

					if data=='data': 
						dst = cv2.resize(image, dsize=sz, interpolation=cv2.INTER_CUBIC) 
						cv2.imwrite('%s/%04d.png'%(save_folder,idx), dst) 

					elif data=='label': 
						mask = cv2.inRange(image, np.array([0,80,0]), np.array([30,255,30]))
						mask = poolingOverlap(mask, (8, 8)) 
						cv2.imwrite('%s/%04d.png'%(save_folder,idx), mask) 

					idx+=1 
				
