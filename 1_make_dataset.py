import os
import cv2
import torch
import numpy as np


if __name__ == '__main__': 


	if not os.path.isdir('pt_dataset'):
		os.mkdir('pt_dataset') 	

	for phase in ['train', 'valid']: 

		for data in ['data', 'binary']:

			root_dir = 'dataset/%s/%s'%(phase, data) 
			
			images = [] 

			for folder in os.listdir(root_dir): 

				sample = []
				sub_folder = 'dataset/%s/%s/%s'%(phase, data, folder) 
	
				# data_folder = 'dataset/%s/label/%s'%(phase, folder) 
				# label_folder = 'dataset/%s/data/%s'%(phase, folder) 

				data_folder = 'dataset/%s/data/%s'%(phase, folder) 
				label_folder = 'dataset/%s/binary/%s'%(phase, folder) 


#				print(data_folder, len(os.listdir(data_folder)))
#				print(label_folder, len(os.listdir(label_folder))) 

				total_fn = min(len(os.listdir(data_folder)), len(os.listdir(label_folder)))

				for fn in range(int(total_fn/2)):

					filename = '%s/resized_%04d.jpg'%(sub_folder, fn+1)
					print(filename)
					if data=='binary': 
						image = cv2.imread('%s'%(filename), cv2.IMREAD_GRAYSCALE) 
						ret,thresh1 = cv2.threshold(image,127,255,cv2.THRESH_BINARY)
						image = torch.tensor(thresh1, dtype=torch.float)/255.
					else: 
						image = cv2.imread('%s'%(filename)) 
						image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
						image = torch.tensor(image, dtype=torch.float).permute([2,0,1])/255.
						
					# padding 
					h,w = image.size()[-2:] 

					h = h+1 if h%4==0 else (h+4)//4*4+1 
					w = w+1 if w%4==0 else (w+4)//4*4+1 

					if data=='binary': 
						padded_image = torch.zeros(h, w, dtype=torch.float) 
						padded_image[:image.size(0), :image.size(1)] = image 
					else: 
						padded_image = torch.zeros(3, h, w, dtype=torch.float) 
						padded_image[:, :image.size(1), :image.size(2)] = image 
					#print(padded_image.size())
					sample.append(padded_image) 
					if len(sample)==10: 
						images.append(torch.stack(sample, 0)) 
						sample = [] 

			images = torch.stack(images, 0) 
						
			torch.save(images, 'pt_dataset/%s_%s.pt'%(phase, data)) 
