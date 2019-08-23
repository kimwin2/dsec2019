import os
import cv2
import time
import torch
import argparse 
import numpy as np 
from torch import optim 
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
from model.ConvRRN import ST_EncDec as Model 
import seaborn as sns 
import pandas as pd 

class Dataset(torch.utils.data.Dataset): 
	def __init__(self, train=True): 
		super(Dataset, self).__init__() 

		if train==True: 
			input_name = 'dataset/for_training/train_inputs.pt' 
			target_name = 'dataset/for_training/train_targets.pt' 
		else: 
			input_name = 'dataset/for_training/valid_inputs.pt' 
			target_name = 'dataset/for_training/valid_targets.pt' 
		
		self.inputs = (torch.load(input_name)-0.5)/0.5 
		self.targets = torch.load(target_name)

		counts = [] 
		min_count = 999999
		for i in range(6): 
			count = (self.targets==i).sum().item() 
			if min_count>count: 
				min_count = count 
			counts.append(1/count) 

		counts = torch.tensor(counts) 
		counts*=min_count 
	
		self.weight = counts 
		print(self.weight) 

	def __getitem__(self, index): 
		return self.inputs[index], self.targets[index] 

	def __len__(self): 
		return len(self.inputs) 


if __name__ == '__main__': 

	parser = argparse.ArgumentParser(description='PCB anomaly detection')
	parser.add_argument('--batch_size', type=int, default=10)
	parser.add_argument('--num_epochs', type=int, default=100) 
	parser.add_argument('--n_layers', type=int, default=2) 
	parser.add_argument('--lr', type=float, default=1e-6) 
	parser.add_argument('--init_tr', type=float, default=0.25) 
	parser.add_argument('--final_tr', type=float, default=0.0) 
	parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0,1,2,3]) 
	parser.add_argument('--is_attn', action='store_true') 
	parser.add_argument('--mode', type=str, default='ConvConjLSTM', help='ConvLSTM, ConvConjLSTM') 

	args = parser.parse_args()

	root_dir = 'result'
	if not os.path.isdir(root_dir): 
		os.mkdir(root_dir) 

	model_dir = root_dir + '/' + ((args.mode+'_a') if args.is_attn else args.mode)
	
	if not os.path.isdir(model_dir): 
		os.mkdir(model_dir) 

	display_folder = 'display' 
	if not os.path.isdir(display_folder): 
		os.mkdir(display_folder) 

	model_file = '%s/%s'%(model_dir, 'model_dictionary.pt') 

	model = Model(args.mode, [3, 64, 64], [5,5], args.n_layers, args.is_attn)
	
	if os.path.isfile(model_file): 
		checkpoint = torch.load(model_file) 
		model.load_state_dict(checkpoint['state_dict']) 
	else:
		assert(False) 

	if torch.cuda.device_count()>1: 
		if args.gpu_ids==None: 
			print("Let's use", torch.cuda.device_count(), "GPUs!") 
			device = torch.device('cuda:0') 
		else: 
			print("Let's use", len(args.gpu_ids), "GPUs!") 
			device = torch.device('cuda:' + str(args.gpu_ids[0])) 
	
	else: 
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

	model = torch.nn.DataParallel(model, device_ids=args.gpu_ids) 
	model = model.to(device) 
	model.eval() 

	# load data 
	all_inputs = torch.load('dataset/for_visualization/vis_inputs.pt') 
	all_targets = torch.load('dataset/for_visualization/vis_targets.pt') 

	# colormaps 
	colormaps = [[1,1,1], [1,0,0], [0,1,0], [0,0,1], [1,1,0], [1, 0.4, 0.7]] 

	cm = torch.zeros(6,6, dtype=torch.float) 

	all_keys = ['Normal'] 
	for key,_ in all_inputs.items(): 
		
		print(key) 
		all_keys.append(key) 
		inputs_scaled = [] 
		all_scaled = [] 

		model.eval() 
		for inputs, targets in zip(all_inputs[key], all_targets[key]): 

			inputs_ref = inputs.clone().to(device)  
			inputs = inputs[None] 
			inputs = inputs.to(device) 
			targets = targets.to(device) 
			inputs = (inputs-0.5)/0.5 
			inputs = inputs.detach() 
			outputs = model(inputs) # 1 by 10 by 6 by h by w 
	
			offset = outputs.new(*outputs.size()).zero_() 
			offset[:,:,0] = 1.0 
			outputs += offset 

			arg_outputs = torch.argmax(outputs, dim=2)  # 1 by 10 by h by w
			arg_outputs = arg_outputs.squeeze() # 10 by h by w	

			for i in range(6): 
				for j in range(6): 
					pred = arg_outputs==j
					real = targets==i 
					cm[i][j] += (pred&real).cpu().float().sum() 
			

			# 10 by h by w 
			scaled = [] 
			inputs_colored = inputs.new(*inputs_ref.size()).zero_() 
			for k, (r,g,b) in enumerate(colormaps): 
				mask = (arg_outputs==k).float().to(device) 
				
				mask_c = (mask*255).cpu().numpy() 
				mask_c = mask_c.astype(np.uint8) 
				# inputs_scaled -> # (10 by 3 by h by w) 
				rgb_mask = torch.stack([r*mask, g*mask, b*mask], 1) # 10 by 3 by h by w 
#				scaled.append(torch.stack([r*mask, g*mask, b*mask], 1)) # 10 by 3 by h by w
				mask = mask.unsqueeze(1).expand_as(inputs_ref) 

				if k==0: 
					inputs_colored = inputs_ref*mask 
#					inputs_colored = (rgb_mask*0.7 + inputs_ref*0.3)*mask 
				else: 
					inputs_colored += (rgb_mask*0.7 + inputs_ref*0.3)*mask
					for t in range(10): 
						img = inputs_colored[t].clone() 
						img = (img.cpu().numpy().transpose(1,2,0)*255).astype(np.uint8) 
						img = img.copy() 
	
						ret, thresh = cv2.threshold(mask_c[t], 127, 255, 0) 
						im2, countours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
						cv2.drawContours(img, countours, -1, (int(r*255),int(g*255),int(b*255)), 1)  
	
						img = img.transpose(2,0,1)  
						img = torch.tensor(img, dtype=torch.float).to(device) 
						inputs_colored[t] = img/255. 

			inputs_scaled.append(inputs_colored) 

		inputs_scaled = torch.cat(inputs_scaled, 0) 
#		all_scaled = torch.cat(all_scaled, 0) # n by 3 by h by w 
		
		# make display folder 
		disp_folder = '%s/%s'%(display_folder, key) 
		if not os.path.isdir(disp_folder): 
			os.mkdir(disp_folder) 

		origin_folder = '%s/%s_origin'%(display_folder, key) 
		if not os.path.isdir(origin_folder): 
			os.mkdir(origin_folder) 
		
		# overlay
#		all_scaled = all_scaled*0.6 + inputs_scaled.to(device)*0.4

		origin = all_inputs[key].view(-1, *all_inputs[key].size()[2:]) 
		for i in range(len(inputs_scaled)): 
			torchvision.utils.save_image(origin[i], 
				'%s/%s_%04d.png'%(origin_folder, key, i)) 

		for i in range(len(inputs_scaled)): 
			torchvision.utils.save_image(inputs_scaled[i], 
				'%s/%s_%04d.png'%(disp_folder, key, i)) 


	cm_sum = torch.sum(cm, dim=1).unsqueeze(1).expand_as(cm)
	cm/=cm_sum 
	print(cm) 

	df_cm = pd.DataFrame(cm.numpy(), index = all_keys, columns = all_keys) 
	plt.figure(figsize=(10,7)) 
	sns.heatmap(df_cm, annot=True) 
	plt.savefig('cm.png') 

