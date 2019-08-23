import os
import time
import torch
import argparse 
from torch import optim 
import torch.utils.data
import torchvision 
from model.ConvRRN import ST_EncDec as Model 
from model.CNN import CNN 


class Dataset(torch.utils.data.Dataset):
	def __init__(self, train=True, aug=False): 
		super(Dataset, self).__init__() 

		if train==True: 
			data_name = 'pt_dataset/train_data.pt' 
			label_name = 'pt_dataset/train_binary.pt'
		else: 
			data_name = 'pt_dataset/valid_data.pt' 
			label_name = 'pt_dataset/valid_binary.pt'

		# normalization 
		self.inputs = (torch.load(data_name)-0.5)/0.5 
		self.targets = torch.load(label_name).long() 
		
		counts = [] 

		counts = [(self.targets==0).sum().item(), 
			      (self.targets==1).sum().item()] 

		
		counts = [counts[1]/sum(counts), counts[0]/sum(counts)] 
		counts = torch.tensor(counts) 
		self.weight = counts 

		if train is True and aug is True: 
		# 1. shift 20 
			new_inputs = [self.inputs] 
			new_targets = [self.targets] 
			while True: 
				new_input = torch.zeros(self.inputs.size()) 
				new_target = torch.zeros(self.targets.size()).long() 
				# self.inputs: n by 10 by 3 by h by w 
				new_input[:,:,:,:-20] = new_inputs[-1][:,:,:,20:] 
				new_input[:,:,:,:20] = new_inputs[-1][:,:,:,-20:] 
				
				new_target[:,:,:-20] = new_targets[-1][:,:,20:] 
				new_target[:,:,:20] = new_targets[-1][:,:,-20:] 

				new_inputs.append(new_input) 
				new_targets.append(new_target) 

				if len(new_inputs)==5: 
					break 

			self.inputs = torch.cat(new_inputs, 0) 
			self.targets = torch.cat(new_targets, 0).long() 

			# 2. flip vertically, flip horizontally, rotate 180 
			
			# 2-1) flip vertically 
			new_inputs = self.inputs.chunk(self.inputs.size()[-2], 3)[::-1] 
			new_inputs = torch.cat(new_inputs, 3) 

			new_targets = self.targets.chunk(self.targets.size()[-2], 2)[::-1] 
			new_targets = torch.cat(new_targets, 2).long() 

			self.inputs = torch.cat([self.inputs, new_inputs], 0) 
			self.targets = torch.cat([self.targets, new_targets], 0) 

			# 2-2) flip horizontally 
			new_inputs = self.inputs.chunk(self.inputs.size()[-1], 4)[::-1] 
			new_inputs = torch.cat(new_inputs, 4) 

			new_targets = self.targets.chunk(self.targets.size()[-1], 3)[::-1] 
			new_targets = torch.cat(new_targets, 3) 

			self.inputs = torch.cat([self.inputs, new_inputs], 0) 
			self.targets = torch.cat([self.targets, new_targets], 0) 

		# 3. reverse order 
			new_inputs = self.inputs.chunk(self.inputs.size()[1], 1)[::-1] 
			new_inputs = torch.cat(new_inputs, 1) 

			new_targets = self.targets.chunk(self.targets.size()[1], 1)[::-1] 
			new_targets = torch.cat(new_targets, 1) 

			self.inputs = torch.cat([self.inputs, new_inputs], 0) 
			self.targets = torch.cat([self.targets, new_targets], 0) 

		if train is True: 
			print('Trainset size:') 
		else: 
			print('Validset size:') 

		print(self.inputs.size(), self.targets.size()) 


	def __getitem__(self, index): 
		return self.inputs[index], self.targets[index] 

	def __len__(self): 	
		return len(self.inputs) 


if __name__ == '__main__': 

	parser = argparse.ArgumentParser(description='PCB anomaly detection')
	parser.add_argument('--batch_size', type=int, default=10)
	parser.add_argument('--num_epochs', type=int, default=100) 
	parser.add_argument('--n_layers', type=int, default=2) 
	parser.add_argument('--lr', type=float, default=1e-4) 
#	parser.add_argument('--init_tr', type=float, default=0.25) 
#	parser.add_argument('--final_tr', type=float, default=0.0) 
	parser.add_argument('--gpu_ids', nargs='+', type=int, default=[0]) 
	parser.add_argument('--is_attn', action='store_true') 
	parser.add_argument('--aug', action='store_true') 
	parser.add_argument('--mode', type=str, default='ConvLSTM', help='ConvLSTM, ConvConjLSTM, CNN') 	

	args = parser.parse_args()

	if not os.path.isdir('visualization'):
		os.mkdir('visualization') 

	if args.mode=='CNN': 
		vis_dir = 'visualization/CNN' 
	else: 
		vis_dir = 'visualization/CRRN' 

	if not os.path.isdir(vis_dir): 
		os.mkdir(vis_dir) 

#	root_dir = 'result'
#	if not os.path.isdir(root_dir): 
#		os.mkdir(root_dir) 

#	model_dir = root_dir + '/' + ((args.mode+'_a') if args.is_attn else args.mode)
	
#	if not os.path.isdir(model_dir): 
#		os.mkdir(model_dir) 
#
#	model_file = '%s/%s'%(model_dir, 'model_dictionary.pt') 

	train_dataset = Dataset(train=True, aug=args.aug) 
	train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

	valid_dataset = Dataset(train=False) 
	valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=1)

	if args.mode=='CNN': 
		model = CNN([3, 64, 64], [5,5]) 
	else: 
		model = Model(args.mode, [3, 64, 64], [5,5], args.n_layers, args.is_attn)
	
	optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5) 
	if torch.cuda.device_count()>1: 
		if args.gpu_ids==None: 
			print("Let's use", torch.cuda.device_count(), "GPUs!") 
			device = torch.device('cuda:0') 
		else: 
			print("Let's use", len(args.gpu_ids), "GPUs!") 
			device = torch.device('cuda:' + str(args.gpu_ids[0])) 
	
	else: 
		device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') 

	print('args.gpu_ids',  args.gpu_ids)

	model = torch.nn.DataParallel(model, device_ids=args.gpu_ids) 
	model = model.to(device) 

#	criterion = torch.nn.BCELoss() 
	criterion = torch.nn.CrossEntropyLoss(weight=train_dataset.weight.to(device)) 
#
#	test_loss = [] 
#
	for epoch in range(args.num_epochs): 
#
		start_time = time.time() 
		each_train_loss = [] 
		model.train() 

		origin_inputs = [] 
		origin_outputs = [] 
		out_prob = []

		with torch.set_grad_enabled(True): 
			for inputs, targets in train_dataloader:  

				inputs = inputs.to(device) 
				targets = targets.to(device)

				if (len(inputs)!=args.batch_size):
					break 

				optimizer.zero_grad() 
				outputs = model(inputs) 

#				out_prob.append(torch.nn.Sigmoid()(outputs[:,:,1])) 
#				origin_outputs.append(targets) # batch by 10 by h by w 
				# print("targets", targets.size())
				# print("outputs", outputs.size())

				if args.mode!='CNN': 
					outputs = torch.nn.Sigmoid()(outputs.view(-1, *outputs.size()[2:])) 

				# outputs = outputs.view(outputs.size(0), -1) 
				# targets = targets.view(targets.size(0), -1) 
				
				targets = targets.view(-1, *targets.size()[2:])

				err = criterion(outputs, targets.long()) 

				err.backward() 
				torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5) 
				optimizer.step() 

				each_train_loss.append(err.item())
			
		epoch_train_loss = sum(each_train_loss)/len(each_train_loss) 

		each_valid_loss = [] 
		model.eval() 


#		origin_inputs = [] 
#		origin_outputs = [] 
#		out_prob = []

		threshold = torch.linspace(0,1,100).tolist() 	
		cm = torch.zeros(2,2, dtype=torch.float) 

		tp, fp, fn, tn = {}, {}, {}, {} 

		for th in threshold: 
			tp[th] = 0 
			fp[th] = 0
			fn[th] = 0 
			tn[th] = 0 

		# Validation 
		with torch.set_grad_enabled(False): 
			for inputs, targets in valid_dataloader: 

				inputs = inputs.to(device) 
				targets = targets.to(device) 

				if (len(inputs)!=args.batch_size):
					break 

#				origin_inputs.append(inputs) 
				origin_inputs.append(inputs*0.5+0.5) 

				outputs = model(inputs)  # batch by 10 by 2 by h by w
				outputs = torch.nn.Sigmoid()(outputs) 

				if args.mode=='CNN': 
					temp_outputs = outputs.view(-1, 10, *temp_outputs.size()[1:]) 
					out_prob.append(temp_outputs) 
				else: 
					outputs = outputs.squeeze() 
					out_prob.append(outputs) 

				origin_outputs.append(targets) # batch by 10 by h by w 

				if args.mode!='CNN': 
					outputs = outputs.view(-1, *outputs.size()[2:]) 
				targets = targets.view(-1, *targets.size()[2:])

				err = criterion(outputs, targets.long()) 
				each_valid_loss.append(err.item()) 

#				arg_outputs = torch.argmax(outputs, dim=1) 

#				for i in range(2): 
#					for j in range(2): 
#						pred = (arg_outputs==j)
#						real = (targets==i)
		
#						cm[i][j] += (pred&real).cpu().float().sum() 
 

				for th in threshold: 					
					th_code = targets.int()*2 + (outputs[:,1]>th).int()

					tp[th] += (th_code==3).sum().item() 
					fp[th] += (th_code==1).sum().item() 
					fn[th] += (th_code==2).sum().item() 
					tn[th] += (th_code==0).sum().item() 

		
		precision, recall, f1 = [], [], []

		for th in threshold:
			p = tp[th]/(tp[th]+fp[th]+1e-7) 
			r = tp[th]/(tp[th]+fn[th]+1e-7) 
			
			if p!=0 and r!=0: 
				precision.append(p) 
				recall.append(r) 
				f1_each = 2/(1/p + 1/r) 
				f1.append(f1_each) 

		print(max(f1)) 
						
		
		epoch_valid_loss = sum(each_valid_loss)/len(each_valid_loss) 

		# save out_prob 
		origin_inputs = torch.cat(origin_inputs, 0) # N by 10 by 3 by h by w 
		origin_outputs = torch.cat(origin_outputs, 0) 
		out_prob = torch.cat(out_prob, 0) # 10*k by 10 by h by w

		for i in range(len(out_prob)): 

			if not os.path.isdir('%s/iter_%03d'%(vis_dir,epoch)): 
				os.mkdir('%s/iter_%03d'%(vis_dir,epoch)) 

			scaled_file = '%s/iter_%03d/%04d.png'%(vis_dir,epoch, i) 
	
			scaled_input = torchvision.utils.make_grid(origin_inputs[i], nrow=10, padding=2, pad_value=1) 

#			output_and_target = torch.cat([out_prob[i].unsqueeze(1), origin_outputs[i].unsqueeze(1).float()], 0) 

			output_and_target = torch.cat([origin_outputs[i].float(), out_prob[i][:,1]], 0).unsqueeze(1) 


#			scaled_result = torchvision.utils.make_grid(out_prob[i].unsqueeze(1), nrow=10, padding=2, pad_value=1) 
			scaled_result = torchvision.utils.make_grid(output_and_target, nrow=10, padding=2, pad_value=1) 

			
			scaled = torch.cat([scaled_input, scaled_result], 1) 

			torchvision.utils.save_image(scaled, scaled_file) 
		
		print('Epoch %03d:'%(epoch)) 
		print('-> Train loss: %f'%(epoch_train_loss)) 
		print('-> Valid loss: %f'%(epoch_valid_loss)) 
		print(' -> Confusion matrix') 
#
#		print(epoch, 'train: ', epoch_train_loss, ', test: ', epoch_valid_loss, 'correct: ', correct/all_count, 'elapsed: %4.2f'%(time.time()-start_time)) 
#
#		cm_sum = torch.sum(cm, dim=1).unsqueeze(1).expand_as(cm)
#		cm/=cm_sum 
#		print(cm) 
#
#		model_dictionary = {'epoch': epoch, 
#			'test_loss': test_loss, 
#			'state_dict': list(model.children())[0].state_dict(),
#			'optimizer': optimizer.state_dict()
#		} 
#
#		torch.save(model_dictionary, model_file) 
#	
