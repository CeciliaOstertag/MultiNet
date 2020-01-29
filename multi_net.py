import os
import time
import numpy as np
import random
import glob
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import f1_score

import scipy.ndimage

from skimage import exposure

from sklearn.metrics import roc_curve

import csv
from itertools import zip_longest

os.environ["CUDA_VISIBLE_DEVICES"]="1"

torch.backends.cudnn.enabled = False
torch.manual_seed(42)
print(torch.random.initial_seed())

def plotExampleImage(image,title):
	fig = plt.figure(figsize=(10,2))
	plt.title(title)
	cols = 3
	rows = 1
	volume = image.reshape(image.shape[0],image.shape[1],image.shape[2])
	proj0 = np.mean(volume, axis=0)
	proj1 = np.mean(volume, axis=1)
	proj2 = np.mean(volume, axis=2)
	ax1 = fig.add_subplot(rows, cols, 1)
	ax1.title.set_text("axis 0")
	plt.imshow(proj0,cmap="gray") 
	ax2 = fig.add_subplot(rows, cols, 2)
	ax2.title.set_text("axis 1")
	plt.imshow(proj1,cmap="gray")
	ax3 = fig.add_subplot(rows, cols, 3)
	ax3.title.set_text("axis 2")
	plt.imshow(proj2,cmap="gray")


class MultiDataset(Dataset):
	def __init__(self, root_dir, train, augment = False, dataset=None, list_ids=None, list_labels=None, val_size=80, fold=0, test = False, missing_data = False, inference=False):
		self.augment = augment
		self.missing = missing_data
		self.test = test
		self.inference = inference
		start = fold * val_size
		stop = fold * val_size + val_size
		if (train == True) and (dataset == None):
			self.root_dir = root_dir
	
			self.list_files = []
			self.list_ids = []
			self.list_labels = []
			for i in range(377):
				clinbl_file = glob.glob(self.root_dir+"/"+str(i)+"_clinbl_*.pt")[0]
				m12_test = glob.glob(self.root_dir+"/"+str(i)+"_clinm12_*.pt")
				m06_test = glob.glob(self.root_dir+"/"+str(i)+"_clinm06_*.pt")
				if len(m12_test) == 1:
					clinm12_file = glob.glob(self.root_dir+"/"+str(i)+"_clinm12_*.pt")[0]
					mrbl_file = glob.glob(self.root_dir+"/"+str(i)+"_mrbl_*.pt")[0]
					mrm12_file = glob.glob(self.root_dir+"/"+str(i)+"_mrm12_*.pt")[0]
					label = int(clinbl_file[-4])
					id_ = clinbl_file[-16:-6]
					clinbl = np.asarray(torch.load(clinbl_file), dtype=np.float32)
					clinm12 = np.asarray(torch.load(clinm12_file), dtype=np.float32)
					mrbl = torch.load(mrbl_file)
					mrm12 = torch.load(mrm12_file)
					if (mrbl.size == 826200) and (mrm12.size == 826200):
						self.list_files.append((mrbl, mrm12, clinbl, clinm12, label))
						self.list_ids.append(id_)
						self.list_labels.append(label)
				
				if (len(m12_test) == 0) and (len(m06_test) == 1):
					clinm06_file = glob.glob(self.root_dir+"/"+str(i)+"_clinm06_*.pt")[0]
					mrbl_file = glob.glob(self.root_dir+"/"+str(i)+"_mrbl_*.pt")[0]
					mrm06_file = glob.glob(self.root_dir+"/"+str(i)+"_mrm06_*.pt")[0]
					label = int(clinbl_file[-4])
					id_ = clinbl_file[-16:-6]
					clinbl = np.asarray(torch.load(clinbl_file), dtype=np.float32)
					clinm06 = np.asarray(torch.load(clinm06_file), dtype=np.float32)
					mrbl = torch.load(mrbl_file)
					mrm06 = torch.load(mrm06_file)
					if (mrbl.size == 826200) and (mrm06.size == 826200):
						self.list_files.append((mrbl, mrm06, clinbl, clinm06, label))
						self.list_ids.append(id_)
						self.list_labels.append(label)
				"""		
				if (len(m06_test) == 1) and (len(m12_test) == 1):
					if (mrm06.size == 826200) and (mrm12.size == 826200):
						self.list_files.append((mrm06, mrm12, clinm06, clinm12, label))
						self.list_ids.append(id_)
						self.list_labels.append(label)
				"""

			c = list(zip(self.list_files, self.list_ids, self.list_labels))
			c = sorted(c, key=lambda tup: tup[1])
			self.list_files, self.list_ids, self.list_labels = zip(*c)
			c = list(zip(self.list_files, self.list_ids, self.list_labels))
			#seed = np.random.randint(0,1024)
			seed = 42
			#print("SEED ", seed)
			random.seed(seed)
			#print(self.list_ids)
			random.shuffle(c)
			self.list_files, self.list_ids, self.list_labels = zip(*c)
			self.dataset = self.list_files[57:]
			self.list_ids = self.list_ids[57:]
			self.list_labels = self.list_labels[57:]
			self.test = self.list_files[:57]
			self.test_ids = self.list_ids[:57]
			self.test_labels = self.list_labels[:57]
			#print(self.list_ids)
			if self.list_files[0:start] == None:
				self.data = self.list_files[stop:len(self.list_files)]
				self.labels = self.list_labels[stop:len(self.list_labels)]
				self.ids = self.list_ids[stop:len(self.list_ids)]
			
			elif self.list_files[stop:len(self.list_files)] == None:
				self.data = self.list_files[0:start]
				self.labels = self.list_labels[0:start]
				self.ids = self.list_ids[0:start]
			
			else:
				self.data = self.list_files[0:start] + self.list_files[stop:len(self.list_files)]
				self.labels = self.list_labels[0:start] + self.list_labels[stop:len(self.list_labels)]
				self.ids = self.list_ids[0:start] + self.list_ids[stop:len(self.list_ids)]
		elif (train == True) and (dataset != None):
			print('Using pre-shuffled dataset')
			self.dataset = dataset
			self.list_ids = list_ids
			self.list_labels = list_labels
			if self.dataset[0:start] == None:
				self.data = self.dataset[stop:len(self.dataset)]
				self.labels = self.list_labels[stop:len(self.list_labels)]
				self.ids = self.list_ids[stop:len(self.list_ids)]
			
			elif self.dataset[stop:len(self.dataset)] == None:
				self.data = self.dataset[0:start]
				self.labels = self.list_labels[0:start]
				self.ids = self.list_ids[0:start]
			
			else:
				self.data = self.dataset[0:start] + self.dataset[stop:len(self.dataset)]
				self.labels = self.list_labels[0:start] + self.list_labels[stop:len(self.list_labels)]
				self.ids = self.list_ids[0:start] + self.list_ids[stop:len(self.list_ids)]
		elif (train == False) and (test == False):
			print('Using pre-shuffled dataset')
			self.dataset = dataset
			self.list_ids = list_ids
			self.list_labels = list_labels
			self.data = self.dataset[start:stop]
			self.labels = self.list_labels[start:stop]
			self.ids = self.list_ids[start:stop]
		elif (train == False) and (test == True):
			print('Using pre-shuffled dataset\tTEST DATA')
			self.dataset = dataset
			self.list_ids = list_ids
			self.list_labels = list_labels
			self.data = self.dataset[:57]
			self.labels = self.list_labels[:57]
			self.ids = self.list_ids[:57]
		print("STABLE ",self.labels.count(0))
		print("DECLINE ",self.labels.count(1))
		
	def __len__(self):
		'Denotes the number of batches per epoch'
		return len(self.data)
		
	def __getitem__(self, idx):
		'Generate one batch of data'
		zmax = 150
		xmax = 205
		ymax = 216
		xmax = xmax//2
		ymax = ymax//2
		zmax = zmax//2
		imgs, imgs2, bl, m12, labels = self.data[idx]
		var_bl = bl[3:]
		var_m12 = m12[3:]
		dem = bl[:3]
		m = 0
		imgs = exposure.rescale_intensity(imgs, in_range='image' ,out_range=(0.,1.))
		imgs2 = exposure.rescale_intensity(imgs2, in_range='image' ,out_range=(0.,1.))
		
		#plotExampleImage(imgs,"1")
		#plotExampleImage(imgs2,"2")
		#plt.figure()
		#plt.imshow(imgs[45,:,:],cmap="gray")
		#plt.figure()
		#plt.imshow(imgs2[45,:,:],cmap="gray")
		if self.augment == True:
			
			sigma = torch.randint(low=0, high=8, size=(1,)).item()*0.1
			imgs = scipy.ndimage.gaussian_filter(imgs, sigma=sigma, mode='nearest')
			imgs2 = scipy.ndimage.gaussian_filter(imgs2, sigma=sigma, mode='nearest')
			
			
			angle = torch.randint(low=0, high=6, size=(1,)).item()
			angle2 = torch.randint(low=0, high=6, size=(1,)).item()
			neg = torch.randint(0,2,(1,)).item()
			neg2 = torch.randint(0,2,(1,)).item()
			if neg == 1:
				angle = - angle
			if neg2 == 1:
				angle2 = - angle2
				
			imgs= scipy.ndimage.interpolation.rotate(imgs, angle, axes=(1,2), reshape=False, mode='nearest')
			imgs2 = scipy.ndimage.interpolation.rotate(imgs2, angle2, axes=(1,2), reshape=False, mode='nearest')
			
			
			flip = torch.randint(0,5,(1,)).item()
			
			if flip == 1:
				imgs = np.flip(imgs, 2).copy()	
				imgs2 = np.flip(imgs2, 2).copy()
			
			
			
			ib = torch.randint(0,3,(1,)).item()
			ih = torch.randint(98,101,(1,)).item()
			pb, ph = np.percentile(imgs, (ib, ih))
			pb2, ph2 = np.percentile(imgs2, (ib, ih))

			imgs = exposure.rescale_intensity(imgs, in_range=(pb, ph) ,out_range=(0.,1.))
			imgs2 = exposure.rescale_intensity(imgs2, in_range=(pb2, ph2) ,out_range=(0.,1.))
			
			"""
			ow = torch.randint(0,11,(1,)).item()
			oh = torch.randint(0,11,(1,)).item()
			od = torch.randint(0,11,(1,)).item()
			W = imgs.shape[0]
			H = imgs.shape[1]
			D = imgs.shape[2]
			imgs = imgs[ow:imgs.shape[0]-ow, oh:imgs.shape[1]-oh, od:imgs.shape[2]-od]
			imgs2 = imgs2[ow:imgs2.shape[0]-ow, oh:imgs2.shape[1]-oh, od:imgs2.shape[2]-od]
			w = imgs.shape[0]
			h = imgs.shape[1]
			d = imgs.shape[2]
			imgs = scipy.ndimage.zoom(imgs,(W/w,H/h,D/d))
			imgs2 = scipy.ndimage.zoom(imgs2,(W/w,H/h,D/d))	
			"""
		#plotExampleImage(imgs,"3")
		#plotExampleImage(imgs2,"4")
		#plt.figure()
		#plt.imshow(imgs[45,:,:],cmap="gray")
		#plt.figure()
		#plt.imshow(imgs2[45,:,:],cmap="gray")
		imgs = torch.as_tensor(imgs).permute(2,0,1).view(1,75,102,108)
		imgs2 = torch.as_tensor(imgs2).permute(2,0,1).view(1,75,102,108)
		#plt.show()
		
		if self.missing == True:
			if self.inference == False:
				mask = torch.randint(0,10,(1,)).item()
				if mask == 0:
					a1 = torch.randint(0,8,(1,)).item()
					a2 = torch.randint(0,8,(1,)).item()
					a3 = torch.randint(0,8,(1,)).item()
					b1 = torch.randint(0,8,(1,)).item()
					b2 = torch.randint(0,8,(1,)).item()
					b3 = torch.randint(0,8,(1,)).item()
				
					var_bl[a1]=0
					var_bl[a2]=0
					var_m12[b1]=0
					var_m12[b2]=0
			else:
					indexes = random.sample(range(8),4) 
					for i in indexes:
						var_bl[i]=0
					indexes = random.sample(range(8),4) 
					for i in indexes:
						var_m12[i]=0

		return imgs,imgs2,var_bl,var_m12,dem,labels
		
	def getShuffledDataset(self):
		return self.dataset, self.list_ids, self.list_labels
		
	def getTestDataset(self):
		return self.test, self.test_ids, self.test_labels


class ClinNet(nn.Module):
	def __init__(self):
		super(ClinNet, self).__init__()
		
		self.BN1_cl = torch.nn.BatchNorm1d(8)
		self.D1_cl = torch.nn.Linear(8, 1024)
		self.BN2_cl = torch.nn.BatchNorm1d(1024)
		self.LR1_cl = torch.nn.LeakyReLU()
		self.D2_cl = torch.nn.Linear(1024, 1024)
		self.BN3_cl = torch.nn.BatchNorm1d(1024)
		self.LR2_cl = torch.nn.LeakyReLU()
		
		self.BN4_cl = torch.nn.BatchNorm1d(1024)
		self.LR3_cl = torch.nn.LeakyReLU()
		self.D3_cl = torch.nn.Linear(1024, 8)
		self.BN5_cl = torch.nn.BatchNorm1d(8)
		self.LR4_cl = torch.nn.LeakyReLU() 
		
		self.BN6_cl = torch.nn.BatchNorm1d(3)
		self.D4_cl = torch.nn.Linear(3, 512)
		self.BN7_cl = torch.nn.BatchNorm1d(512)
		self.LR5_cl = torch.nn.LeakyReLU()
		self.D5_cl = torch.nn.Linear(512, 512)
		self.BN8_cl = torch.nn.BatchNorm1d(512)
		self.LR6_cl = torch.nn.LeakyReLU()
		self.D6_cl = torch.nn.Linear(512, 3)
		self.BN9_cl = torch.nn.BatchNorm1d(3)
		self.LR7_cl = torch.nn.LeakyReLU()
		
		self.Drop = torch.nn.Dropout(0.5)
		self.Out = torch.nn.Linear(11, 1)
		self.Sig = torch.nn.Sigmoid()
		
	def forward_once(self, cl):
		x = self.BN1_cl(cl)
		x = self.D1_cl(x)
		x = self.BN2_cl(x)
		x = self.LR1_cl(x)
		x = self.D2_cl(x)
		x = self.BN3_cl(x)
		x = self.LR2_cl(x)
		return x
	
	def forward(self, left_cl, right_cl, dem):
		l_cl = self.forward_once(left_cl)
		r_cl = self.forward_once(right_cl)
		diff_cl = torch.abs(l_cl-r_cl)
		
		x = self.BN4_cl(diff_cl)
		x = self.LR3_cl(x)
		x = self.D3_cl(x)
		x = self.BN5_cl(x)
		x = self.LR4_cl(x)
		
		d = self.BN6_cl(dem)
		d = self.D4_cl(d)
		d = self.BN7_cl(d)
		d = self.LR5_cl(d)
		d = self.D5_cl(d)
		d = self.BN8_cl(d)
		d = self.LR6_cl(d)
		d = self.D6_cl(d)
		d = self.BN9_cl(d)
		d = self.LR7_cl(d)
		
		cl = torch.cat((x,d),1)
		out = self.Drop(cl)
		out = self.Out(out)
		out = self.Sig(out)
		return cl, out # TODO softmax
		
class TDSNet(nn.Module):
	def __init__(self):
		super(TDSNet, self).__init__()
		
		self.BN1_mri = torch.nn.BatchNorm3d(1)
		self.C1 = torch.nn.Conv3d(1, 4, 3)
		self.BN2_mri = torch.nn.BatchNorm3d(4)
		self.LR1_mri = torch.nn.LeakyReLU()
		self.C2 = torch.nn.Conv3d(4, 8, 3)
		self.BN3_mri = torch.nn.BatchNorm3d(8)
		self.LR2_mri = torch.nn.LeakyReLU()
		self.C3 = torch.nn.Conv3d(8, 16, 3)
		self.BN4_mri = torch.nn.BatchNorm3d(16)
		self.LR3_mri = torch.nn.LeakyReLU()
		self.C4 = torch.nn.Conv3d(16, 16, 3)
		self.BN5_mri = torch.nn.BatchNorm3d(16)
		self.LR4_mri = torch.nn.LeakyReLU()
		
		self.BN6_mri = torch.nn.BatchNorm3d(16)
		self.LR5_mri = torch.nn.LeakyReLU()
		self.Flat = torch.nn.Flatten()
		self.D1_mri = torch.nn.Linear(16*4*8*8, 512)
		self.BN7_mri = torch.nn.BatchNorm1d(512)
		self.LR6_mri = torch.nn.LeakyReLU()
		self.D2_mri = torch.nn.Linear(512, 256)
		self.BN8_mri = torch.nn.BatchNorm1d(256)
		self.LR7_mri = torch.nn.LeakyReLU()
		
		self.Pool = torch.nn.AvgPool3d(3,2)
		self.Drop3d = torch.nn.Dropout3d(p=0.2)
		self.Drop1 = torch.nn.Dropout(0.7)
		self.Drop2 = torch.nn.Dropout(0.7)
		self.Out = torch.nn.Linear(256,1)
		self.Sig = torch.nn.Sigmoid()
		
	def forward_once(self, mri):
		x = self.BN1_mri(mri)
		x = self.C1(x)
		x = self.BN2_mri(x)
		x = self.LR1_mri(x)
		x = self.Pool(x)
		#print(x.shape)
		#x = self.Drop3d(x)
		x = self.C2(x)
		x = self.BN3_mri(x)
		x = self.LR2_mri(x)
		x = self.Pool(x)
		#print(x.shape)
		
		#x = self.Drop3d(x)
		x = self.C3(x)
		x = self.BN4_mri(x)
		x = self.LR3_mri(x)
		x = self.Pool(x)
		#print(x.shape)
		
		#x = self.Drop3d(x)
		x = self.C4(x)
		x = self.BN5_mri(x)
		x = self.LR4_mri(x)
		
		return x
	
	def forward(self, left_mri, right_mri):
		l_mri = self.forward_once(left_mri)
		r_mri = self.forward_once(right_mri)
		diff_mri = torch.abs(torch.add(l_mri,torch.neg(r_mri)))
		#print(diff_mri.shape)
		
		#x = self.BN6_mri(diff_mri)
		#x = self.LR5_mri(x)
		x = self.Flat(diff_mri)
		x = self.D1_mri(x)
		x = self.BN7_mri(x)
		x = self.LR6_mri(x)
		x = self.Drop1(x)
		x = self.D2_mri(x)
		#x = self.BN8_mri(x)
		#x = self.LR7_mri(x)
		out = self.Drop2(x)
		out = self.Out(out)
		out = self.Sig(out)
		return x, out # TODO softmax


class MultiNet(nn.Module):
	def __init__(self):
		super(MultiNet, self).__init__()
		self.clin_module = ClinNet()
		self.mri_module = TDSNet()
		self.BN1_both = torch.nn.BatchNorm1d(267)
		self.LR1_both = torch.nn.LeakyReLU()
		self.D1_both = torch.nn.Linear(267, 256)
		self.BN2_both = torch.nn.BatchNorm1d(256)
		self.LR2_both = torch.nn.LeakyReLU()
		self.D2_both = torch.nn.Linear(256, 256)
		self.BN3_both = torch.nn.BatchNorm1d(256)
		self.LR3_both = torch.nn.LeakyReLU()
		self.Drop = torch.nn.Dropout(0.5)
		self.Out =torch.nn.Linear(256, 1)
		self.Sig = torch.nn.Sigmoid()
		
	def forward(self, l_mri, r_mri, l_cl, r_cl, dem):
		cl, _ = self.clin_module(l_cl, r_cl, dem)
		mri, _ = self.mri_module(l_mri, r_mri)
		x = torch.cat((mri, cl), 1)	
		x = self.BN1_both(x)
		x = self.LR1_both(x)
		x = self.D1_both(x)
		x = self.BN2_both(x)
		x = self.LR2_both(x)
		x = self.D2_both(x)
		x = self.BN3_both(x)
		x = self.LR3_both(x)
		x = self.Drop(x)
		x = self.Out(x)
		x = self.Sig(x)
		return x


class MultiAdaptNet(nn.Module):
	def __init__(self):
		super(MultiAdaptNet, self).__init__()
		self.clin_module = ClinNet()
		self.mri_module = TDSNet()
		self.BN1_both = torch.nn.BatchNorm1d(267)
		self.LR1_both = torch.nn.LeakyReLU()
		self.D1_both = torch.nn.Linear(267, 256)
		self.BN2_both = torch.nn.BatchNorm1d(256)
		self.LR2_both = torch.nn.LeakyReLU()
		self.D2_both = torch.nn.Linear(256, 256)
		self.BN3_both = torch.nn.BatchNorm1d(256)
		self.LR3_both = torch.nn.LeakyReLU()
		self.Drop = torch.nn.Dropout(0.5)
		self.Out = torch.nn.Linear(256, 1)
		
		self.D1 = torch.nn.Linear(1, 1)
		self.LR = torch.nn.LeakyReLU()
		self.BN = torch.nn.BatchNorm1d(512)
		self.Result = torch.nn.Sigmoid()
		
	def forward(self, l_mri, r_mri, l_cl, r_cl, dem):
		cl, outcl = self.clin_module(l_cl, r_cl, dem)
		mri, outmri = self.mri_module(l_mri, r_mri)
		
		x = torch.cat((mri, cl), 1)	
		x = self.BN1_both(x)
		x = self.LR1_both(x)
		x = self.D1_both(x)
		x = self.BN2_both(x)
		x = self.LR2_both(x)
		x = self.D2_both(x)
		x = self.BN3_both(x)
		x = self.LR3_both(x)
		x = self.Drop(x)
		x = self.Out(x)
		outmulti = F.sigmoid(x)
		x = torch.add(torch.mul(outmulti,outcl),torch.mul(outmulti,outmri))
		x = self.D1(x)
		x = self.Result(x)
		return x	

batch_size = 10
num_classes = 2
epochs = 75
val_size = 80

datatrain = MultiDataset("/data1/cecilia/ADNI_mri", train=True, augment=True, val_size=val_size)
print("Nb of training data: ",len(datatrain))
shuffled_dataset, ids, labels = datatrain.getShuffledDataset()
test_dataset, test_ids, test_labels = datatrain.getTestDataset()
print("Example of test data: ")
print(test_ids[:10])
#print(ids)
train_dataloader = DataLoader(datatrain, shuffle=True, num_workers=20,batch_size=batch_size, drop_last=True)

dataval = MultiDataset("/data1/cecilia/ADNI_mri", train = False, augment=False, dataset = shuffled_dataset, list_ids = ids, list_labels = labels, val_size=val_size)
print("Nb of validation data: ",len(dataval))
val_dataloader = DataLoader(dataval, shuffle=True, num_workers=20,batch_size=batch_size, drop_last=True)

nb_folds = (len(datatrain) + len(dataval)) // val_size
print((len(datatrain) + len(dataval)) % val_size)
print("\nRunning training with "+str(nb_folds)+"-fold validation")

def add_weight_decay(net, l2_value, skip_list=()):
	decay, no_decay = [], []
	for name, param in net.named_parameters():
		if not param.requires_grad: 
			continue # frozen weights		            
		if len(param.shape) == 1 or name.endswith(".bias") or name in skip_list:
			no_decay.append(param)
		else: decay.append(param)
	return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': l2_value}]


for fold in range(nb_folds):

	datatrain = MultiDataset("/data1/cecilia/ADNI_mri", train=True, augment=True, dataset = shuffled_dataset, list_ids = ids, list_labels = labels, val_size=val_size, missing_data=True, fold=fold)
	print("Nb of training data: ",len(datatrain))
	shuffled_dataset, ids, labels = datatrain.getShuffledDataset()
	#print(ids)
	train_dataloader = DataLoader(datatrain, shuffle=True, num_workers=20,batch_size=batch_size, drop_last=True)

	dataval = MultiDataset("/data1/cecilia/ADNI_mri", train = False, augment=False, dataset = shuffled_dataset, list_ids = ids, list_labels = labels, val_size=val_size, missing_data=True,fold=fold)
	print("Nb of validation data: ",len(dataval))
	val_dataloader = DataLoader(dataval, shuffle=True, num_workers=20,batch_size=batch_size, drop_last=True)
	
	datatest = MultiDataset("/data1/cecilia/ADNI_mri", train = False, augment=False, dataset = shuffled_dataset, list_ids = ids, list_labels = labels, val_size=val_size, test= True, missing_data=False,fold=fold)
	print("Nb of test data: ",len(datatest))
	test_dataloader = DataLoader(datatest, shuffle=True, num_workers=20,batch_size=57, drop_last=True)



	# Create model
	tdsnet = ClinNet()

	#tdsnet = torch.nn.DataParallel(tdsnet,device_ids=[0,1])
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	tdsnet = tdsnet.to(device)
	ce = torch.nn.BCELoss() #nn.CrossEntropyLoss()
	mse = nn.MSELoss()
	#params = add_weight_decay(tdsnet, 2 * 0.01)
	#optimizer = optim.Adam(tdsnet.parameters(), lr=0.000005)
	#optimizer = optim.AdamW(tdsnet.parameters(), lr=0.000005, weight_decay = 0.005 )
	optimizer = optim.AdamW(tdsnet.parameters(), lr=0.005, weight_decay = 0.2)

	loss_history = []
	valloss_history = []
	acc_history = []
	valacc_history = []
	f1_history = []
	valf1_history = [] 
	for epoch in range(epochs): 
		print ("Fold "+str(fold)+" Epoch "+str(epoch+1)+"/"+str(epochs))
		start_time = time.time()  
		total=0
		correct=0
		tot_train_f1 = 0
		tot_train_loss=0
		for i, data in enumerate(train_dataloader):
			imgs,imgs2,var_bl,var_m12,dem,label = data
			optimizer.zero_grad()
			label = label.to(device)
			imgs = imgs.to(device)
			imgs2 = imgs2.to(device)
			var_bl = var_bl.to(device)
			var_m12 = var_m12.to(device)
			dem = dem.to(device)
			_,output = tdsnet(var_bl, var_m12, dem)
		
			ce_loss = ce(output, label.to(torch.float).view(batch_size,-1))
			mse_loss = mse(output, label.to(torch.float).view(batch_size,-1))
			train_loss =  ce_loss #+ mse_loss
			train_loss.backward()
			optimizer.step()
		
			tot_train_loss += train_loss.item()
			total += label.size(0)
			predicted = np.ndarray.flatten(np.rint(output.cpu().detach().numpy())).astype(np.int)
			label = label.data.cpu().detach().numpy()
			correct += (predicted == label).sum().item()
			tot_train_f1 += f1_score(label, predicted)
		print("Training loss ",tot_train_loss/float(i+1))
		loss_history.append(tot_train_loss/float(i+1))
		print("Training acc ",float(correct)/float(total))
		acc_history.append(float(correct)/float(total))
		print("Training f1 ",float(tot_train_f1)/float(i+1))
		f1_history.append(float(tot_train_f1)/float(i+1))
	
		total=0
		correct=0
		tot_val_loss=0
		tot_val_f1 = 0
	
		with torch.no_grad():
			tdsnet.train(False)
			for j, data in enumerate(val_dataloader):
				imgs,imgs2,var_bl,var_m12,dem,label = data
				#print("true: ",label.detach().numpy())
				label = label.to(device)
				imgs = imgs.to(device)
				imgs2 = imgs2.to(device)
				var_bl = var_bl.to(device)
				var_m12 = var_m12.to(device)
				dem = dem.to(device)
				_,output = tdsnet(var_bl, var_m12, dem)
		
				ce_loss = ce(output, label.to(torch.float).view(batch_size,-1))
				mse_loss = mse(output, label.to(torch.float).view(batch_size,-1))
				val_loss =  ce_loss #+ mse_loss
			
				tot_val_loss += val_loss.item()
				total += label.size(0)
				predicted = np.ndarray.flatten(np.rint(output.cpu().detach().numpy())).astype(np.int)
				#print("pred: ",predicted)
				label = label.data.cpu().detach().numpy()
				correct += (predicted == label).sum().item()
				tot_val_f1 += f1_score(label, predicted)
		print("Val loss ",tot_val_loss/float(j+1))
		valloss_history.append(tot_val_loss/float(j+1))
		print("Val acc ",float(correct)/float(total))
		valacc_history.append(float(correct)/float(total))
		print("Val f1 ",float(tot_val_f1)/float(j+1))
		valf1_history.append(float(tot_val_f1)/float(j+1))
		
		print("Time (s): ",(time.time() - start_time))
	
		tdsnet.train()
		
	
	d = [loss_history, valloss_history,acc_history,valacc_history, f1_history,valf1_history]
	export_data = zip_longest(*d, fillvalue = '')
	with open('clinical_adapt_'+str(fold)+'.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
		  wr = csv.writer(myfile)
		  wr.writerow(("loss", "valloss","acc","valacc", "f1","valf1"))
		  wr.writerows(export_data)
	myfile.close()
	
	torch.save(tdsnet.state_dict(), "clin_adapt_net_"+str(fold)+".pt")
	del tdsnet
	print("Weights saved")
"""
########## inference

batch_size = 1	
for fold in range(nb_folds):

	datatrain = MultiDataset("/data1/cecilia/ADNI_mri", train=True, augment=True, dataset = shuffled_dataset, list_ids = ids, list_labels = labels, val_size=val_size, missing_data=False, fold=fold)
	print("Nb of training data: ",len(datatrain))
	shuffled_dataset, ids, labels = datatrain.getShuffledDataset()
	#print(ids)
	train_dataloader = DataLoader(datatrain, shuffle=True, num_workers=20,batch_size=batch_size, drop_last=True)

	dataval = MultiDataset("/data1/cecilia/ADNI_mri", train = False, augment=False, dataset = shuffled_dataset, list_ids = ids, list_labels = labels, val_size=val_size, missing_data=True,fold=fold)
	print("Nb of validation data: ",len(dataval))
	val_dataloader = DataLoader(dataval, shuffle=True, num_workers=20,batch_size=batch_size, drop_last=True)
	
	datatest = MultiDataset("/data1/cecilia/ADNI_mri", train = False, augment=False, dataset = shuffled_dataset, list_ids = ids, list_labels = labels, val_size=val_size, test= True, missing_data=True,fold=fold, inference=True)
	print("Nb of test data: ",len(datatest))
	test_dataloader = DataLoader(datatest, shuffle=True, num_workers=20,batch_size=batch_size, drop_last=True)
	
	tdsnet = MultiAdaptNet()
	tdsnet.load_state_dict(torch.load("multi_adapt_net_"+str(fold)+".pt"))
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	tdsnet = tdsnet.to(device)
	
	y_pred_list=[]
	y_test_list=[]
	with torch.no_grad():
		tdsnet.train(False)
		for k, data in enumerate(test_dataloader):
			print(k)
			imgs,imgs2,var_bl,var_m12,dem,label = data
			label = label.to(device)
			imgs = imgs.to(device)
			imgs2 = imgs2.to(device)
			print(var_bl)
			print(var_m12)
			print('\n')
			var_bl = var_bl.to(device)
			var_m12 = var_m12.to(device)
			dem = dem.to(device)
			y_pred = tdsnet(imgs, imgs2, var_bl, var_m12, dem).cpu().detach().numpy()
			y_test = label.data.cpu().detach().numpy()
			y_pred_list.append(y_pred[0][0])
			y_test_list.append(y_test[0])
	
	print("true ",y_test_list)
	print("pred ",y_pred_list)
	
	fpr, tpr, thresh = roc_curve(y_test_list, y_pred_list, drop_intermediate = False)
	
	d = [fpr, tpr, thresh]
	export_data = zip_longest(*d, fillvalue = '')
	with open('roc_'+str(fold)+'.csv', 'w', encoding="ISO-8859-1", newline='') as myfile:
		  wr = csv.writer(myfile)
		  wr.writerow(("fpr", "tpr", "thresh"))
		  wr.writerows(export_data)
	myfile.close()

	
	plt.figure()
	plt.plot(loss_history)
	plt.plot(valloss_history)
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train loss', 'Val loss'], loc='upper left')
	plt.figure()
	plt.plot(acc_history)
	plt.plot(valacc_history)
	plt.ylabel('Acc')
	plt.xlabel('Epoch')
	plt.legend(['Train acc','Val acc'], loc='lower left')
	plt.figure()
	plt.plot(f1_history)
	plt.plot(valf1_history)
	plt.ylabel('F1')
	plt.xlabel('Epoch')
	plt.legend(['Train F1', 'Val F1'], loc='upper left')
	plt.show()
	
	del tdsnet
"""

		
