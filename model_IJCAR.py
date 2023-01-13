
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch.optim as optim



torch.backends.cudnn.enabled = False
torch.manual_seed(42)


		
class TDSNet(nn.Module):
	def __init__(self):
		super(TDSNet, self).__init__()
		
		self.BN1_mri = torch.nn.BatchNorm3d(1)
		self.C1 = torch.nn.Conv3d(1, 8, 3, padding=1)
		self.BN2_mri = torch.nn.BatchNorm3d(8)
		self.LR1_mri = torch.nn.LeakyReLU()
		self.p_C1 = torch.nn.Conv3d(8, 8, 3, padding=0,stride=2)
		self.C2 = torch.nn.Conv3d(8, 16, 3, padding=1)
		self.BN3_mri = torch.nn.BatchNorm3d(16)
		self.LR2_mri = torch.nn.LeakyReLU()
		self.p_C2 = torch.nn.Conv3d(16, 16, 3, padding=0,stride=2)
		self.C3 = torch.nn.Conv3d(16, 32, 3, padding=1)
		self.BN4_mri = torch.nn.BatchNorm3d(32)
		self.LR3_mri = torch.nn.LeakyReLU()
		self.p_C3 = torch.nn.Conv3d(32, 32, 3, padding=0,stride=2)
		self.C4 = torch.nn.Conv3d(32, 32, 3, padding=1)
		self.BN5_mri = torch.nn.BatchNorm3d(32)
		self.LR4_mri = torch.nn.LeakyReLU()
		
		self.BN6_mri = torch.nn.BatchNorm3d(16)
		self.LR5_mri = torch.nn.LeakyReLU()
		self.Flat = torch.nn.Flatten()
		self.D1_mri = torch.nn.Linear(32*15*15*9, 512)
		self.BN7_mri = torch.nn.BatchNorm1d(512)
		self.LR6_mri = torch.nn.LeakyReLU()
		self.D2_mri = torch.nn.Linear(512, 512)
		self.BN8_mri = torch.nn.BatchNorm1d(512)
		self.LR7_mri = torch.nn.LeakyReLU()
		
		self.Pool = torch.nn.AvgPool3d(3,2)
		
	def forward(self, mri):
		#print("0",mri.shape)
		x = self.BN1_mri(mri)
		x = self.C1(x)
		x = self.BN2_mri(x)
		x = self.LR1_mri(x)
		x = self.p_C1(x)
		#print("1",x.shape)

		x = self.C2(x)
		x = self.BN3_mri(x)
		x = self.LR2_mri(x)
		x = self.p_C2(x)
		#print("2",x.shape)

		x = self.C3(x)
		x = self.BN4_mri(x)
		x = self.LR3_mri(x)
		x = self.p_C3(x)
		#print("3",x.shape)
		
		x = self.C4(x)
		x = self.BN5_mri(x)
		x = self.LR4_mri(x)
		#print("4",x.shape)

		x = self.Flat(x)
		x = self.D1_mri(x)
		x = self.BN7_mri(x)
		x = self.LR6_mri(x)
		x = self.D2_mri(x)
		x = self.BN8_mri(x)
		x = self.LR7_mri(x)
		
		return x

class ClinNet(nn.Module):
	def __init__(self):
		super(ClinNet, self).__init__()
		
		self.BN1_cl = torch.nn.BatchNorm1d(8)
		self.D1_cl = torch.nn.Linear(8, 512)
		self.BN2_cl = torch.nn.BatchNorm1d(512)
		self.LR1_cl = torch.nn.LeakyReLU()
		self.D2_cl = torch.nn.Linear(512, 512)
		self.BN3_cl = torch.nn.BatchNorm1d(512)
		self.LR2_cl = torch.nn.LeakyReLU()
		
		self.Drop = torch.nn.Dropout(0.3)
		
	def forward(self, cl):
		x = self.BN1_cl(cl)
		x = self.D1_cl(x)
		x = self.BN2_cl(x)
		x = self.LR1_cl(x)
		x = self.Drop(x)
		x = self.D2_cl(x)
		x = self.BN3_cl(x)
		x = self.LR2_cl(x)
		
		return x
		
class MultiNet(nn.Module):
	def __init__(self):
		super(MultiNet, self).__init__()
		self.clin_module = ClinNet()
		self.mri_module = TDSNet()
		self.BN1_both = torch.nn.BatchNorm1d(1024)
		self.LR1_both = torch.nn.LeakyReLU()
		self.D1_both = torch.nn.Linear(1024, 512)
		self.BN2_both = torch.nn.BatchNorm1d(512)
		self.LR2_both = torch.nn.LeakyReLU()
		self.D2_both = torch.nn.Linear(512, 512)
		self.BN3_both = torch.nn.BatchNorm1d(512)
		self.Drop = torch.nn.Dropout(0.5)
		
	def forward_once(self, mri, cl):
		mri = self.mri_module(mri)
		cl = self.clin_module(cl)
		x = torch.cat((mri, cl), 1)
		del mri
		del cl
		x = self.Drop(x)
		x = self.D1_both(x)
		x = self.Drop(x)
		x = self.D2_both(x)
		return x
		
	def forward(self, l_mri, r_mri, l_cl, r_cl):
		left = self.forward_once(l_mri, l_cl)
		right = self.forward_once(r_mri, r_cl)
		return left, right


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def count_all_parameters(model):
	return sum(p.numel() for p in model.parameters())


# Create model
tdsnet = MultiNet()

for name, param in tdsnet.named_parameters():
    if param.requires_grad:
        if "mri_module" in name: # freeze MRI module
        	param.requires_grad = False
        
print("All parameters: ",count_all_parameters(tdsnet))
print("Trainable parameters: ",count_trainable_parameters(tdsnet))


