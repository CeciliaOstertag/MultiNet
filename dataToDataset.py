"""
<Script to read Nifty data and corresponding labels in CSV file, and save data as binary file>
    Copyright (C) <2019>  <Cecilia Ostertag>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

import os
import sys
import gc
from matplotlib import pyplot as plt
import numpy as np
import cv2
import argparse
import nibabel as nib
import skimage
import scipy
from scipy.ndimage import zoom
from random import shuffle
import glob
from deepbrain import Extractor
from datetime import datetime
import torch


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
	
def saveExampleImage(image,title):
	volume = image.reshape(image.shape[0],image.shape[1],image.shape[2])
	proj0 = np.mean(volume, axis=0)
	cv2.imwrite(title+".jpg",proj0)

def loadNifti(imgname, ext):
	img = nib.load(imgname)
	data = img.get_fdata()
	data = ((data - data.min()) * (1/(data.max() - data.min()) * 255.))
	data = cropAroundBrain(data, ext)
	data = np.asarray(data).astype(np.float32).reshape((data.shape[0], data.shape[1], data.shape[2]))
	zmax = 150
	xmax = 205
	ymax = 216
	if data.shape[0] < xmax: #zero padding one side if z < zmax
		data=np.pad(data, ((((xmax-data.shape[0])//2)+((xmax-data.shape[0])%2),((xmax-data.shape[0])//2)), (0,0), (0,0)), 'minimum')
	if data.shape[1] < ymax: #zero padding one side if z < zmax
		data=np.pad(data, ((0,0), (((ymax-data.shape[1])//2)+((ymax-data.shape[1])%2),((ymax-data.shape[1])//2)), (0,0)), 'minimum')
	if data.shape[2] < zmax: #zero padding one side if z < zmax
		data=np.pad(data, ((0,0), (0,0), (((zmax-data.shape[2])//2)+((zmax-data.shape[2])%2),((zmax-data.shape[2])//2))), 'minimum')
	#data = zoom(data,(0.5,0.5,0.5))
	#mask = range(0,zmax,2)
	#data = np.delete(data,mask, axis=2)
	data = data / 255.
	data = zoom(data,(0.5,0.5,0.5))
	return data

def cropAroundBrain(image, ext):
	prob = ext.run(image) 
	mask = prob > 0.6
	np.putmask(image, prob < 0.6, np.min(image))
	mask = mask.astype(np.uint8)
	D00, H00, D01, H01 = rectangleAroundBrainInAxis(mask, 0)
	W00, D10, W01, D11 = rectangleAroundBrainInAxis(mask, 1)
	W10, H10, W11, H11 = rectangleAroundBrainInAxis(mask, 2)
	W0 = min(D00, W00)
	W1 = max(D01, W01)
	D0 = min(W10, H00)
	D1 = max(W11, H01)
	H0 = min(H10, D10)
	H1 = max(H11, D11)
	o = 10
	croped = image[H0-o: H0+H1+o, D0-o: D0+D1+o, W0-o: W0+W1+o]
	return croped

def rectangleAroundBrainInAxis(mask, axis):
	proj = np.max(mask, axis=axis)
	contours, hierarchy = cv2.findContours(proj,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	cont_sorted = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
	x,y,w,h = cv2.boundingRect(cont_sorted[0])
	
	return x,y,w,h
	
def getYear(name):
	return int(name[:4])

ext = Extractor()
traj = {}
clinbl = {}
clinm06 = {}
clinm12 = {}
f = open("/home/cecilia/LSN/legacy_code/ADNI_trajectory_labels_4class_MMSE_3cstp_from_m72_autoselect.csv")
for line in f.readlines():
	fields=line.split(",")
	try:
		traj[fields[1]]=int(fields[17][0])
	except ValueError:
		continue
f.close()

f = open("/home/cecilia/clinical4.csv")
for i,line in enumerate(f.readlines()):
	if i == 0:
		continue
	fields=line.split(",")
	#print(fields[0])
	if "bl" in fields[1]:
		#print("bl")
		clinbl[fields[0]]=fields[2:]
	elif "m06" in fields[1]:
		#print("m06")
		clinm06[fields[0]]=fields[2:]
	elif "m12" in fields[1]:
		#print("m12")
		clinm12[fields[0]]=fields[2:]
f.close()

path = "/home/cecilia/ADNI/"
labelstrain = []
labelsval = []
labelstest = []
print("NB OF SUBJECTS: ",len(os.listdir(path)))
#f=open("visits_repartition.csv","w")
f = open("data_mri_train2","wb")
f2 = open("data_mri_val2","wb")
f3 = open("data_mri_test2","wb")
nbval = 0
nbtest = 0
nbtrain = 0
nbm06 = 0
nbm12 = 0
fsubj = open("subjects.csv","w")
count = 0
for i,filename in enumerate(os.listdir(path)):
	found = False
	print("\t"+filename)
	if i < len(os.listdir(path)):
		#print(filename)
		dates = []
		tmpname = None
		components_img1 = None
		try:
			list1 = os.listdir(path+filename+"/MPR__GradWarp__B1_Correction__N3__Scaled/")
		except FileNotFoundError:
			list1 = []
		liste = list1
		liste = sorted(liste,key=getYear)
		if len(liste) > 1:
			for filename2 in liste:
				#print(filename2)
				func_filenames = []
				date = datetime(int(filename2[:4]),int(filename2[5:7]),int(filename2[8:10]))
				if len(dates) == 0:
					try:
						tmpname = glob.glob(path+filename+"/*/"+filename2+"/*/*.nii")[0]
					except IndexError:
						continue
					dates.append(date)
				if len(dates) == 1:
					laps = date - dates[0]
					#print("jours ",laps.days)
					#f.write(str(laps.days)+"\n")
					
					if laps.days > 100 and laps.days < 250:
						try:
						
							tmp = loadNifti(tmpname,ext)
							result_img = loadNifti(glob.glob(path+filename+"/*/"+filename2+"/*/*.nii")[0],ext)
						except IndexError:
							continue
						except OSError:
							continue
							

							
						try:
							print(filename)
							t = traj[filename]
							print(t)
							print(tmp.shape)
							bl = np.asarray(clinbl[filename],dtype=np.float32)
							m06 = np.asarray(clinm06[filename],dtype=np.float32)
							assert (tmp.shape == (102,108,75) and result_img.shape == (102,108,75))
							
							"""
							fout.write(np.asarray(t).tobytes())
							
							fout.write(bl.tobytes())
							fout.write(m06.tobytes())

							fout.write(tmp.tobytes())

							fout.write(result_img.tobytes())
							
							fsubj.write(filename+"\n")
							nbm06 += 1
							"""
							
							torch.save(bl, os.path.join("/home/cecilia/ADNI_mri/", str(count)+"_clinbl_"+filename+'_L'+str(t)+'.pt'))
							torch.save(m06, os.path.join("/home/cecilia/ADNI_mri/", str(count)+"_clinm06_"+filename+'_L'+str(t)+'.pt'))
							torch.save(tmp, os.path.join("/home/cecilia/ADNI_mri/", str(count)+"_mrbl_"+filename+'_L'+str(t)+'.pt'))
							torch.save(result_img, os.path.join("/home/cecilia/ADNI_mri/", str(count)+"_mrm06_"+filename+'_L'+str(t)+'.pt'))
							print("SAVED")
							fsubj.write(filename+"\n")
							found = True
							nbm06 += 1
							print("M06")

						except KeyError:
							print("Error")
							continue
						except AssertionError:
							print("Assertion error")
							continue
						except OSError:
							continue
					
					if laps.days > 300 and laps.days < 450:
						try:
						
							tmp = loadNifti(tmpname,ext)
							result_img = loadNifti(glob.glob(path+filename+"/*/"+filename2+"/*/*.nii")[0],ext)
						except IndexError:
							print("Error")
							continue
						except OSError:
							print("Error")
							continue
							
							
						try:
							print(filename)
							t = traj[filename]
							print(t)
							print(tmp.shape)
							bl = np.asarray(clinbl[filename],dtype=np.float32)
							m12 = np.asarray(clinm12[filename],dtype=np.float32)
							assert (tmp.shape == (102,108,75) and result_img.shape == (102,108,75))
							

						except KeyError:
							print("Error")
							continue
						except AssertionError:
							print("Assertion error")
							continue
						except OSError:
							print("Error")
							continue
						else:	
							#fout.write(np.asarray(t,dtype=np.int64).tobytes())
								
							#fout.write(bl.tobytes())
							#fout.write(m12.tobytes())

							#fout.write(tmp.tobytes())
							#print(tmp.shape)

							#fout.write(result_img.tobytes())
							
							torch.save(bl, os.path.join("/home/cecilia/ADNI_mri/", str(count)+"_clinbl_"+filename+'_L'+str(t)+'.pt'))
							torch.save(m12, os.path.join("/home/cecilia/ADNI_mri/", str(count)+"_clinm12_"+filename+'_L'+str(t)+'.pt'))
							torch.save(tmp, os.path.join("/home/cecilia/ADNI_mri/", str(count)+"_mrbl_"+filename+'_L'+str(t)+'.pt'))
							torch.save(result_img, os.path.join("/home/cecilia/ADNI_mri/", str(count)+"_mrm12_"+filename+'_L'+str(t)+'.pt'))
							print("SAVED")
							fsubj.write(filename+"\n")
							found = True
							nbm12 += 1
							print("M12")
							
			if found == True:
				count += 1
				#p = input("pause...")

print("Nb of M06 pairs: ", nbm06)
print("Nb of M12 pairs: ", nbm12)
f.close()
print("file complete")
