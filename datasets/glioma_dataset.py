# dataloader
'''
raw_dataset:
    ├── train  # (27~46)共20个样本
    │   ├── data
    │   │   ├── volume-27.nii
    │   │   ├── volume-28.nii ...
    │   └── label
    │       ├── segmentation-27.nii
    │       ├── segmentation-28.nii ...
    │       
    ├── test # 0~26和47~131）共111个样本
    │   ├── data
    │   │   ├── volume-0.nii
    │   │   ├── volume-1.nii ...
    │   └── label
    │       ├── segmentation-0.nii
    │       ├── segmentation-1.nii ...
    '''
import os  # 管理文件路径
import os.path as osp

import numpy as np
import pandas as pd
# https://zhuanlan.zhihu.com/p/113318562
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2

#这个是医学图像最重要的一个库Nibabal / SimpleITK ---->ITK-SNAP
#import nibabel
#import SimpleITK as sitk

#对所有的npy文件做resize
#强制数据格式能保证数据可读

class GliomaDataset(Dataset):

	def __init__(self, args, type):
		self.args = args #args里面包含了dataset.path, training_data.txt | self.load_file
		data_dir = osp.join(args.root, 'raw')
		self.path = osp.join(data_dir, type)
		#self.label_path = osp.join(args.root, 'node_feature.csv')
		self.label_path = osp.join(args.root, 'node_feature_1.csv')
		self.type = type
            #if self.type == 'train':
            #self.num = 28
            ####################   修改   ######################
		if self.type == 'train_val':
			self.num = 35
		elif self.type == 'train':
			self.num = 28
		else:
			self.num = 7

		#self.filename_list = self.load_file(os.path.join(data_dir, type)) #完整的一个路径名,
		self.data_list = self.process()

		#下一步是做数据增强    ##################   怎么做合适   ################
		'''
		self.transforms = Compose([
			RandomCrop(self.args.crop_size), #crop_size在args里面指出 只需要这个
			RandomFlip_LR(prob=0.5),         #翻转
			RandomCrop_UD(prob=0.5),         #部分信息（多实例）
			])
		'''
		#数据增强-->增加数据

	def process(self):
		files = os.listdir(self.path)  # 得到文件夹下的所有文件名称
		aug_1_list = []  #
		aug_2_list = []  #
		img_list = []  #
		mask_list = []  #
		aug_1_k = 0
		aug_2_k = 0
		img_k = 0
		mask_k = 0
		for index in range(self.num):
			# 数据编号从1开始
			if self.type == 'val':
				index += 29
			elif self.type == 'test':
				index += 36
			else:
				index += 1
			index_str = str(index)
			for file in files:
				if file != '.DS_Store':
					# 获取图片编号，str
					digit_filter = filter(str.isdigit, file)
					digit_list = list(digit_filter)
					digit_str = "".join(digit_list)
					# 区分1～9和10～
					if digit_str[1]=='0':
						id = digit_str[2]
					else:
						id = digit_str[1:3]
					# 根据对应编号取图像
					if id==index_str:
						img = Image.open(osp.join(self.path, file))
						#height = 670(up20down25), width = 752(-40)
						img = img.crop((40, 20, 752-40, 670-24))
						img = img.resize([313,336], Image.ANTIALIAS)
						# 将rgb图像转化为单通道
						img_gray = self.change_image_channels(img)
						file_np = np.array(img_gray)        #0102.png
						#file_np = self.crop(file_np)
						#print("file_np:", file_np[:5, :5])
					###############    开始区分是那一类数据：0，1，2，3   ###############
						# 生成多模态的aug1数据
						if digit_str[3] == '0':
							channel = int(digit_str[0])
							# 该类数据采集到的第一张图像生成空矩阵
							if aug_1_k % 3 == 0:
								H, W = file_np.shape
								aug_1_Data = np.zeros((3, H, W))  # 返回一个跟file同尺度的三通道空矩阵，3*H*W
								aug_1_Data[channel,:,:] = file_np
								#print("aug_1_Data:", aug_1_Data[:, :5, :5])
							else:
								aug_1_Data[channel, :, :] = file_np
							aug_1_k += 1
							if aug_1_k % 3==2:
								aug_1_list.append(aug_1_Data)
						# 生成多模态的img数据
						if digit_str[3] == '1':
							channel = int(digit_str[0])
							if img_k % 3 == 0:
								H, W = file_np.shape
								img_Data = np.zeros((3, H, W))  # 返回一个跟file同尺度的三通道空矩阵，3*H*W
								img_Data[channel,:,:] = file_np
							else:
								img_Data[channel, :, :] = file_np
							img_k += 1
							if img_k % 3 ==2:
								img_list.append(img_Data)
						# 生成多模态的aug2数据
						if digit_str[3] == '2':
							channel = int(digit_str[0])
							if aug_2_k % 3 == 0:
								H, W = file_np.shape
								aug_2_Data = np.zeros((3, H, W))  # 返回一个跟file同尺度的三通道空矩阵，3*H*W
								aug_2_Data[channel,:,:] = file_np
							else:
								aug_2_Data[channel, :, :] = file_np
							aug_2_k += 1
							if aug_2_k % 3==2:
								aug_2_list.append(aug_2_Data)
						# 生成多模态的mask数据
						if digit_str[3] == '3':
							channel = int(digit_str[0])
							if mask_k % 3 == 0:
								H, W = file_np.shape
								mask_Data = np.zeros((3, H, W))  # 返回一个跟file同尺度的三通道空矩阵，3*H*W
								mask_Data[channel,:,:] = file_np
							else:
								mask_Data[channel, :, :] = file_np
							mask_k += 1
							if mask_k % 3==2:
								mask_list.append(mask_Data)
				#print("index: aug_1_k, aug_2_k, img_k, mask_k:", index, aug_1_k, aug_2_k, img_k, mask_k)
		return aug_1_list, img_list, aug_2_list, mask_list

#training_data.txt:
	def __getitem__(self, index):
		#training_data = sitk.ReadImage(self.filename_list[index][0], sitk.sitkInt16)
		#training_label = sitk.ReadImage(self.filename_list[index][1], sitk.sitkInt8)
		node_feature = pd.read_csv(self.label_path)
		############  4个分类任务  ##########
		#label_list = node_feature["level"]
		#label_list = node_feature["HL"]
		#label_list = node_feature["ATRX"]
		label_list = node_feature["IDH1"]
		'''
		if self.type == 'train':
			label_list = label_list[:28]
			label = np.array(label_list[index])
			#print("label_list:", label_list)
		elif self.type == 'val':
			label_list = label_list[28:35]
			label = np.array(label_list[index+28])
			#print("label_list:", label_list)
		else:
			label_list = label_list[35:]
			label = np.array(label_list[index+35])
			#print("label_list:", label_list)
		'''
		if self.type == 'train_val':
			label_list = label_list[:35]
			label = np.array(label_list[index])
		else:
			label_list = label_list[35:]
			label = np.array(label_list[index+35])
			#print("label_list:", label_list)
				
		aug_1_list, img_list, aug_2_list, mask_list = self.data_list
		aug_1 = aug_1_list[index]
		#print("aug_1:", aug_1[:,:2,:2])
		img = img_list[index]
		aug_2 = aug_2_list[index]
		mask = mask_list[index]

		#normalization   self.args.norm_factor = 2或0。8
		aug_1 = aug_1 / self.args.norm_factor
		aug_1 = aug_1.astype(np.float32) #申明数据类型
		img = img / self.args.norm_factor
		img = img.astype(np.float32) #申明数据类型
		aug_2 = aug_2 / self.args.norm_factor
		aug_2 = aug_2.astype(np.float32) #申明数据类型
		#
		aug_1 = torch.FloatTensor(aug_1)
		img = torch.FloatTensor(img)
		aug_2 = torch.FloatTensor(aug_2)
		mask = torch.FloatTensor(mask)
		#X = Variable(X).float()
		label = torch.from_numpy(label).long()
		#aug_1, img, aug_2, mask = self.crop(aug_1, img, aug_2, mask)
		#print("index, aug_1, img, aug_2, mask, label:",index, aug_1[:, 100:103, 100:103], img[:, 100:103, 100:103], aug_2[:, 100:103, 100:103], mask[:, 100:103, 100:103])
		# 教训！以后有问题立刻print
		#print("index, label:", index, label)

		return aug_1, img, aug_2, mask, label

	def __len__(self):
		#0,1,2,3任意一个都可以
		return len(self.data_list[1])

	def change_image_channels(self, image):
		# 3通道转单通道
		if image.mode == 'RGB':
			r, g, b = image.split()  # 分离通道函数
		return b

	def crop(self, img):
		h, w = img.shape[-2:]
		#height = 670(up20down25), width = 752(-40)
		img = img[:, 20:h-25, 40:w-40]

		return img

	'''
	def crop(self, aug_1, img, aug_2, mask):
		h, w = img.shape[-2:]
		#height = 670(up20down25), width = 752(-40)
		aug_1 = aug_1[:, 20:h-25, 40:w-40]
		img = img[:, 20:h-25, 40:w-40]
		aug_2 = aug_2[:, 20:h-25, 40:w-40]
		mask = mask[:, 20:h-25, 40:w-40]

		return aug_1, img, aug_2, mask
	
	
#https://blog.csdn.net/qq_36653505/article/details/84728855 
	if __name__ = '__main__': #.size() = .shape
		sys.append('...')

		training_dict = Dataloader(training_data, num_workers = 1)
		for i, (data, label) in enumerate(training_dict):
			print(i, data.size(), label.size())

	'''





