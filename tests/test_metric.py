import numpy as np
import cv2
import torch
import sys
sys.path.append('..')
from metric_PyTorch import Metric
#from perceptual.metric import Metric

device = torch.device('cuda:0')

m_g = Metric(device = device)
#m_c = Metric()

path = ''
img1 = cv2.imread(path,0)
img1_batch = torch.from_numpy(img1).to(device)
img1_batch = img1_batch.unsqueeze(0).float().unsqueeze(0)

path = ''
img2 = cv2.imread(path,0)
img2_batch = torch.from_numpy(img2).to(device)
img2_batch = img2_batch.unsqueeze(0).float().unsqueeze(0)


#m_c.STSIM(img1,img2)
m_g.STSIM(img1_batch,img2_batch)
