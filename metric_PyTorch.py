from __future__ import division
import numpy as np
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
import cv2
from scipy import signal
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F


def fspecial(win, sigma, device):
	"""
	2D gaussian mask - should give the same result as MATLAB's
	fspecial('gaussian',[shape],[sigma])
	"""
	shape = (win, win)
	m, n = [(ss-1.)/2. for ss in shape]
	y, x = np.ogrid[-m:m+1,-n:n+1]
	h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
	h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
	sumh = h.sum()

	if sumh != 0:
		h /= sumh
	return torch.from_numpy(h).to(device).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]


class Metric:
	def __init__(self, win=7, device=None):
		self.win = win
		self.k = fspecial(win, win/6, device)	#[out_C, in_C/group, H, W]
		self.device = torch.device('cpu') if device is None else device
		self.C = 1e-3

	def conj(self, img):
		if len(img.shape)==4:
			return img

		assert len(img.shape)==5
		return torch.stack((img[...,0], -img[...,1]), -1)

	def mul(self, img1, img2):
		assert img1.shape == img2.shape
		if len(img1.shape)==4:
			return img1*img2

		assert len(img1.shape)==5
		real = img1[...,0]*img2[...,0] - img1[...,1]*img2[...,1]
		imag = img1[...,0]*img2[...,1] + img1[...,1]*img2[...,0]
		return torch.stack((real, imag), -1)


	def conv2d(self, img, k):
		if len(img.shape)==4:
			return F.conv2d(img, k)

		assert len(img.shape) == 5
		real = F.conv2d(img[...,0], k)
		imag = F.conv2d(img[...,1], k)
		return torch.stack((real, imag), -1)

	def abs(self, img):
		if len(img.shape)==4:
			return torch.abs(img)
		assert len(img.shape) == 5
		return torch.sqrt(img[...,0]**2 + img[...,1]**2)

	def STSIM(self, img1, img2, sub_sample=True):
		assert img1.shape == img2.shape
		assert len(img1.shape) == 4  # [N,C,H,W]
		assert img1.shape[1] == 1	# gray image

		s = SCFpyr_PyTorch(sub_sample = sub_sample, device = self.device)

		pyrA = s.getlist(s.build(img1))
		pyrB = s.getlist(s.build(img2))

		stsim = map(self.pooling, pyrA, pyrB)

		return torch.mean(torch.tensor(list(stsim)))


	def pooling(self, img1, img2):
		tmp = self.compute_L_term(img1, img2) * self.compute_C_term(img1, img2) * self.compute_C01_term(img1, img2) * self.compute_C10_term(img1, img2)
		return torch.mean(tmp**0.25)

	def compute_L_term(self, img1, img2):
		mu1 = self.abs(self.conv2d(img1, self.k))
		mu2 = self.abs(self.conv2d(img2, self.k))

		Lmap = (2 * mu1 * mu2 + self.C)/( mu1 * mu1 + mu2 * mu2 + self.C)
		return Lmap

	def compute_C_term(self, img1, img2):
		mu1 = self.abs(self.conv2d(img1, self.k))
		mu2 = self.abs(self.conv2d(img2, self.k))

		sigma1_sq = F.conv2d(self.abs(img1)**2, self.k) - mu1 * mu1
		sigma1 = torch.sqrt(sigma1_sq)
		sigma2_sq = F.conv2d(self.abs(img2)**2, self.k) - mu2 * mu2
		sigma2 = torch.sqrt(sigma2_sq)

		Cmap = (2*sigma1*sigma2 + self.C)/(sigma1_sq + sigma2_sq + self.C)
		return Cmap

	def compute_C01_term(self, img1, img2):
		win = self.win
		window2 = 1/(win*(win-1)) * np.ones((win,win-1))
		window2 = torch.from_numpy(window2).to(self.device).float().unsqueeze(0).unsqueeze(0)

		if len(img1.shape)==4:
			img11 = img1[..., :-1]
			img12 = img1[..., 1:]
			img21 = img2[..., :-1]
			img22 = img2[..., 1:]
		else:
			img11 = img1[..., :-1, :]
			img12 = img1[..., 1: , :]
			img21 = img2[..., :-1, :]
			img22 = img2[..., 1: , :]

		mu11 = self.conv2d(img11, window2)
		mu12 = self.conv2d(img12, window2)
		mu21 = self.conv2d(img21, window2)
		mu22 = self.conv2d(img22, window2)

		sigma11_sq = F.conv2d(self.abs(img11)**2, window2) - self.abs(mu11)**2
		sigma12_sq = F.conv2d(self.abs(img12)**2, window2) - self.abs(mu12)**2
		sigma21_sq = F.conv2d(self.abs(img21)**2, window2) - self.abs(mu21)**2
		sigma22_sq = F.conv2d(self.abs(img22)**2, window2) - self.abs(mu22)**2

		sigma1_cross = self.conv2d(self.mul(img11,self.conj(img12)), window2) - self.mul(mu11,self.conj(mu12))
		sigma2_cross = self.conv2d(self.mul(img21,self.conj(img22)), window2) - self.mul(mu21,self.conj(mu22))

		if len(img1.shape) == 4:
			rho1 = (sigma1_cross + self.C) / (torch.sqrt(sigma11_sq * sigma12_sq) + self.C)
			rho2 = (sigma2_cross + self.C) / (torch.sqrt(sigma21_sq * sigma22_sq) + self.C)
		else:
			rho1 = 1/(torch.sqrt(sigma11_sq * sigma12_sq) + self.C)	#[N,C,H,W]
			rho1 = torch.stack([(sigma1_cross[...,0] + self.C) * rho1, sigma1_cross[...,1]*rho1], -1)					##[N,C,H,W,2]
			rho2 = 1/(torch.sqrt(sigma21_sq * sigma22_sq) + self.C)
			rho2 = torch.stack([(sigma2_cross[...,0] + self.C) * rho2, sigma2_cross[...,1]*rho2], -1)
		C01map = 1 - 0.5*self.abs(rho1 - rho2)

		return C01map

	def compute_C10_term(self, img1, img2):
		win = self.win
		window2 = 1/(win*(win-1)) * np.ones((win-1,win))
		window2 = torch.from_numpy(window2).to(self.device).float().unsqueeze(0).unsqueeze(0)

		if len(img1.shape) == 4:
			img11 = img1[:,:, :-1, :]
			img12 = img1[:,:, 1: , :]
			img21 = img2[:,:, :-1, :]
			img22 = img2[:,:, 1: , :]
		else:
			img11 = img1[:, :, :-1, :, :]
			img12 = img1[:, :, 1: , :, :]
			img21 = img2[:, :, :-1, :, :]
			img22 = img2[:, :, 1: , :, :]

		mu11 = self.conv2d(img11, window2)
		mu12 = self.conv2d(img12, window2)
		mu21 = self.conv2d(img21, window2)
		mu22 = self.conv2d(img22, window2)

		sigma11_sq = self.conv2d(self.abs(img11)**2, window2) - self.abs(mu11)**2
		sigma12_sq = self.conv2d(self.abs(img12)**2, window2) - self.abs(mu12)**2
		sigma21_sq = self.conv2d(self.abs(img21)**2, window2) - self.abs(mu21)**2
		sigma22_sq = self.conv2d(self.abs(img22)**2, window2) - self.abs(mu22)**2

		sigma1_cross = self.conv2d(self.mul(img11,self.conj(img12)), window2) - self.mul(mu11,self.conj(mu12))
		sigma2_cross = self.conv2d(self.mul(img21,self.conj(img22)), window2) - self.mul(mu21,self.conj(mu22))

		if len(img1.shape) == 4:
			rho1 = (sigma1_cross + self.C)/(torch.sqrt(sigma11_sq)*torch.sqrt(sigma12_sq) + self.C)
			rho2 = (sigma2_cross + self.C)/(torch.sqrt(sigma21_sq)*torch.sqrt(sigma22_sq) + self.C)
		else:
			rho1 = 1 / (torch.sqrt(sigma11_sq) * torch.sqrt(sigma12_sq) + self.C)
			rho1 = torch.stack([(sigma1_cross[...,0] + self.C) * rho1, sigma1_cross[...,1]*rho1], -1)
			rho2 = 1 / (torch.sqrt(sigma21_sq) * torch.sqrt(sigma22_sq) + self.C)
			rho2 = torch.stack([(sigma2_cross[...,0] + self.C) * rho2, sigma2_cross[...,1]*rho2], -1)
		C10map = 1 - 0.5*self.abs(rho1 - rho2)

		return C10map

	def compute_cross_term(self, img11, img12, img21, img22):
		# not used yet
		window2 = 1/(self.win**2)*np.ones((self.win, self.win))
		window2 = torch.from_numpy(window2).to(self.device).float().unsqueeze(0).unsqueeze(0)

		mu11 = self.conv2d(img11, window2)
		mu12 = self.conv2d(img12, window2)
		mu21 = self.conv2d(img21, window2)
		mu22 = self.conv2d(img22, window2)

		sigma11_sq = self.conv2d(self.mul(img11,img11), window2) - self.mul(mu11,mu11)
		sigma12_sq = self.conv2d(self.mul(img12,img12), window2) - self.mul(mu12,mu12)
		sigma21_sq = self.conv2d(self.mul(img21,img21), window2) - self.mul(mu21,mu21)
		sigma22_sq = self.conv2d(self.mul(img22,img22), window2) - self.mul(mu22,mu22)
		sigma1_cross = self.conv2d(self.mul(img11,img12), window2) - self.mul(mu11,mu12)
		sigma2_cross = self.conv2d(self.mul(img21,img22), window2) - self.mul(mu21,mu22)

		rho1 = (sigma1_cross + self.C)/(torch.sqrt(sigma11_sq*sigma12_sq) + self.C)
		rho2 = (sigma2_cross + self.C)/(torch.sqrt(sigma21_sq*sigma22_sq) + self.C)

		Crossmap = 1 - 0.5*self.abs(rho1 - rho2)
		return Crossmap