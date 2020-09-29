import numpy as np
import cv2
import torch
import sys
sys.path.append('..')
from steerable.SCFpyr_PyTorch import SCFpyr_PyTorch
from perceptual.filterbank import SteerableNoSub


device = torch.device('cuda:0')
pyr_NoSub = SCFpyr_PyTorch(sub_sample = False, device = device)

path = ''
img = cv2.imread(path,0)
img_batch = torch.from_numpy(img).to(device)
img_batch = img_batch.unsqueeze(0).float().unsqueeze(0)


coeffs = pyr_NoSub.build(img_batch)

pyr_NoSub_c = SteerableNoSub()
coeffs_c = pyr_NoSub_c.buildSCFpyr(img)

tolerance = 1e-3

coeff = coeffs[0].cpu().numpy().squeeze(0)
all_close = np.allclose(coeff, coeffs_c[0], atol=tolerance)
s = np.sum(coeff-coeffs_c[0])
print('Succesful for subband {}: {}, with tolerance of {}'.format(0,all_close, tolerance))
print('Sum of difference: {}'.format(s))
for i in range(1,len(coeffs)-1):
    for j in range(len(coeffs[i])):
        coeff = coeffs[i][j].cpu().numpy().squeeze(0)
        coeff = coeff[...,0] + 1j * coeff[...,1]
        all_close = np.allclose(coeff, coeffs_c[i][j], atol=tolerance)
        s = np.sum(coeff - coeffs_c[i][j])
        print('Succesful for subband {} {}: {}'.format(i,j,all_close))
        print('Sum of difference: {}'.format(s))


coeff = coeffs[-1].cpu().numpy().squeeze(0)
all_close = np.allclose(coeff, coeffs_c[-1], atol=tolerance)
s = np.sum(coeff-coeffs_c[-1])
print('Succesful for subband {}: {}'.format(len(coeffs),all_close))
print('Sum of difference: {}'.format(s))

# reconstruction
tolerance = 1e-2
rec = pyr_NoSub.reconstruct(coeffs)
rec = rec.cpu().numpy().squeeze(0)
#rec_c = pyr_NoSub_c.reconSCFpyr(coeffs_c)
all_close = np.allclose(rec, img, atol=tolerance)
print('Succesful for reconstruction of GPU implementation: {}, with tolerance of {}'.format(all_close,tolerance))
print('Sum of difference: {}'.format((rec-img).sum()))