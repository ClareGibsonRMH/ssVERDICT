import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib

import train

# load predictions

X_real_pred = train.X_real_pred.numpy() 
f_ic = train.f_ic.numpy()
f_ees = train.f_ees.numpy()
r = train.r.numpy()
d_ees = train.d_ees.numpy()

f_vasc = 1 - f_ic - f_ees
f_vasc = f_vasc/(f_ic + f_ees + f_vasc)
A = f_vasc
normA = A - min(A)
f_vasc = 0.2 * (normA/max(normA)) # constraining fvasc
f_ees = f_ees/(f_ic + f_ees + f_vasc)
f_ic = f_ic/(f_ic + f_ees + f_vasc)
cell = f_ic/r**3

bvals = [1e-6, 0.090, 1e-6, 0.500, 1e-6, 1.5, 1e-6, 2, 1e-6, 3]

datadir = './verdict_data_directory/' # load data - change file path accordingly
imgnii = nib.load(datadir + "verdict_data_file.nii")
imgdata = np.rot90(imgnii.get_data())
imgdim = np.shape(imgdata)
masknii = nib.load(datadir + "mask.nii.gz") # load mask
mask = np.rot90(masknii.get_data())
maskvox = np.reshape(mask,np.prod(imgdim[0:3]))

# create parameter maps from model predictions

fic_vox = np.zeros(np.shape(maskvox)) 
fic_vox[maskvox==1] = np.squeeze(f_ic[:])
fic_map = np.reshape(fic_vox,np.shape(mask))
fees_vox = np.zeros(np.shape(maskvox))
fees_vox[maskvox==1] = np.squeeze(f_ees[:])
fees_map = np.reshape(fees_vox,np.shape(mask))
r_vox = np.zeros(np.shape(maskvox))
r_vox[maskvox==1] = np.squeeze(r[:])
r_map = np.reshape(r_vox,np.shape(mask))
dees_vox = np.zeros(np.shape(maskvox))
dees_vox[maskvox==1] = np.squeeze(d_ees[:])
dees_map = np.reshape(dees_vox,np.shape(mask))
fvasc_vox = np.zeros(np.shape(maskvox))
fvasc_vox[maskvox==1] = np.squeeze(f_vasc[:])
fvasc_map = np.reshape(fvasc_vox,np.shape(mask))
cell_vox = np.zeros(np.shape(maskvox))
cell_vox[maskvox==1] = np.squeeze(cell[:])
cell_map = np.reshape(cell_vox,np.shape(mask))

fig, ax = plt.subplots(1, 5, figsize=(7,3))

zslice = 6 # choose slice to view

# plot maps

plt0 = ax[0].imshow(fic_map[:,:,zslice], cmap='jet') 
ax[0].set_xlim(50,125)
ax[0].set_ylim(125,50)
ax[0].set_title('fIC')
ax[0].axis('off')
plt.colorbar(plt0,ax=ax[0],fraction=0.046, pad=0.04)
plt0.set_clim(0,1)

plt0 = ax[1].imshow(fees_map[:,:,zslice], cmap='jet')
plt.colorbar(plt0,ax=ax[1],fraction=0.046, pad=0.04)
plt0.set_clim(0,1)
ax[1].set_xlim(50,125)
ax[1].set_ylim(125,50)
ax[1].set_title('fEES')
ax[1].axis('off')

plt0 = ax[2].imshow(r_map[:,:,zslice], cmap='jet')
plt.colorbar(plt0,ax=ax[2],fraction=0.046, pad=0.04)
plt0.set_clim(0,15)
ax[2].set_xlim(50,125)
ax[2].set_ylim(125,50)
ax[2].set_title('R')
ax[2].axis('off')

plt0 = ax[3].imshow(dees_map[:,:,zslice], cmap='jet')
plt.colorbar(plt0,ax=ax[3],fraction=0.046, pad=0.04)
plt0.set_clim(0,3)
ax[3].set_xlim(50,125)
ax[3].set_ylim(125,50)
ax[3].set_title('dEES')
ax[3].axis('off')

plt0 = ax[4].imshow(fvasc_map[:,:,zslice], cmap='jet')
plt.colorbar(plt0,ax=ax[4],fraction=0.046, pad=0.04)
plt0.set_clim(0,0.2)
ax[4].set_xlim(50,125)
ax[4].set_ylim(125,50)
ax[4].set_title('fVASC')
ax[4].axis('off')

plt.show()

# save full maps

ficsave = nib.Nifti1Image(fic_map, np.eye(4))
nib.save(ficsave, datadir + 'fitdees/SS/fic.nii.gz')

feessave = nib.Nifti1Image(fees_map, np.eye(4))
nib.save(feessave, datadir + 'fitdees/SS/fees.nii.gz')

fvascsave = nib.Nifti1Image(fvasc_map, np.eye(4))
nib.save(fvascsave, datadir + 'fitdees/SS/fvasc.nii.gz')

rsave = nib.Nifti1Image(r_map, np.eye(4))
nib.save(rsave, datadir + 'fitdees/SS/r.nii.gz')

deessave = nib.Nifti1Image(dees_map, np.eye(4))
nib.save(deessave, datadir + 'fitdees/SS/dees.nii.gz')

