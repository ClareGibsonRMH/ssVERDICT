import numpy as np
import nibabel as nib
import torch
    
def data_loader():

    # set acquisition params

    bvals = [1e-6, 0.090, 1e-6, 0.500, 1e-6, 1.5, 1e-6, 2, 1e-6, 3]
    bvalsn = [1e-6, 0.090]
    Delta = torch.FloatTensor([23.8, 23.8, 23.8, 31.3, 23.8, 43.8, 23.8, 34.3, 23.8, 38.8])
    delta = torch.FloatTensor([3.9, 3.9, 3.9, 11.4, 3.9, 23.9, 3.9, 14.4, 3.9, 18.9])
    
    gamma = 2.675987e2
    b_values = torch.FloatTensor(bvals)
    gradient_strength = torch.FloatTensor([np.sqrt(b_values[i])/(gamma*delta[i]*np.sqrt(Delta[i]-delta[i]/3)) for i,_ in enumerate(b_values)])

    datadir = './verdict_data_directory/' # load data - change file path as needed
    imgnii = nib.load(datadir + "verdict_data_file.nii")
    imgdata = np.rot90(imgnii.get_data())

    # order data: matched b0 image, then spherically averaged diffusion weighted image

    b90 = np.stack((imgdata[:,:,:,0],(imgdata[:,:,:,1] + imgdata[:,:,:,2] + imgdata[:,:,:,3])/3),axis=-1)
    b500 = np.stack((imgdata[:,:,:,4],(imgdata[:,:,:,5] + imgdata[:,:,:,6] + imgdata[:,:,:,7])/3),axis=-1)
    b1500 = np.stack((imgdata[:,:,:,8],(imgdata[:,:,:,9] + imgdata[:,:,:,10] + imgdata[:,:,:,11])/3),axis=-1)
    b2000 = np.stack((imgdata[:,:,:,12],(imgdata[:,:,:,13] + imgdata[:,:,:,14] + imgdata[:,:,:,15])/3),axis=-1)
    b3000 = np.stack((imgdata[:,:,:,16],(imgdata[:,:,:,17] + imgdata[:,:,:,18] + imgdata[:,:,:,19])/3),axis=-1)

    masknii = nib.load(datadir + "mask.nii.gz") # load mask
    mask = np.rot90(masknii.get_data())
    normvol = np.where([i == 1e-6 for i in bvalsn])
    imgdim = np.shape(b90)
    maskvox = np.reshape(mask,np.prod(imgdim[0:3]))
    nvol = imgdim[3]

    b90vox = (np.reshape(b90,(np.prod(imgdim[0:3]),imgdim[3])))[maskvox==1]
    b500vox = (np.reshape(b500,(np.prod(imgdim[0:3]),imgdim[3])))[maskvox==1]
    b1500vox = (np.reshape(b1500,(np.prod(imgdim[0:3]),imgdim[3])))[maskvox==1]
    b2000vox = (np.reshape(b2000,(np.prod(imgdim[0:3]),imgdim[3])))[maskvox==1]
    b3000vox = (np.reshape(b3000,(np.prod(imgdim[0:3]),imgdim[3])))[maskvox==1]

    # normalise by matched b0 image

    b90fit = b90vox/((np.tile(np.mean(b90vox[:,normvol], axis=2),(1, nvol))))
    b500fit = b500vox/((np.tile(np.mean(b500vox[:,normvol], axis=2),(1, nvol))))
    b1500fit = b1500vox/((np.tile(np.mean(b1500vox[:,normvol], axis=2),(1, nvol))))
    b2000fit = b2000vox/((np.tile(np.mean(b2000vox[:,normvol], axis=2),(1, nvol))))
    b3000fit = b3000vox/((np.tile(np.mean(b3000vox[:,normvol], axis=2),(1, nvol))))

    # create normalised image to fit

    imgvoxtofit = (np.stack((b90fit,b500fit,b1500fit,b2000fit,b3000fit), axis=1)).reshape(np.shape(b90fit)[0],10)

    return imgvoxtofit, b_values, Delta, delta, gradient_strength
