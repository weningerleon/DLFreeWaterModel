import numpy as np
from difreewater import learning


def subtract_water(dwi, brainmask, water_percentage, bvals, csf_voxel, threshold_waterpercentage = 0.8):

    s = dwi.shape

    dwi_1d = dwi.reshape((s[0] * s[1] * s[2], s[3]), order='F')
    brainmask_1d = brainmask.ravel(order='F')

    S_orig = dwi_1d[brainmask_1d]
    b0 = np.expand_dims(np.mean(S_orig[:, bvals == 0], axis=1), axis=-1)

    S_orig = np.divide(S_orig, b0)

    #%% Calculate water contribution to diffusion signal

    water_percentage_1d = water_percentage.reshape((s[0] * s[1] * s[2]), order='F')
    f = water_percentage_1d[brainmask_1d]
    b0_csf = np.expand_dims(np.mean(csf_voxel[:,bvals==0]), axis=-1)
    S_csf = np.divide(csf_voxel, b0_csf)

    S_watercompartment = np.multiply(np.expand_dims(f,axis=-1), S_csf)

    #%%

    S_corr = np.divide(S_orig - S_watercompartment, np.expand_dims(1-f, axis=1))

    S_corr = np.multiply(S_corr, b0)

    S_corr[S_corr<0] = 0
    S_corr[np.isnan(S_corr)] = 0
    S_corr[np.isinf(S_corr)] = 0

    corrected = np.zeros((s[0] * s[1] * s[2], s[3]))
    corrected[brainmask_1d] = S_corr

    corrected[water_percentage_1d>threshold_waterpercentage] = dwi_1d[water_percentage_1d>threshold_waterpercentage]

    corrected = corrected.reshape((s[0],s[1],s[2], s[3]), order='F')

    return corrected


def dehydrate(dwi, brainmask, synDataLoader, net, iterative_correction=False):
    dwi = dwi.copy()
    dwi_original = dwi.copy()

    relData = synDataLoader.data2model(dwi[brainmask])

    # %% Predict & Reconstruct with Neural Network
    pred = learning.predict(relData, net)
    r = np.zeros(brainmask.shape)
    r[brainmask] = pred.squeeze()

    reconst_brain = subtract_water(dwi, brainmask, r, synDataLoader.gtab.bvals,
                                   synDataLoader.get_csf_voxel())

    if iterative_correction == True:
        for i in range(20):

            relData = synDataLoader.data2model(dwi[brainmask])

            # %% Predict & Reconstruct with Neural Network
            pred_old = pred.copy()
            pred = learning.predict(relData, net)

            pred = pred_old + (1-pred_old)*pred
            r[brainmask] = pred
            reconst_brain = subtract_water(dwi_original, brainmask, r, synDataLoader.gtab.bvals,
                                           synDataLoader.tenmodel, synDataLoader.get_csf_voxel())

            dwi = reconst_brain

    return r, reconst_brain