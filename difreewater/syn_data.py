import numpy as np
import dipy.reconst.dti as dti
import random
from dipy.sims.phantom import add_noise

import warnings
from tqdm import tqdm

from dipy.reconst.csdeconv import fa_superior
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.core.gradients import gradient_table
from dipy.reconst.shm import sf_to_sh, sh_to_sf
from dipy.core.sphere import Sphere

from dipy.core.geometry import vec2vec_rotmat

def get_rotation_matrix(degrees):
    alpha, beta, gamma = degrees

    R_z = np.zeros((3,3))
    R_z[0,0] = np.cos(alpha*2*np.pi/360)
    R_z[1,1] = np.cos(alpha*2*np.pi/360)
    R_z[0,1] = -np.sin(alpha*2*np.pi/360)
    R_z[1,0] = np.sin(alpha*2*np.pi/360)
    R_z[2,2] = 1

    R_x = np.zeros((3,3))
    R_x[0,0] = 1
    R_x[1,1] = np.cos(gamma*2*np.pi/360)
    R_x[1,2] = -np.sin(gamma*2*np.pi/360)
    R_x[2,1] = np.sin(gamma*2*np.pi/360)
    R_x[2,2] = np.cos(gamma*2*np.pi/360)

    R_y = np.zeros((3,3))
    R_y[0,0] = np.cos(beta*2*np.pi/360)
    R_y[1,1] = 1
    R_y[0,2] = np.sin(beta*2*np.pi/360)
    R_y[2,0] = -np.sin(beta*2*np.pi/360)
    R_y[2,2] = np.cos(beta*2*np.pi/360)

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R


def rotate_edw(degrees, edw, bvecs, bvals, sh_order=8):

    R = get_rotation_matrix(degrees)

    edw_rotated = np.zeros(edw.shape)

    for bval in np.unique(bvals):
        if bval<50:
            edw_rotated[bvals==bval]=edw[bvals==bval]
        else:
            bvecs_ss = bvecs[bvals==bval]
            bvecs_rotated = np.dot(bvecs_ss, R)

            sphere = Sphere(xyz=bvecs_ss)
            sphere_rotated = Sphere(xyz=bvecs_rotated)

            sh_dipy = sf_to_sh(edw[bvals==bval], sphere, sh_order)
            edw_rotated[bvals==bval] = sh_to_sf(sh_dipy, sphere_rotated, sh_order)

    return edw_rotated


class SynDiffData():
    def __init__(self, gtab, data, mask_fast=None, csf_mask=None, gm_mask=None, wm_mask=None):
        self.gtab = gtab
        self.data = data
        if mask_fast != None:
            self.csf_mask = (mask_fast == 1)
            self.gm_mask = (mask_fast == 2)
            self.wm_mask = (mask_fast == 3)
        elif not ((wm_mask is None) or (gm_mask is None) or (wm_mask is None)):
            self.csf_mask = csf_mask
            self.gm_mask = gm_mask
            self.wm_mask = wm_mask
        else:
            raise AttributeError("Either mask_fast or the indiviual masks must be populated")

        self.datashape = None
        self.tenmodel = TensorModel(gtab)

        self.shells = []

        s = np.unique(self.gtab.bvals)
        for b in s[s != 0]:
            select_shell = np.logical_or(self.gtab.bvals == 0, self.gtab.bvals == b)
            bvals = self.gtab.bvals[select_shell]
            bvecs = self.gtab.bvecs[select_shell,:]

            ss_gtab = gradient_table(bvals, bvecs)
            self.shells.append({"select_shell": select_shell, "gtab": ss_gtab, "tenmodel": TensorModel(ss_gtab)})

    def response2sf(self, response):
        # transforms response (b0, 3 eigenvalues) in sphere/spheres (sampled at bvecs)
        voxels = np.zeros((response.shape[0],self.gtab.bvals.size))
        for i, s in enumerate(self.shells):
            ten = np.zeros((response.shape[0], 3, 3))
            ten[:,0, 0] = response[:,(i*3)+1]
            ten[:,1, 1] = response[:,(i*3)+2]
            ten[:,2, 2] = response[:,(i*3)+3]
            evals, evecs = dti.decompose_tensor(ten)
            voxels_ss = dti.tensor_prediction(np.concatenate((evals, np.reshape(evecs, (response.shape[0],9))), axis=1),
                                           s["gtab"], response[:,0])

            voxels[:,s["select_shell"]] = voxels_ss

        return voxels

    def _wm_tensors_sample(self, fa_thr=0.7,
                           fa_callable=fa_superior):
        # This function is similar to the dipy auto_response function, but returns all samples, not a mean value

        roi = self.data[self.wm_mask]

        r = np.zeros(roi.shape[0])
        for s in self.shells:
            roi_s  = roi[:,s["select_shell"]]
            tenfit_s = s["tenmodel"].fit(roi_s)
            FA = fractional_anisotropy(tenfit_s.evals)
            FA[np.isnan(FA)] = 0
            indices = np.where(fa_callable(FA, fa_thr))
            r[indices] = 1

        indices = np.where(r!=0) # if FA>fa_thr for at least one shell
        if indices[0].size == 0:
            msg = "No voxel with a FA higher than " + str(fa_thr) + " were found."
            msg += " Try a larger roi or a lower threshold."
            warnings.warn(msg, UserWarning)

        S0s = roi[indices][:, np.nonzero(self.gtab.b0s_mask)[0]]
        S0s = np.mean(S0s, axis=1)
        evals_ms = np.zeros((S0s.size, self.shells.__len__()*3))
        for i, s in enumerate(self.shells):
            roi_s  = roi[:,s["select_shell"]]
            tenfit_s = s["tenmodel"].fit(roi_s)

            lambdas = tenfit_s.evals[indices][:, :2]
            evals = np.stack((lambdas[:,0], lambdas[:,1], lambdas[:,1]), axis=1)
            evals_ms[:,i*3:(i+1)*3] = evals

        S0s = np.expand_dims(np.asarray(S0s), axis=1)
        response = np.append(S0s, evals_ms, axis=1)

        return response

    def _csf_tensor_mean(self):
        roi = self.data[self.csf_mask]

        S0 = np.mean(roi[:, np.nonzero(self.gtab.b0s_mask)[0]])
        evals_ms = []
        for s in self.shells:
            roi_s = roi[:, s["select_shell"]]
            tenfit_s = s["tenmodel"].fit(roi_s)

            lambda_s = np.mean(tenfit_s.evals)
            evals = np.array([lambda_s, lambda_s, lambda_s])
            evals_ms.append(evals)

        evals_ms = np.asarray(evals_ms)
        evals_ms = np.reshape(evals_ms, (self.shells.__len__()*3,))
        response = np.append(np.expand_dims(S0, axis=1), evals_ms)

        return np.expand_dims(response, axis=0)

    def _gm_tensor_mean(self):
        roi = self.data[self.gm_mask]

        S0 = np.mean(roi[:, np.nonzero(self.gtab.b0s_mask)[0]])
        evals_ms = []
        for s in self.shells:
            roi_s = roi[:, s["select_shell"]]
            tenfit_s = s["tenmodel"].fit(roi_s)

            lambda_s = np.mean(tenfit_s.evals)
            evals = np.array([lambda_s, lambda_s, lambda_s])
            evals_ms.append(evals)

        evals_ms = np.asarray(evals_ms)
        evals_ms = np.reshape(evals_ms, (self.shells.__len__()*3,))
        response = np.append(np.expand_dims(S0, axis=1), evals_ms)

        return np.expand_dims(response, axis=0)

    def data2model(self, voxels):
        S0 = np.mean(voxels[:, self.gtab.bvals == 0], axis=1)
        edw = voxels / np.expand_dims(S0, axis=1)
        edw = edw[:,self.gtab.bvals!=0]
        edw[np.isnan(edw)] = 1
        edw[np.isinf(edw)] = 1
        return edw

    def get_csf_voxel(self):
        return self.csf_voxel

    def create_waterfraction_data(self, sample_size=1000):
        wm_data =  self.response2sf(self._wm_tensors_sample())
        self.csf_response = self._csf_tensor_mean()
        csf_data = self.response2sf(self.csf_response)
        gm_data = self.response2sf(self._gm_tensor_mean())

        self.csf_voxel = csf_data

        len_csf = csf_data.shape[0]
        len_wm = wm_data.shape[0]
        len_gm = gm_data.shape[0]

        bvecs = self.gtab.bvecs
        bvals = self.gtab.bvals

        x = []
        y = []
        random.seed(1)
        for i in tqdm(range(sample_size)):
            f_csf = random.uniform(0, 1)
            f_gm = random.uniform(0, 1)
            f_wm = random.uniform(0, 1)

            # Normalize all percentages
            f_total = f_csf + f_gm + f_wm
            f_wm = f_wm / f_total
            f_gm = f_gm / f_total
            f_csf = f_csf / f_total

            csf_voxel = csf_data[random.randrange(len_csf),:]
            gm_voxel = gm_data[random.randrange(len_gm),:]

            wm_voxel1 = wm_data[random.randrange(len_wm),:]
            wm_voxel2 = wm_data[random.randrange(len_wm),:]
            wm_voxel3 = wm_data[random.randrange(len_wm),:]

            degrees = (np.random.rand(3) - 0.5) * 360
            wm_voxel1 = rotate_edw(degrees, wm_voxel1, bvecs, bvals)
            degrees = (np.random.rand(3) - 0.5) * 360
            wm_voxel2 = rotate_edw(degrees, wm_voxel2, bvecs, bvals)
            degrees = (np.random.rand(3) - 0.5) * 360
            wm_voxel3 = rotate_edw(degrees, wm_voxel3, bvecs, bvals)

            f_wm_voxels = np.random.rand(3)
            f_wm_voxels = (f_wm_voxels / np.sum(f_wm_voxels)) * f_wm

            syn_voxel = f_csf * csf_voxel + f_gm * gm_voxel + \
                        f_wm_voxels[0] * wm_voxel1 + f_wm_voxels[1] * wm_voxel2 + f_wm_voxels[2] * wm_voxel3

            syn_voxel = add_noise(syn_voxel, snr=25, noise_type='rician')

            x.append(syn_voxel)
            y.append(f_csf)

        x = np.asarray(x)
        x = self.data2model(x)
        y = np.asarray(y)

        self.datashape = x.shape[1]

        return x, y


