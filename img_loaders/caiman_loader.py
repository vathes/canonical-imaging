import h5py
import caiman as cm
import scipy
import numpy as np
from datetime import datetime
import os


class CaImAn:
    """
    Parse the CaImAn output file
    Expecting the following objects:
    - 'dims':                 
    - 'dview':                
    - 'estimates':            
    - 'mmap_file':            
    - 'params':                 Input parameters
    - 'remove_very_bad_comps': 
    - 'skip_refinement':       
    - 'motion_correction':      Motion correction shifts and summary images
    CaImAn results doc: https://caiman.readthedocs.io/en/master/Getting_Started.html#result-variables-for-2p-batch-analysis
    """

    def __init__(self, caiman_fp):
        # ---- Verify dataset exists ----
        if caiman_fp is None:
            raise FileNotFoundError('CaImAn results (.hdf5) file not found at {}'.format(caiman_fp))

        self.h5f = h5py.File(caiman_fp, 'r')

        if not any(s in self.h5f for s in ('/motion_correction/reference_image',
                                           '/motion_correction/correlation_image',
                                           '/motion_correction/average_image',
                                           '/motion_correction/max_image',
                                           '/estimates/A')):
            raise NameError('CaImAn results (.hdf5) file found at {} does not contain all datasets'.format(caiman_fp))

        # ---- Initialize CaImAn's results ----
        self.cnmf = cm.source_extraction.cnmf.cnmf.load_CNMF(caiman_fp)
        self._masks = None

        # ---- Metainfo ----
        self.creation_time = datetime.fromtimestamp(os.stat(caiman_fp).st_ctime)
        self.curation_time = datetime.fromtimestamp(os.stat(caiman_fp).st_ctime)

    @property
    def masks(self):
        if self._masks is None:
            self._masks = self.extract_masks()
        return self._masks

    @property
    def alignment_channel(self):
        return 0  # hard-code to channel index 0

    @property
    def segmentation_channel(self):
        return 0  # hard-code to channel index 0

    def extract_masks(self):
        if self.cnmf.params.motion['is3D']:
            raise NotImplemented('CaImAn mask extraction for volumetric data not yet implemented')

        comp_contours = cm.utils.visualization.get_contours(self.cnmf.estimates.A, self.cnmf.dims)

        masks = []
        for comp_idx, comp_contour in enumerate(comp_contours):
            ind, _, weights = scipy.sparse.find(self.cnmf.estimates.A[:, comp_idx])
            xpix, ypix = np.unravel_index(ind, self.cnmf.dims, order='F')
            center_x, center_y = comp_contour['CoM'].astype(int)
            masks.append({'mask_id': comp_contour['neuron_id'], 'mask_plane': 0,
                          'mask_npix': len(weights), 'mask_weights': weights,
                          'mask_center_x': center_x, 'mask_center_y': center_y,
                          'mask_xpix': xpix, 'mask_ypix': ypix,
                          'inferred_trace': self.cnmf.estimates.C[comp_idx, :],
                          'dff': self.cnmf.estimates.F_dff[comp_idx, :],
                          'spikes': self.cnmf.estimates.S[comp_idx, :]})
        return masks

    def save_mc(mc,caiman_fp):
        """
        DataJoint Imaging Element - CaImAn Integration
        Run these commands after the CaImAn analysis has completed.
        This will save the relevant motion correction data into the '*.hdf5' file.
        Please do not clear variables from memory prior to running these commands.
        The motion correction (mc) object will be read from memory.
        
        'mc' :                CaImAn motion correction object
        'caiman_fp' :         CaImAn output (*.hdf5) file path

        'shifts_rig' :        Rigid transformation x and y shifts per frame
        'x_shifts_els' :      Non rigid transformation x shifts per frame per block
        'y_shifts_els' :      Non rigid transformation y shifts per frame per block
        """

        # Load motion corrected mmap image
        mc_image = cm.load(mc.mmap_file)

        # Compute motion corrected summary images
        average_image = np.mean(mc_image, axis=0)
        max_image = np.max(mc_image, axis=0)

        # Compute motion corrected correlation image
        correlation_image = cm.local_correlations(mc_image.transpose(1,2,0))
        correlation_image[np.isnan(correlation_image)] = 0

        # Compute mc.coord_shifts_els
        xy_grid = []
        for _, _, x, y, _ in cm.motion_correction.sliding_window(mc_image[0,:,:], mc.overlaps, mc.strides):
            xy_grid.append([x, x + mc.overlaps[0] + mc.strides[0], y, y + mc.overlaps[1] + mc.strides[1]])

        # Open hdf5 file and create 'motion_correction' group
        h5f = h5py.File(caiman_fp,'r+')
        h5g = h5f.require_group("motion_correction")

        # Write motion correction shifts and motion corrected summary images to hdf5 file
        if mc.pw_rigid:
            h5g.require_dataset("x_shifts_els",shape=np.shape(mc.x_shifts_els),data=mc.x_shifts_els,dtype=mc.x_shifts_els[0][0].dtype)
            h5g.require_dataset("y_shifts_els",shape=np.shape(mc.y_shifts_els),data=mc.y_shifts_els,dtype=mc.y_shifts_els[0][0].dtype)
            h5g.require_dataset("coord_shifts_els",shape=np.shape(xy_grid),data=xy_grid,dtype=type(xy_grid[0][0]))
            h5g.require_dataset("reference_image",shape=np.shape(mc.total_template_els),data=mc.total_template_els,dtype=mc.total_template_els.dtype)
        else:
            h5g.require_dataset("shifts_rig",shape=np.shape(mc.shifts_rig),data=mc.shifts_rig,dtype=mc.shifts_rig[0].dtype)
            h5g.require_dataset("coord_shifts_rig",shape=np.shape(xy_grid),data=xy_grid,dtype=type(xy_grid[0][0]))
            h5g.require_dataset("reference_image",shape=np.shape(mc.total_template_rig),data=mc.total_template_rig,dtype=mc.total_template_rig.dtype)

        h5g.require_dataset("correlation_image",shape=np.shape(correlation_image),data=correlation_image,dtype=correlation_image.dtype)
        h5g.require_dataset("average_image",shape=np.shape(average_image),data=average_image,dtype=average_image.dtype)
        h5g.require_dataset("max_image",shape=np.shape(max_image),data=max_image,dtype=max_image.dtype)

        # Close hdf5 file
        h5f.close()

    def extract_mc(self, caiman_fp):
        # ---- Open CaImAn output file ----
        self.h5f = h5py.File(caiman_fp, 'r')

        # ---- Declare attributes ----
        self.dims                  = self.h5f['dims']
        self.dview                 = self.h5f['dview']
        self.estimates             = self.h5f['estimates']
        self.mmap_file             = self.h5f['mmap_file']
        self.params                = self.h5f['params']
        self.remove_very_bad_comps = self.h5f['remove_very_bad_comps']
        self.skip_refinement       = self.h5f['skip_refinement']
        self.motion_correction     = self.h5f['motion_correction']

        return self