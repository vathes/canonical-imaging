import h5py
import caiman
import scipy
import numpy as np
from datetime import datetime
import pathlib


class CaImAn:
    """
    Parse the CaImAn output file
    Expecting the following objects:
    - 'dims':                 
    - 'dview':                
    - 'estimates':            
    - 'mmap_file':            
    - 'params':                
    - 'remove_very_bad_comps': 
    - 'skip_refinement':       
    - 'shifts_rig':             Motion correction object with x and y shifts per frame for rigid transformation
    - 'x_shifts_els':           Motion correction object with x shifts per frame per block for non rigid transformation
    - 'y_shifts_els':           Motion correction object with y shifts per frame per block for non rigid transformation
    - 'correlation_image':      
    CaImAn results doc: https://caiman.readthedocs.io/en/master/Getting_Started.html#result-variables-for-2p-batch-analysis
    """

    def __init__(self, caiman_dir):
        caiman_dir = pathlib.Path(caiman_dir)
        if not caiman_dir.exists():
            raise FileNotFoundError('CaImAn directory not found: {}'.format(caiman_dir))

        self.estimates_fp, self.mc_fp = None, None

        h5_files = list(caiman_dir.glob('*.hdf5'))
        for fp in h5_files:
            h5f = h5py.File(fp, 'r')
            if 'estimates' in h5f and 'A' in h5f['estimates']:
                self.estimates_fp = fp

        # ---- initialization for CaImAn's "estimates" ----
        if self.estimates_fp is None:
            raise FileNotFoundError('CaImAn estimates results (.h5py) file not found at {}'.format(caiman_dir))

        self.cnmf = caiman.source_extraction.cnmf.cnmf.load_CNMF(self.estimates_fp)
        self._masks = None

        # some metainfo
        self.creation_time = datetime.fromtimestamp(self.estimates_fp.stat().st_ctime)
        self.curation_time = datetime.fromtimestamp(self.estimates_fp.stat().st_ctime)

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

        comp_contours = caiman.utils.visualization.get_contours(self.cnmf.estimates.A, self.cnmf.dims)

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

    def extract_mc(self, caiman_filepath_manual):
        # ---- Open CaImAn output files ----
        h5f_manual = h5py.File(caiman_filepath_manual, 'r')

        if not any(s in h5f_manual for s in ('/mc/shifts_rig', '/mc/x_shifts_els', '/mc/y_shifts_els', '/cnmf/correlation_image')):
            raise FileNotFoundError('Rigid or non rigid shifts or correlation image not found in CaImAn file: {}'.format(caiman_filepath_manual))

        # ---- Declare attributes ----
        self._dims                  = self.analysis_h5['dims']
        self._dview                 = self.analysis_h5['dview']
        self._estimates             = self.analysis_h5['estimates']
        self._mmap_file             = self.analysis_h5['mmap_file']
        self._params                = self.analysis_h5['params']
        self._remove_very_bad_comps = self.analysis_h5['remove_very_bad_comps']
        self._skip_refinement       = self.analysis_h5['skip_refinement']

        self._correlation_image     = h5f_manual['correlation_image']

        if self._params['motion']['pw_rigid'][...]:
            self._x_shifts_els      = h5f_manual['x_shifts_els']
            self._y_shifts_els      = h5f_manual['y_shifts_els']
            return self._dims, self._dview, self._estimates, self._mmap_file, self._params, self._remove_very_bad_comps, self._skip_refinement, self._correlation_image, self._x_shifts_els, self._y_shifts_els
        else:
            self._shifts_rig        = h5f_manual['shifts_rig']
            return self._dims, self._dview, self._estimates, self._mmap_file, self._params, self._remove_very_bad_comps, self._skip_refinement, self._correlation_image, self._shifts_rig


def process_scanimage_tiff(scan_filenames, output_dir='./'):
    """
    Read scanimage tiffs - reshape into volumetric data based on scanning depths and channels
    Save new `tif` files for each channel - with shape (frame x height x width x depth)
    """
    from skimage.external.tifffile import imsave
    import scanreader
    from tqdm import tqdm

    # ============ CaImAn multi-channel multi-plane tiff file ==============
    for scan_filename in tqdm(scan_filenames):
        scan = scanreader.read_scan(scan_filename)
        cm_movie = caiman.load(scan_filename)

        # ---- Volumetric movie: (depth x height x width x channel x frame) ----
        # tiff pages are ordered as:
        # ch0-pln0-t0, ch1-pln0-t0, ch0-pln1-t0, ch1-pln1-t0, ..., ch0-pln1-t5, ch1-pln1-t5, ...

        vol_timeseries = np.full((scan.num_scanning_depths, scan.image_height, scan.image_width,
                                  scan.num_channels, scan.num_frames), 0).astype(scan.dtype)
        for pln_idx in range(scan.num_scanning_depths):
            for chn_idx in range(scan.num_channels):
                pln_chn_ind = np.arange(pln_idx * scan.num_channels + chn_idx, scan._num_pages,
                                        scan.num_scanning_depths * scan.num_channels)
                vol_timeseries[pln_idx, :, :, chn_idx, :] = cm_movie[pln_chn_ind, :, :].transpose(1, 2, 0)

        # save volumetric movie for individual channel
        output_dir = pathlib.Path(output_dir)
        fname = pathlib.Path(scan_filename).stem

        for chn_idx in range(scan.num_channels):
            chn_vol = vol_timeseries[:, :, :, 0, :].transpose(3, 1, 2, 0)  # (frame x height x width x depth)
            save_fp = output_dir / 'chn{}_{}.tif'.format(chn_idx, fname)
            imsave(save_fp.as_posix(), chn_vol)
