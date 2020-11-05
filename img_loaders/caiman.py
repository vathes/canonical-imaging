import h5py

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

    def __init__(self, caiman_filepath_automated, caiman_filepath_manual):
        # ---- Verify files exists ----
        if not caiman_filepath_automated.exists() or not caiman_filepath_manual.exists():
            raise FileNotFoundError('Both "*.hdf5" files are not found. Invalid CaImAn file: {}, {}'.format(caiman_filepath_automated,caiman_filepath_manual))
        else:
            # ---- Open CaImAn output files ----
            h5f_automated = h5py.File(caiman_filepath_automated, 'r')
            h5f_manual = h5py.File(caiman_filepath_manual, 'r')

            if not any(s in h5f_manual for s in ('/mc/shifts_rig', '/mc/x_shifts_els', '/mc/y_shifts_els', '/cnmf/correlation_image')):
                raise FileNotFoundError('Rigid or non rigid shifts or correlation image not found in CaImAn file: {}'.format(caiman_filepath_manual))

            # ---- Declare attributes ----
            self._dims                  = h5f_automated['dims']
            self._dview                 = h5f_automated['dview']
            self._estimates             = h5f_automated['estimates']
            self._mmap_file             = h5f_automated['mmap_file']
            self._params                = h5f_automated['params']
            self._remove_very_bad_comps = h5f_automated['remove_very_bad_comps']
            self._skip_refinement       = h5f_automated['skip_refinement']

            self._correlation_image     = h5f_manual['correlation_image']

            if self._params['motion']['pw_rigid'][...]:
                self._x_shifts_els      = h5f_manual['x_shifts_els']
                self._y_shifts_els      = h5f_manual['y_shifts_els']
                return self._dims, self._dview, self._estimates, self._mmap_file, self._params, self._remove_very_bad_comps, self._skip_refinement, self._correlation_image, self._x_shifts_els, self._y_shifts_els
            else:
                self._shifts_rig        = h5f_manual['shifts_rig']
                return self._dims, self._dview, self._estimates, self._mmap_file, self._params, self._remove_very_bad_comps, self._skip_refinement, self._correlation_image, self._shifts_rig