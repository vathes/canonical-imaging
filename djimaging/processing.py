import datajoint as dj
import scanreader
import numpy as np
import pathlib
from datetime import datetime
from uuid import UUID

from .imaging import schema, Scan, ScanInfo, Channel, PhysicalFile
from .utils import dict_to_hash

from djutils.templates import required, optional

from img_loaders import suite2p

# ===================================== Lookup =====================================


@schema
class ProcessingMethod(dj.Lookup):
    definition = """
    processing_method: char(8)
    """

    contents = zip(['suite2p', 'caiman'])


@schema
class ProcessingParamSet(dj.Lookup):
    definition = """
    paramset_idx:  smallint
    ---
    -> ProcessingMethod    
    paramset_desc: varchar(128)
    param_set_hash: uuid
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable parameters
    """

    @classmethod
    def insert_new_params(cls, processing_method: str, paramset_idx: int, paramset_desc: str, params: dict):
        param_dict = {'processing_method': processing_method,
                      'paramset_idx': paramset_idx,
                      'paramset_desc': paramset_desc,
                      'params': params,
                      'param_set_hash': UUID(dict_to_hash(params))}
        q_param = cls & {'param_set_hash': param_dict['param_set_hash']}

        if q_param:  # If the specified param-set already exists
            pname = q_param.fetch1('param_set_name')
            if pname == paramset_idx:  # If the existed set has the same name: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError('The specified param-set already exists - name: {}'.format(pname))
        else:
            cls.insert1(param_dict)


@schema
class CellCompartment(dj.Lookup):
    definition = """  # cell compartments that can be imaged
    cell_compartment         : char(16)
    """

    contents = zip(['axon', 'soma', 'bouton'])


@schema
class MaskType(dj.Lookup):
    definition = """ # possible classifications for a segmented mask
    mask_type        : varchar(16)
    """

    contents = zip(['soma', 'axon', 'dendrite', 'neuropil', 'artefact', 'unknown'])


# ===================================== Trigger a processing routine =====================================

@schema
class ProcessingTask(dj.Manual):
    definition = """
    -> Scan
    -> ProcessingParamSet
    """


@schema
class Processing(dj.Computed):
    definition = """
    -> ProcessingTask
    ---
    proc_completion_time     : datetime  # time of generation of this set of processed, segmented results
    proc_start_time=null     : datetime  # execution time of this processing task (not available if analysis triggering is NOT required)
    proc_curation_time=null  : datetime  # time of lastest curation (modification to the file) on this result set
    """

    class ProcessingOutputFile(dj.Part):
        definition = """
        -> master
        -> PhysicalFile
        """

    @staticmethod
    @optional
    def _get_caiman_dir(processing_task_key: dict) -> str:
        """
        Retrieve the CaImAn output directory for a given ProcessingTask
        :param processing_task_key: a dictionary of one ProcessingTask
        :return: a string for full path to the resulting CaImAn output directory
        """
        return None

    @staticmethod
    @optional
    def _get_suite2p_dir(processing_task_key: dict) -> str:
        """
        Retrieve the Suite2p output directory for a given ProcessingTask
        :param processing_task_key: a dictionary of one ProcessingTask
        :return: a string for full path to the resulting CaImAn output directory
        """
        return None

    # Run processing only on Scan with ScanInfo inserted
    @property
    def key_source(self):
        return ProcessingTask & ScanInfo

    def make(self, key):
        # ----
        # trigger suite2p or caiman here
        # ----

        method = (ProcessingParamSet * ProcessingTask & key).fetch1('processing_method')

        if method == 'suite2p':
            if (ScanInfo & key).fetch1('nrois') > 0:
                raise NotImplementedError(f'Suite2p ingestion error - Unable to handle ScanImage multi-ROI scanning mode yet')

            data_dir = pathlib.Path(Processing._get_suite2p_dir(key))
            if data_dir.exists():
                s2p_loader = suite2p.Suite2p(data_dir)
                key = {**key, 'proc_completion_time': s2p_loader.creation_time, 'proc_curation_time': s2p_loader.curation_time}
                self.insert1(key)
                # Insert file(s)
                root = pathlib.Path(PhysicalFile._get_root_data_dir())
                files = data_dir.glob('*')  # works for Suite2p, maybe something more file-specific for CaImAn
                files = [pathlib.Path(f).relative_to(root).as_posix() for f in files if f.is_file()]

                PhysicalFile.insert(zip(files), skip_duplicates=True)
                self.ProcessingOutputFile.insert([{**key, 'file_path': f} for f in files], ignore_extra_fields=True)
            else:
                start_time = datetime.now()
                # trigger Suite2p here
                # wait for completion, then insert with "completion_time", "start_time", no "curation_time"
                return
        else:
            raise NotImplementedError('Unknown method: {}'.format(method))


# ===================================== Motion Correction =====================================

@schema
class MotionCorrection(dj.Imported):
    definition = """ 
    -> Processing
    ---
    -> Channel.proj(mc_channel='channel')              # channel used for motion correction in this processing task
    """

    class RigidMotionCorrection(dj.Part):
        definition = """ 
        -> master
        -> ScanInfo.Field
        ---
        outlier_frames                  : longblob      # mask with true for frames with outlier shifts (already corrected)
        y_shifts                        : longblob      # (pixels) y motion correction shifts
        x_shifts                        : longblob      # (pixels) x motion correction shifts
        y_std                           : float         # (pixels) standard deviation of y shifts
        x_std                           : float         # (pixels) standard deviation of x shifts
        z_drift=null                    : longblob      # z-drift over frame of this Field (plane)
        """

    class NonRigidMotionCorrection(dj.Part):
        """ Piece-wise rigid motion correction - tile the FOV into multiple 2D blocks/patches"""
        definition = """ 
        -> master
        -> ScanInfo.Field
        ---
        outlier_frames                  : longblob      # mask with true for frames with outlier shifts (already corrected)
        block_height                    : int           # (px)
        block_width                     : int           # (px)
        block_count_y                   : int           # number of blocks tiled in the y direction
        block_count_x                   : int           # number of blocks tiled in the x direction
        z_drift=null                    : longblob      # z-drift over frame of this Field (plane)
        """

    class Block(dj.Part):
        definition = """  # FOV-tiled blocks used for non-rigid motion correction
        -> master.NonRigidMotionCorrection
        block_id                        : int
        ---
        block_y                         : longblob      # (y_start, y_end) in pixel of this block
        block_x                         : longblob      # (x_start, x_end) in pixel of this block
        y_shifts                        : longblob      # (pixels) y motion correction shifts for every frame
        x_shifts                        : longblob      # (pixels) x motion correction shifts for every frame
        y_std                           : float         # (pixels) standard deviation of y shifts
        x_std                           : float         # (pixels) standard deviation of x shifts
        """

    class Summary(dj.Part):
        definition = """ # summary images for each field and channel after corrections
        -> master
        -> ScanInfo.Field
        ---
        ref_image                    : longblob      # image used as alignment template
        average_image                : longblob      # mean of registered frames
        correlation_image=null       : longblob      # correlation map (computed during cell detection)
        max_proj_image=null          : longblob      # max of registered frames
        """

    def make(self, key):

        method = (ProcessingParamSet * ProcessingTask & key).fetch1('processing_method')

        if method == 'suite2p':
            data_dir = pathlib.Path(Processing._get_suite2p_dir(key))
            s2p_loader = suite2p.Suite2p(data_dir)

            field_keys = (ScanInfo.Field & key).fetch('KEY', order_by='field_z')

            align_chn = s2p_loader.planes[0].alignment_channel
            self.insert1({**key, 'mc_channel': align_chn})

            # ---- iterate through all s2p plane outputs ----
            for plane, s2p in s2p_loader.planes.items():
                mc_key = (ScanInfo.Field * ProcessingTask & key & field_keys[plane]).fetch1('KEY')

                # -- rigid motion correction --
                rigid_mc = {'y_shifts': s2p.ops['yoff'],
                            'x_shifts': s2p.ops['xoff'],
                            'y_std': np.nanstd(s2p.ops['yoff']),
                            'x_std': np.nanstd(s2p.ops['xoff']),
                            'outlier_frames': s2p.ops['badframes']}

                self.RigidMotionCorrection.insert1({**mc_key, **rigid_mc})

                # -- non-rigid motion correction --
                if s2p.ops['nonrigid']:
                    nonrigid_mc = {'block_height': s2p.ops['block_size'][0],
                                   'block_width': s2p.ops['block_size'][1],
                                   'block_count_y': s2p.ops['nblocks'][0],
                                   'block_count_x': s2p.ops['nblocks'][1],
                                   'outlier_frames': s2p.ops['badframes']}
                    nr_blocks = [{**mc_key, 'block_id': b_id,
                                  'block_y': b_y, 'block_x': b_x,
                                  'y_shifts': bshift_y, 'x_shifts': bshift_x,
                                  'y_std': np.nanstd(bshift_y), 'x_std': np.nanstd(bshift_x)}
                                 for b_id, (b_y, b_x, bshift_y, bshift_x)
                                 in enumerate(zip(s2p.ops['xblock'], s2p.ops['yblock'],
                                                  s2p.ops['yoff1'].T, s2p.ops['xoff1'].T))]
                    self.NonRigidMotionCorrection.insert1({**mc_key, **nonrigid_mc})
                    self.Block.insert(nr_blocks)

                # -- summary images --
                img_dict = {'ref_image': s2p.ref_image,
                            'average_image': s2p.mean_image,
                            'correlation_image': s2p.correlation_map,
                            'max_proj_image': s2p.max_proj_image}
                self.Summary.insert1({**mc_key, **img_dict})

        else:
            raise NotImplementedError('Unknown/unimplemented method: {}'.format(method))

# ===================================== Segmentation =====================================


@schema
class Segmentation(dj.Computed):
    definition = """ # Different mask segmentations.
    -> MotionCorrection    
    """

    class Mask(dj.Part):
        definition = """ # A mask produced by segmentation.
        -> master
        mask                : smallint
        ---
        -> Channel.proj(seg_channel='channel')   # channel used for the segmentation
        -> ScanInfo.Field                        # the field this ROI comes from
        mask_npix                : int           # number of pixels in ROIs
        mask_center_x            : int           # center x coordinate in pixels
        mask_center_y            : int           # center y coordinate in pixels
        mask_xpix                : longblob      # x coordinates in pixels
        mask_ypix                : longblob      # y coordinates in pixels        
        mask_weights             : longblob      # weights of the mask at the indices above in column major (Fortran) order
        """

    def make(self, key):
        method = (ProcessingParamSet * ProcessingTask & key).fetch1('processing_method')

        if method == 'suite2p':
            data_dir = pathlib.Path(Processing._get_suite2p_dir(key))
            s2p_loader = suite2p.Suite2p(data_dir)

            field_keys = (ScanInfo.Field & key).fetch('KEY', order_by='field_z')

            # ---- iterate through all s2p plane outputs ----
            masks, cells = [], []
            for plane, s2p in s2p_loader.planes.items():
                seg_key = (ScanInfo.Field * ProcessingTask & key & field_keys[plane]).fetch1('KEY')
                mask_count = len(masks)  # increment mask id from all "plane"
                for mask_idx, (is_cell, cell_prob, mask_stat) in enumerate(zip(s2p.iscell, s2p.cell_prob, s2p.stat)):
                    masks.append({**seg_key, 'mask': mask_idx + mask_count, 'seg_channel': s2p.segmentation_channel,
                                  'mask_npix': mask_stat['npix'],
                                  'mask_center_x':  mask_stat['med'][1],
                                  'mask_center_y':  mask_stat['med'][0],
                                  'mask_xpix':  mask_stat['xpix'],
                                  'mask_ypix':  mask_stat['ypix'],
                                  'mask_weights':  mask_stat['lam']})
                    if is_cell:
                        cells.append({**seg_key, 'mask_classification_method': 'suite2p_default_classifier',
                                      'mask': mask_idx + mask_count, 'mask_type': 'soma', 'confidence': cell_prob})

            self.insert1(key)
            self.Mask.insert(masks, ignore_extra_fields=True)

            if cells:
                MaskClassification.insert1({**key, 'mask_classification_method': 'suite2p_default_classifier'}, allow_direct_insert=True)
                MaskClassification.MaskType.insert(cells, ignore_extra_fields=True, allow_direct_insert=True)
        else:
            raise NotImplementedError('Unknown/unimplemented method: {}'.format(method))


@schema
class MaskClassificationMethod(dj.Lookup):
    definition = """
    mask_classification_method: varchar(32)
    """

    contents = zip(['suite2p_default_classifier'])


@schema
class MaskClassification(dj.Computed):
    definition = """
    -> Segmentation
    -> MaskClassificationMethod
    """

    class MaskType(dj.Part):
        definition = """
        -> master
        -> Segmentation.Mask
        ---
        -> MaskType
        confidence: float
        """


# ===================================== Activity Trace =====================================


@schema
class Fluorescence(dj.Computed):
    definition = """  # fluorescence traces before spike extraction or filtering
    -> Segmentation
    """

    class Trace(dj.Part):
        definition = """
        -> master
        -> Segmentation.Mask
        -> Channel.proj(fluo_channel='channel')  # the channel that this trace comes from         
        ---
        fluorescence                : longblob  # Raw fluorescence trace
        neuropil_fluorescence=null  : longblob  # Neuropil fluorescence trace
        """

    def make(self, key):
        method = (ProcessingParamSet * ProcessingTask & key).fetch1('processing_method')

        if method == 'suite2p':
            data_dir = pathlib.Path(Processing._get_suite2p_dir(key))
            s2p_loader = suite2p.Suite2p(data_dir)

            self.insert1(key)

            # ---- iterate through all s2p plane outputs ----
            fluo_traces, fluo_chn2_traces = [], []
            for s2p in s2p_loader.planes.values():
                mask_count = len(fluo_traces)  # increment mask id from all "plane"
                for mask_idx, (f, fneu) in enumerate(zip(s2p.F, s2p.Fneu)):
                    fluo_traces.append({**key, 'mask': mask_idx + mask_count,
                                        'fluo_channel': 0,
                                        'fluorescence': f, 'neuropil_fluorescence': fneu})
                if len(s2p.F_chan2):
                    mask_chn2_count = len(fluo_chn2_traces)  # increment mask id from all "plane"
                    for mask_idx, (f2, fneu2) in enumerate(zip(s2p.F_chan2, s2p.Fneu_chan2)):
                        fluo_chn2_traces.append({**key, 'mask': mask_idx + mask_chn2_count,
                                                 'fluo_channel': 1,
                                                 'fluorescence': f2, 'neuropil_fluorescence': fneu2})

            self.Trace.insert(fluo_traces + fluo_chn2_traces)

        else:
            raise NotImplementedError('Unknown/unimplemented method: {}'.format(method))


@schema
class ActivityExtractionMethod(dj.Lookup):
    definition = """
    extraction_method: varchar(16)
    """


@schema
class Activity(dj.Computed):
    definition = """  # deconvolved calcium acitivity from fluorescence trace
    -> Fluorescence
    -> ActivityExtractionMethod
    """

    class Trace(dj.Part):
        definition = """  # delta F/F
        -> master
        -> Fluorescence.Trace
        ---
        activity_trace                : longblob  # 
        """
