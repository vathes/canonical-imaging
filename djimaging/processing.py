import datajoint as dj
import scanreader
import numpy as np
import pathlib
from datetime import datetime
from uuid import UUID
import os

from .imaging import schema, Scan, ScanInfo, Channel, PhysicalFile
from .utils import dict_to_hash

from djutils.templates import required, optional

from img_loaders import suite2p_loader, caiman_loader

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
    ---
    task_mode='load': enum('load', 'trigger')  # 'load': load computed analysis results, 'trigger': trigger computation
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
        :return: a string for full path to the resulting Suite2p output directory
        """
        return None

    # Run processing only on Scan with ScanInfo inserted
    @property
    def key_source(self):
        return ProcessingTask & ScanInfo

    def make(self, key):
        method, task_mode = (ProcessingParamSet * ProcessingTask & key).fetch1('processing_method', 'task_mode')

        if task_mode == 'load':
            if method == 'suite2p':
                if (ScanInfo & key).fetch1('nrois') > 0:
                    raise NotImplementedError(f'Suite2p ingestion error - Unable to handle ScanImage multi-ROI scanning mode yet')

                data_dir = pathlib.Path(Processing._get_suite2p_dir(key))
                suite2p_loader_inst = suite2p_loader.Suite2p(data_dir)
                key = {**key, 'proc_completion_time': suite2p_loader_inst.creation_time, 'proc_curation_time': suite2p_loader_inst.curation_time}
                # Insert file(s)
                root = pathlib.Path(PhysicalFile._get_root_data_dir())
                output_files = data_dir.glob('*')
                output_files = [pathlib.Path(f).relative_to(root).as_posix() for f in output_files if f.is_file()]

            elif method == 'caiman':
                data_dir = pathlib.Path(Processing._get_caiman_dir(key))
                hdf5_fp = list(data_dir.glob('*.hdf5'))
                caiman_fp = hdf5_fp[0].as_posix()
                caiman_loader_inst = caiman_loader.CaImAn(caiman_fp)

                key = {**key, 'proc_completion_time': caiman_loader_inst.creation_time,
                              'proc_curation_time': caiman_loader_inst.curation_time}

                # Insert file(s)
                # root = pathlib.Path(PhysicalFile._get_root_data_dir())
                # output_files = [pathlib.Path(f).relative_to(root).as_posix() for f in output_files if f.is_file()]
                output_files=[caiman_fp]
            else:
                raise NotImplementedError('Unknown method: {}'.format(method))

            self.insert1(key)
            PhysicalFile.insert(zip(output_files), skip_duplicates=True)
            self.ProcessingOutputFile.insert([{**key, 'file_path': f} for f in output_files], ignore_extra_fields=True)

        elif task_mode == 'trigger':
            start_time = datetime.now()
            # trigger Suite2p or CaImAn here
            # wait for completion, then insert with "completion_time", "start_time", no "curation_time"
            return


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
        outlier_frames=null             : longblob      # mask with true for frames with outlier shifts (already corrected)
        y_shifts                        : longblob      # (pixels) y motion correction shifts
        x_shifts                        : longblob      # (pixels) x motion correction shifts
        y_std                           : float         # (pixels) standard deviation of y shifts across all frames
        x_std                           : float         # (pixels) standard deviation of x shifts across all frames
        z_drift=null                    : longblob      # z-drift over frame of this Field (plane)
        """

    class NonRigidMotionCorrection(dj.Part):
        """ Piece-wise rigid motion correction - tile the FOV into multiple 2D blocks/patches"""
        definition = """ 
        -> master
        -> ScanInfo.Field
        ---
        outlier_frames=null             : longblob      # mask with true for frames with outlier shifts (already corrected)
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
        y_std                           : float         # (pixels) standard deviation of y shifts across all frames
        x_std                           : float         # (pixels) standard deviation of x shifts across all frames
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
            suite2p_loader_inst = suite2p_loader.Suite2p(data_dir)

            field_keys = (ScanInfo.Field & key).fetch('KEY', order_by='field_z')

            align_chn = suite2p_loader_inst.planes[0].alignment_channel
            self.insert1({**key, 'mc_channel': align_chn})

            # ---- iterate through all s2p plane outputs ----
            for plane, s2p in suite2p_loader_inst.planes.items():
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

        elif method == 'caiman':
            data_dir = pathlib.Path(Processing._get_caiman_dir(key))
            hdf5_fp = list(data_dir.glob('*.hdf5'))
            caiman_fp = hdf5_fp[0].as_posix()
            caiman_loader_inst = caiman_loader.CaImAn(caiman_fp)
            caiman_results = caiman_loader_inst.extract_mc(caiman_fp)

            self.insert1({**key, 'mc_channel': caiman_results.alignment_channel})
            mc_key = (ScanInfo.Field * ProcessingTask & key).fetch1('KEY')

            # -- rigid motion correction --
            if not caiman_results.params['motion']['pw_rigid'][...]:
                rigid_mc = {'y_shifts': caiman_results.motion_correction['shifts_rig'][:,1],
                            'x_shifts': caiman_results.motion_correction['shifts_rig'][:,0],
                            'y_std': np.nanstd(caiman_results.motion_correction['shifts_rig'][:,1]),
                            'x_std': np.nanstd(caiman_results.motion_correction['shifts_rig'][:,0]),
                            'outlier_frames': None}

                self.RigidMotionCorrection.insert1({**mc_key, **rigid_mc})

            # -- non-rigid motion correction --
            else:
                nonrigid_mc = {'block_height': caiman_results.params['motion']['strides'][...][0]+caiman_results.params['motion']['overlaps'][...][0],
                                'block_width': caiman_results.params['motion']['strides'][...][1]+caiman_results.params['motion']['overlaps'][...][1],
                                'block_count_y': np.count_nonzero((caiman_results.motion_correction['coord_shifts_els'][:,0])==0),
                                'block_count_x': np.count_nonzero((caiman_results.motion_correction['coord_shifts_els'][0,:])==0),
                                'outlier_frames': None}
 
                nr_blocks = []
                for b_id in range(len(caiman_results.motion_correction['x_shifts_els'][0,:])):
                    nr_blocks.append({**mc_key, 'block_id': b_id,
                                    'block_y': caiman_results.motion_correction['coord_shifts_els'][b_id,2:4],
                                    'block_x': caiman_results.motion_correction['coord_shifts_els'][b_id,0:2],
                                    'y_shifts': caiman_results.motion_correction['y_shifts_els'][:,b_id], 
                                    'x_shifts': caiman_results.motion_correction['x_shifts_els'][:,b_id],
                                    'y_std': np.nanstd(caiman_results.motion_correction['y_shifts_els'][:,b_id]), 
                                    'x_std': np.nanstd(caiman_results.motion_correction['x_shifts_els'][:,b_id])
                                    })

                self.NonRigidMotionCorrection.insert1({**mc_key, **nonrigid_mc})
                self.Block.insert(nr_blocks)

            # -- summary images --
            img_dict = {'ref_image': caiman_results.motion_correction['reference_image'][...],
                        'average_image': caiman_results.motion_correction['average_image'][...],
                        'correlation_image': caiman_results.motion_correction['correlation_image'][...],
                        'max_proj_image': caiman_results.motion_correction['max_image'][...]}
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
        mask_npix                : int           # number of pixels in ROIs
        mask_center_x            : int           # center x coordinate in pixel
        mask_center_y            : int           # center y coordinate in pixel
        mask_center_z            : int           # center z coordinate in pixel
        mask_xpix                : longblob      # x coordinates in pixels
        mask_ypix                : longblob      # y coordinates in pixels      
        mask_zpix                : longblob      # z coordinates in pixels        
        mask_weights             : longblob      # weights of the mask at the indices above in column major (Fortran) order
        """

    def make(self, key):
        method = (ProcessingParamSet * ProcessingTask & key).fetch1('processing_method')

        if method == 'suite2p':
            data_dir = pathlib.Path(Processing._get_suite2p_dir(key))
            suite2p_loader_inst = suite2p_loader.Suite2p(data_dir)
            field_keys = (ScanInfo.Field & key).fetch('KEY', order_by='field_z')

            # ---- iterate through all s2p plane outputs ----
            masks, cells = [], []
            for plane, s2p in suite2p_loader_inst.planes.items():
                mask_count = len(masks)  # increment mask id from all "plane"
                for mask_idx, (is_cell, cell_prob, mask_stat) in enumerate(zip(s2p.iscell, s2p.cell_prob, s2p.stat)):
                    masks.append({**key, 'mask': mask_idx + mask_count, 'seg_channel': s2p.segmentation_channel,
                                  'mask_npix': mask_stat['npix'],
                                  'mask_center_x':  mask_stat['med'][1],
                                  'mask_center_y':  mask_stat['med'][0],
                                  'mask_center_z': mask_stat.get('iplane', plane),
                                  'mask_xpix':  mask_stat['xpix'],
                                  'mask_ypix':  mask_stat['ypix'],
                                  'mask_zpix': np.full(mask_stat['npix'], mask_stat.get('iplane', plane)),
                                  'mask_weights':  mask_stat['lam']})
                    if is_cell:
                        cells.append({**key, 'mask_classification_method': 'suite2p_default_classifier',
                                      'mask': mask_idx + mask_count, 'mask_type': 'soma', 'confidence': cell_prob})

            self.insert1(key)
            self.Mask.insert(masks, ignore_extra_fields=True)

            if cells:
                MaskClassification.insert1({**key, 'mask_classification_method': 'suite2p_default_classifier'}, allow_direct_insert=True)
                MaskClassification.MaskType.insert(cells, ignore_extra_fields=True, allow_direct_insert=True)
        
        elif method == 'caiman':
            data_dir = pathlib.Path(Processing._get_caiman_dir(key))
            hdf5_fp = list(data_dir.glob('*.hdf5'))
            caiman_fp = hdf5_fp[0].as_posix()
            caiman_loader_inst = caiman_loader.CaImAn(caiman_fp)

            # infer "segmentation_channel" - from params if available, else from caiman loader
            params = (ProcessingParamSet * ProcessingTask & key).fetch1('params')
            seg_channel = params.get('segmentation_channel', caiman_loader_inst.segmentation_channel)

            masks, cells = [], []
            for mask in caiman_loader_inst.masks:
                masks.append({**key, 'seg_channel': seg_channel,
                              'mask': mask['mask_id'],
                              'mask_npix': mask['mask_npix'],
                              'mask_center_x': mask['mask_center_x'],
                              'mask_center_y': mask['mask_center_y'],
                              'mask_center_z': mask['mask_center_z'],
                              'mask_xpix': mask['mask_xpix'],
                              'mask_ypix': mask['mask_ypix'],
                              'mask_zpix': mask['mask_zpix'],
                              'mask_weights': mask['mask_weights']})
                if mask['mask_id'] in caiman_loader_inst.cnmf.estimates.idx_components:
                    cells.append({**key, 'mask_classification_method': 'caiman_default',
                                  'mask': mask['mask_id'], 'mask_type': 'soma'})

            self.insert1(key)
            self.Mask.insert(masks, ignore_extra_fields=True)

            if cells:
                MaskClassification.insert1({**key, 'mask_classification_method': 'caiman_default'}, allow_direct_insert=True)
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
        confidence=null: float
        """

    def make(self, key):
        pass


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
        fluorescence                : longblob  # fluorescence trace associated with this mask
        neuropil_fluorescence=null  : longblob  # Neuropil fluorescence trace
        """

    def make(self, key):
        method = (ProcessingParamSet * ProcessingTask & key).fetch1('processing_method')

        if method == 'suite2p':
            data_dir = pathlib.Path(Processing._get_suite2p_dir(key))
            suite2p_loader_inst = suite2p_loader.Suite2p(data_dir)

            # ---- iterate through all s2p plane outputs ----
            fluo_traces, fluo_chn2_traces = [], []
            for s2p in suite2p_loader_inst.planes.values():
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

            self.insert1(key)
            self.Trace.insert(fluo_traces + fluo_chn2_traces)

        elif method == 'caiman':
            data_dir = pathlib.Path(Processing._get_caiman_dir(key))
            hdf5_fp = list(data_dir.glob('*.hdf5'))
            caiman_fp = hdf5_fp[0].as_posix()
            caiman_loader_inst = caiman_loader.CaImAn(caiman_fp)

            # infer "segmentation_channel" - from params if available, else from caiman loader
            params = (ProcessingParamSet * ProcessingTask & key).fetch1('params')
            seg_channel = params.get('segmentation_channel', caiman_loader_inst.segmentation_channel)

            fluo_traces = []
            for mask in caiman_loader_inst.masks:
                fluo_traces.append({**key, 'mask': mask['mask_id'], 'fluo_channel': seg_channel,
                                    'fluorescence': mask['inferred_trace']})

            self.insert1(key)
            self.Trace.insert(fluo_traces)
        else:
            raise NotImplementedError('Unknown/unimplemented method: {}'.format(method))


@schema
class ActivityExtractionMethod(dj.Lookup):
    definition = """
    extraction_method: varchar(32)
    """

    contents = zip(['suite2p_deconvolution', 'caiman_deconvolution', 'caiman_dff'])


@schema
class Activity(dj.Computed):
    definition = """  # inferred neural activity from fluorescence trace - e.g. dff, spikes
    -> Fluorescence
    -> ActivityExtractionMethod
    """

    class Trace(dj.Part):
        definition = """  #
        -> master
        -> Fluorescence.Trace
        ---
        activity_trace: longblob  # 
        """

    def make(self, key):

        method = (ProcessingParamSet * ProcessingTask & key).fetch1('processing_method')

        if method == 'suite2p':
            if key['extraction_method'] == 'suite2p_deconvolution':
                data_dir = pathlib.Path(Processing._get_suite2p_dir(key))
                suite2p_loader_inst = suite2p_loader.Suite2p(data_dir)

                self.insert1(key)

                # ---- iterate through all s2p plane outputs ----
                spikes = []
                for s2p in suite2p_loader_inst.planes.values():
                    mask_count = len(spikes)  # increment mask id from all "plane"
                    for mask_idx, spks in enumerate(s2p.spks):
                        spikes.append({**key, 'mask': mask_idx + mask_count,
                                       'fluo_channel': 0,
                                       'activity_trace': spks})
                self.Trace.insert(spikes)
                
        elif method == 'caiman':
            if key['extraction_method'] in ('caiman_deconvolution', 'caiman_dff'):
                attr_mapper = {'caiman_deconvolution': 'spikes', 'caiman_dff': 'dff'}

                data_dir = pathlib.Path(Processing._get_caiman_dir(key))
                hdf5_fp = list(data_dir.glob('*.hdf5'))
                caiman_fp = hdf5_fp[0].as_posix()
                caiman_loader_inst = caiman_loader.CaImAn(caiman_fp)

                # infer "segmentation_channel" - from params if available, else from caiman loader
                params = (ProcessingParamSet * ProcessingTask & key).fetch1('params')
                seg_channel = params.get('segmentation_channel', caiman_loader_inst.segmentation_channel)

                activities = []
                for mask in caiman_loader_inst.masks:
                    activities.append({**key, 'mask': mask['mask_id'],
                                       'fluo_channel': seg_channel,
                                       'activity_trace': mask[attr_mapper[key['extraction_method']]]})
                self.insert1(key)
                self.Trace.insert(activities)

        else:
            raise NotImplementedError('Unknown/unimplemented method: {}'.format(method))
