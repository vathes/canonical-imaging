import datajoint as dj
import scanreader
import numpy as np

from . import imaging, parameter
from .imaging import schema
from djutils.templates import required, optional

# ===================================== Lookup =====================================


@schema
class ProcessingMethod(dj.Lookup):
    definition = """
    processing_method: varchar(36)
    ---
    method_desc: varchar(255)
    """

    class CaImAnParamSet(dj.Part):
        definition = """
        -> master
        ---
        -> parameter.CaImAnParamSet
        """


@schema
class CellCompartment(dj.Lookup):
    definition = """  # cell compartments that can be imaged
    cell_compartment         : char(16)
    """
    contents = [['axon'], ['soma'], ['bouton']]


@schema
class RoiType(dj.Lookup):
    definition = """ # possible classifications for a segmented mask
    roi_type        : varchar(16)
    """
    contents = [
        ['soma'],
        ['axon'],
        ['dendrite'],
        ['neuropil'],
        ['artifact'],
        ['unknown']
    ]


# ===================================== Trigger a processing routine =====================================

@schema
class ProcessingTask(dj.Manual):
    definition = """
    -> imaging.Scan
    processing_instance: uuid
    ---
    -> ProcessingMethod
    process_mode: enum('trigger', 'import')
    """


@schema
class Processing(dj.Computed):
    definition = """
    -> ProcessingTask
    ---
    processing_time: datetime  # time of generation of this set of processed, segmented results
    """

    @staticmethod
    @optional
    def _get_caiman_dir():
        return None

    @staticmethod
    @optional
    def _get_suite2p_dir():
        return None


# ===================================== Motion Correction =====================================

@schema
class MotionCorrection(dj.Imported):
    definition = """ 
    -> ProcessingTask
    ---
    motion_template                 : longblob      # image used as alignment template
    y_shifts                        : longblob      # (pixels) y motion correction shifts
    x_shifts                        : longblob      # (pixels) x motion correction shifts
    y_std                           : float         # (pixels) standard deviation of y shifts
    x_std                           : float         # (pixels) standard deviation of x shifts
    outlier_frames                  : longblob      # mask with true for frames with outlier shifts (already corrected)
    align_time=CURRENT_TIMESTAMP    : timestamp     # automatic
    """


@schema
class MotionCorrectedImages(dj.Imported):
    definition = """ # summary images for each field and channel after corrections
    -> MotionCorrection
    -> imaging.ScanInfo.Field
    -> imaging.Channel
    """

    class Average(dj.Part):
        definition = """ # mean of each pixel across time
        -> master
        ---
        average_image           : longblob
        """

    class Correlation(dj.Part):
        definition = """ # average temporal correlation between each pixel and its eight neighbors
        -> master
        ---
        correlation_image           : longblob
        """


# ===================================== Segmentation =====================================

@schema
class Segmentation(dj.Computed):
    definition = """ # Different mask segmentations.
    -> processing.MotionCorrection        
    ---
    -> imaging.Channel.proj(seg_channel='channel')  # channel used for the segmentation
    """

    class Roi(dj.Part):
        definition = """ # Region-of-interest (mask) produced by segmentation.
        -> master
        roi                 : smallint
        ---
        -> imaging.ScanInfo.Field           # the field this ROI comes from
        npix = NULL         : int           # number of pixels in ROIs
        center_x            : int           # center x coordinate in pixels
        center_y            : int           # center y coordinate in pixels
        xpix                : longblob      # x coordinates in pixels
        ypix                : longblob      # y coordinates in pixels        
        weights             : longblob      # weights of the mask at the indices above in column major (Fortran) order
        """


@schema
class RoiClassification(dj.Computed):
    definition = """
    -> Segmentation
    """

    class RoiType(dj.Part):
        definition = """
        -> master
        -> Segmentation.Roi
        ---
        -> RoiType        
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
        -> Segmentation.Roi
        -> imaging.Channel.proj(roi_channel='channel')  # the channel that this trace comes from 
        ---
        fluo                : longblob  # Raw fluorescence trace
        neuropil_fluo       : longblob  # Neuropil fluorescence trace
        """


@schema
class DeconvolvedCalciumActivity(dj.Computed):
    definition = """  # fluorescence traces before spike extraction or filtering
    -> Fluorescence
    """

    class DFF(dj.Part):
        definition = """
        -> master
        -> Fluorescence.Trace
        ---
        df_f                : longblob  # delta F/F - deconvolved calcium acitivity 
        """


@schema
class SpikeActivity(dj.Computed):
    definition = """  # Inferred spiking activity
    -> DeconvolvedCalciumActivity
    """

    class DFF(dj.Part):
        definition = """
        -> master
        -> DeconvolvedCalciumActivity.DFF
        ---
        spike                : longblob  # spike train
        """
