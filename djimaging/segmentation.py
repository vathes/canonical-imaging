import datajoint as dj

from . import scope, mesoscan, preprocessing
from .mesoscan import schema


@schema
class SegmentationMethod(dj.Lookup):
    definition = """ # methods for mask extraction for multi-photon scans
    segmentation_method         : tinyint
    ---
    method_name                 : varchar(16)
    method_desc                 : varchar(255)
    language                    : enum('matlab', 'python')  # implementation language
    """
    contents = [
        [1, 'manual', '', 'matlab'],
        [2, 'nmf', 'constrained non-negative matrix factorization from Pnevmatikakis et al. (2016)',
         'python'],
        [3, 'nmf-patches', 'same as nmf but initialized in small image patches', 'python'],
        [4, 'nmf-boutons', 'nmf for axonal terminals', 'python'],
        [5, '3d-conv', 'masks from the segmentation of the stack', 'python'],
        [6, 'nmf-new', 'same as method 3 (nmf-patches) but with some better tuned params', 'python']
    ]


@schema
class CellCompartment(dj.Lookup):
    definition = """  # cell compartments that can be imaged
    compartment         : char(16)
    ---
    """
    contents = [['axon'], ['soma'], ['bouton']]


@schema
class SegmentationTask(dj.Manual):
    definition = """ # defines the target of segmentation and the channel to use
    -> mesoscan.Scan
    -> scope.Field
    -> scope.Channel
    -> SegmentationMethod
   ---
    -> CellCompartment
    """


@schema
class Segmentation(dj.Computed):
    definition = """ # Different mask segmentations.
    -> MotionCorrection         # animal_id, session, scan_idx, version, field
    -> SegmentationTask         # animal_id, session, scan_idx, field, channel, segmentation_method
    ---
    segmentation_time=CURRENT_TIMESTAMP     : timestamp     # automatic
    """

    class Mask(dj.Part):
        definition = """ # mask produced by segmentation.
        -> master
        mask_id         : smallint
        ---
        pixels          : longblob      # indices into the image in column major (Fortran) order
        weights         : longblob      # weights of the mask at the indices above
        """


@schema
class Fluorescence(dj.Computed):
    definition = """  # fluorescence traces before spike extraction or filtering
    -> Segmentation
    """

    class Trace(dj.Part):
        definition = """
        -> Fluorescence
        -> Segmentation.Mask
        ---
        trace                   : longblob
        """