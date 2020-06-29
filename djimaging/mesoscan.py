import datajoint as dj

from . import scope
from .scope import schema
from djutils.templates import required


# ===================================== Scan =====================================


@schema
class Scan(dj.Manual):

    _Session = ...
    _Equipment = ...    # scanner info, e.g. scope, lens, etc.

    definition = """    #
    -> self._Session    # API hook point
    ---
    -> self._Equipment  # API hook point
    scan_notes='' : varchar(4095)         # free-notes
    """


@schema
class ScanLocation(dj.Manual):

    _Location = ...

    definition = """
    -> Scan       
    -> self._Location            # API hook point
    """


@schema
class ScanInfo(dj.Imported):
    definition = """ # general data about mesoscope scans
    -> Scan
    -> Version                                  # meso version
    ---
    nfields                 : tinyint           # number of fields
    nchannels               : tinyint           # number of channels
    nframes                 : int               # number of recorded frames
    nframes_requested       : int               # number of requested frames (from header)
    x                       : float             # (um) ScanImage's 0 point in the motor coordinate system
    y                       : float             # (um) ScanImage's 0 point in the motor coordinate system
    fps                     : float             # (Hz) frames per second
    bidirectional           : boolean           # true = bidirectional scanning
    usecs_per_line          : float             # microseconds per scan line
    fill_fraction           : float             # raster scan temporal fill fraction (see scanimage)
    nrois                   : tinyint           # number of ROIs (see scanimage)
    """

    class Field(dj.Part):
        definition = """ # field-specific scan information
        -> master
        -> scope.Field
        ---
        px_height           : smallint      # height in pixels
        px_width            : smallint      # width in pixels
        um_height           : float         # height in microns
        um_width            : float         # width in microns
        x                   : float         # (um) center of field in the motor coordinate system
        y                   : float         # (um) center of field in the motor coordinate system
        z                   : float         # (um) absolute depth with respect to the surface of the cortex
        delay_image         : longblob      # (ms) delay between the start of the scan and pixels in this field
        roi                 : tinyint       # ROI to which this field belongs
        valid_depth=false   : boolean       # whether depth has been manually check
        """

