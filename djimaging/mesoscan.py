import datajoint as dj
import scanreader

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
        z                   : float         # (um) relative depth of field
        delay_image         : longblob      # (ms) delay between the start of the scan and pixels in this field
        roi                 : tinyint       # ROI to which this field belongs
        valid_depth=false   : boolean       # whether depth has been manually check
        """

    @staticmethod
    @required
    def _get_scan_image_files():
        return None

    def make(self, key):
        """ Read and store some scan parameters."""
        # Read the scan
        print('Reading header...')
        scan_filename = self._get_scan_image_files(key)
        scan = scanreader.read_scan(scan_filename)

        # Insert in ScanInfo
        self.insert1(dict(key,
                          nfields=scan.num_fields,
                          nchannels=scan.nchannels,
                          nframes=scan.nframes,
                          nframes_requested=scan.nframes_requested,
                          x=scan.motor_position_at_zero[0],
                          y=scan.motor_position_at_zero[1],
                          fps=scan.fps,
                          bidirectional=scan.is_bidirectional,
                          usecs_per_line=scan.seconds_per_line * 1e6,
                          fill_fraction=scan.temporal_fill_fraction,
                          nrois=scan.num_rois,
                          valid_depth=True))

        # Insert Field(s)
        for field_id in range(scan.num_fields):
            x_zero, y_zero, _ = scan.motor_position_at_zero  # motor x, y at ScanImage's 0
            self.Field.insert1(dict(key,
                                    field=field_id,
                                    px_height=scan.field_heights[field_id],
                                    px_width=scan.field_widths[field_id],
                                    um_height=scan.field_heights_in_microns[field_id],
                                    um_width=scan.field_widths_in_microns[field_id],
                                    x=x_zero + scan._degrees_to_microns(scan.fields[field_id].x),
                                    y=y_zero + scan._degrees_to_microns(scan.fields[field_id].y),
                                    z=scan.field_depths[field_id],
                                    delay_image=scan.field_offsets[field_id],
                                    roi=scan.field_rois[field_id][0]))

        # Fill in CorrectionChannel if only one channel
        if scan.num_channels == 1:
            CorrectionChannel.insert([dict(key, field=field_id, channel=0)
                                      for field_id in range(scan.num_fields)], ignore_extra_fields=True)


@schema
class CorrectionChannel(dj.Manual):
    definition = """ # channel to use for raster and motion correction
    -> mesoscan.Scan
    -> scope.Field
    ---
    -> scope.Channel
    """