import datajoint as dj
import scanreader
import numpy as np

from djutils.templates import SchemaTemplate, required

schema = SchemaTemplate()


# ===================================== Lookup =====================================

@schema
class Field(dj.Lookup):
    definition = """ # An imaging field within a scan
    field       : tinyint  # 0-based indexing
    """
    contents = zip(range(25))


@schema
class Channel(dj.Lookup):
    definition = """  # A recording channel
    channel     : tinyint  # 0-based indexing
    """
    contents = zip(range(5))


@schema
class Plane(dj.Lookup):
    definition = """  # A recording plane (depth)
    plane     : tinyint  # 0-based indexing
    """
    contents = zip(range(25))


# ===================================== ScanImage's scan =====================================


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
    definition = """ # general data about the reso/meso scans, from ScanImage header
    -> Scan
    ---
    nfields                 : tinyint           # number of fields
    nchannels               : tinyint           # number of channels
    ndepths                 : int               # Number of scanning depths (planes)
    nframes                 : int               # number of recorded frames
    nrois                   : tinyint           # number of ROIs (see scanimage's multi ROI imaging)
    x                       : float             # (um) ScanImage's 0 point in the motor coordinate system
    y                       : float             # (um) ScanImage's 0 point in the motor coordinate system
    fps                     : float             # (Hz) frames per second
    bidirectional           : boolean           # true = bidirectional scanning
    usecs_per_line          : float             # microseconds per scan line
    fill_fraction           : float             # raster scan temporal fill fraction (see scanimage)
    """

    class ROI(dj.Part):
        definition = """ Scan's Region of Interest - for Multi-ROI imaging with ScanImage
        -> master
        roi                 : tinyint       # ROI id
        """

    class Field(dj.Part):
        definition = """ # field-specific scan information
        -> master
        -> Field
        ---
        -> Plane
        px_height           : smallint      # height in pixels
        px_width            : smallint      # width in pixels
        um_height           : float         # height in microns
        um_width            : float         # width in microns
        field_x             : float         # (um) center of field in the motor coordinate system
        field_y             : float         # (um) center of field in the motor coordinate system
        field_z             : float         # (um) relative depth of field
        delay_image         : longblob      # (ms) delay between the start of the scan and pixels in this field
        -> [nullable] ScanInfo.ROI
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
                          ndepths=scan.num_scanning_depths,
                          x=scan.motor_position_at_zero[0],
                          y=scan.motor_position_at_zero[1],
                          fps=scan.fps,
                          bidirectional=scan.is_bidirectional,
                          usecs_per_line=scan.seconds_per_line * 1e6,
                          fill_fraction=scan.temporal_fill_fraction,
                          nrois=scan.num_rois if scan.is_multiROI else 0))

        # Insert ROI(s)
        if scan.is_multiROI:
            self.ROI.insert({**key, 'roi': roi_id} for roi_id in range(scan.num_rois))

        # Insert Field(s)
        x_zero, y_zero, z_zero = scan.motor_position_at_zero  # motor x, y, z at ScanImage's 0
        self.Field.insert([dict(key,
                                field=field_id,
                                plane=np.where(np.unique(scan.field_depths) == scan.field_depths[field_id])[0][0],
                                px_height=scan.field_heights[field_id],
                                px_width=scan.field_widths[field_id],
                                um_height=scan.field_heights_in_microns[field_id],
                                um_width=scan.field_widths_in_microns[field_id],
                                field_x=x_zero + scan._degrees_to_microns(scan.fields[field_id].x),
                                field_y=y_zero + scan._degrees_to_microns(scan.fields[field_id].y),
                                field_z=z_zero + scan.fields[field_id].depth,
                                delay_image=scan.field_offsets[field_id],
                                roi=scan.field_rois[field_id][0] if scan.is_multiROI else None)
                           for field_id in range(scan.num_fields)])

