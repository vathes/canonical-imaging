import datajoint as dj

from . import scope, mesoscan
from .mesoscan import schema
from djutils.templates import required


@schema
class RasterCorrection(dj.Computed):
    definition = """ # raster correction for bidirectional resonant scans
    -> mesoscan.ScanInfo                # animal_id, session, scan_idx, version
    -> CorrectionChannel                # animal_id, session, scan_idx, field
    ---
    raster_template     : longblob      # average frame from the middle of the movie
    raster_phase        : float         # difference between expected and recorded scan angle
    """


@schema
class MotionCorrection(dj.Computed):
    definition = """ # motion correction for galvo scans
    -> RasterCorrection
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
class SummaryImages(dj.Computed):
    definition = """ # summary images for each field and channel after corrections
    -> MotionCorrection
    -> scope.Channel
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

    class L6Norm(dj.Part):
        definition = """ # l6-norm of each pixel across time
        -> master
        ---
        l6norm_image           : longblob
        """
