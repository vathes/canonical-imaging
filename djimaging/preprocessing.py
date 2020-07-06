import datajoint as dj
import scanreader
import numpy as np

from . import scope, imaging
from .imaging import schema
from ..utils import galvo_corrections, performance


@schema
class RasterCorrection(dj.Computed):
    definition = """ # raster correction for bidirectional resonant scans
    -> mesoscan.ScanInfo                # animal_id, session, scan_idx, version
    -> CorrectionChannel                # animal_id, session, scan_idx, field
    ---
    raster_template     : longblob      # average frame from the middle of the movie
    raster_phase        : float         # difference between expected and recorded scan angle
    """

    def get_correct_raster(self):
        """ Returns a function to perform raster correction on the scan. """
        raster_phase = self.fetch1('raster_phase')
        fill_fraction = (imaging.ScanInfo & self).fetch1('fill_fraction')
        if abs(raster_phase) < 1e-7:
            correct_raster = lambda scan: scan.astype(np.float32, copy=False)
        else:
            correct_raster = lambda scan: galvo_corrections.correct_raster(scan,
                                                             raster_phase, fill_fraction)
        return correct_raster

    def make(self, key):
        from scipy.signal.windows import tukey

        # Read the scan
        scan_filename = imaging.ScanInfo._get_scan_image_files(key)
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)

        # Select correction channel
        channel = (scan.CorrectionChannel & key).fetch1('channel')
        field_id = key['field']

        # Load some frames from the middle of the scan
        middle_frame = int(np.floor(scan.num_frames / 2))
        frames = slice(max(middle_frame - 1000, 0), middle_frame + 1000)
        mini_scan = scan[field_id, :, :, channel, frames]

        # Create template (average frame tapered to avoid edge artifacts)
        taper = np.sqrt(np.outer(tukey(scan.field_heights[field_id], 0.4),
                                 tukey(scan.field_widths[field_id], 0.4)))
        anscombed = 2 * np.sqrt(mini_scan - mini_scan.min() + 3 / 8)  # anscombe transform
        raster_template = np.mean(anscombed, axis=-1) * taper

        # Compute raster correction parameters
        if scan.is_bidirectional:
            raster_phase = galvo_corrections.compute_raster_phase(raster_template,
                                                                  scan.temporal_fill_fraction)
        else:
            raster_phase = 0

        # Insert
        self.insert1({**key, 'raster_template': raster_template, 'raster_phase': raster_phase})


@schema
class MotionCorrection(dj.Computed):
    definition = """ 
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

    def make(self, key):
        """Computes the motion shifts per frame needed to correct the scan."""
        from scipy import ndimage

        # Read the scan
        scan_filename = imaging.ScanInfo._get_scan_image_files(key)
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)

        # Get some params
        px_height, px_width = (scan.ScanInfo.Field & key).fetch1('px_height', 'px_width')
        channel = (scan.CorrectionChannel & key).fetch1('channel')
        field_id = key['field']

        # Load some frames from middle of scan to compute template
        skip_rows = int(round(px_height * 0.10))  # we discard some rows/cols to avoid edge artifacts
        skip_cols = int(round(px_width * 0.10))
        middle_frame = int(np.floor(scan.num_frames / 2))
        mini_scan = scan[field_id, skip_rows: -skip_rows, skip_cols: -skip_cols, channel,
                         max(middle_frame - 1000, 0): middle_frame + 1000]
        mini_scan = mini_scan.astype(np.float32, copy=False)

        # Correct mini scan
        correct_raster = (RasterCorrection & key).get_correct_raster()
        mini_scan = correct_raster(mini_scan)

        # Create template
        mini_scan = 2 * np.sqrt(mini_scan - mini_scan.min() + 3 / 8)  # *
        template = np.mean(mini_scan, axis=-1)
        template = ndimage.gaussian_filter(template, 0.7)  # **
        # * Anscombe tranform to normalize noise, increase contrast and decrease outliers' leverage
        # ** Small amount of gaussian smoothing to get rid of high frequency noise

        # Map: compute motion shifts in parallel
        f = performance.parallel_motion_shifts # function to map
        raster_phase = (RasterCorrection & key).fetch1('raster_phase')
        fill_fraction = (scan.ScanInfo & key).fetch1('fill_fraction')
        kwargs = {'raster_phase': raster_phase, 'fill_fraction': fill_fraction,
                  'template': template}
        results = performance.map_frames(f, scan, field_id=field_id,
                                         y=slice(skip_rows, -skip_rows),
                                         x=slice(skip_cols, -skip_cols), channel=channel,
                                         kwargs=kwargs)

        # Reduce
        y_shifts = np.zeros(scan.num_frames)
        x_shifts = np.zeros(scan.num_frames)
        for frames, chunk_y_shifts, chunk_x_shifts in results:
            y_shifts[frames] = chunk_y_shifts
            x_shifts[frames] = chunk_x_shifts

        # Detect outliers
        max_y_shift, max_x_shift = 20 / (scan.ScanInfo.Field & key).microns_per_pixel
        y_shifts, x_shifts, outliers = galvo_corrections.fix_outliers(y_shifts, x_shifts,
                                                                      max_y_shift, max_x_shift)

        # Center shifts around zero
        y_shifts -= np.median(y_shifts)
        x_shifts -= np.median(x_shifts)

        # Create results tuple
        tuple_ = key.copy()
        tuple_['motion_template'] = template
        tuple_['y_shifts'] = y_shifts
        tuple_['x_shifts'] = x_shifts
        tuple_['outlier_frames'] = outliers
        tuple_['y_std'] = np.std(y_shifts)
        tuple_['x_std'] = np.std(x_shifts)

        # Insert
        self.insert1({**key, 'motion_template': template, 'y_shifts': y_shifts, 'x_shifts': x_shifts,
                      'outlier_frames': outliers, 'y_std': np.std(y_shifts), 'x_std': np.std(x_shifts)})


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

    def make(self, key):
        # Read the scan
        scan_filename = imaging.ScanInfo._get_scan_image_files(key)
        scan = scanreader.read_scan(scan_filename, dtype=np.float32)

        for channel in range(scan.num_channels):
            # Map: Compute some statistics in different chunks of the scan
            f = performance.parallel_summary_images # function to map
            raster_phase = (RasterCorrection & key).fetch1('raster_phase')
            fill_fraction = (scan.ScanInfo & key).fetch1('fill_fraction')
            y_shifts, x_shifts = (MotionCorrection & key).fetch1('y_shifts', 'x_shifts')
            kwargs = {'raster_phase': raster_phase, 'fill_fraction': fill_fraction,
                      'y_shifts': y_shifts, 'x_shifts': x_shifts}
            results = performance.map_frames(f, scan, field_id=key['field'],
                                             channel=channel, kwargs=kwargs)

            # Reduce: Compute average images
            average_image = np.sum([r[0] for r in results], axis=0) / scan.num_frames
            l6norm_image = np.sum([r[1] for r in results], axis=0) ** (1 / 6)

            # Reduce: Compute correlation image
            sum_x = np.sum([r[2] for r in results], axis=0) # h x w
            sum_sqx = np.sum([r[3] for r in results], axis=0) # h x w
            sum_xy = np.sum([r[4] for r in results], axis=0) # h x w x 8
            denom_factor = np.sqrt(scan.num_frames * sum_sqx - sum_x ** 2)
            corrs = np.zeros(sum_xy.shape)
            for k in [0, 1, 2, 3]:
                rotated_corrs = np.rot90(corrs, k=k)
                rotated_sum_x = np.rot90(sum_x, k=k)
                rotated_dfactor = np.rot90(denom_factor, k=k)
                rotated_sum_xy = np.rot90(sum_xy, k=k)

                # Compute correlation
                rotated_corrs[1:, :, k] = (scan.num_frames * rotated_sum_xy[1:, :, k] -
                                           rotated_sum_x[1:] * rotated_sum_x[:-1]) / \
                                          (rotated_dfactor[1:] * rotated_dfactor[:-1])
                rotated_corrs[1:, 1:, 4 + k] = ((scan.num_frames * rotated_sum_xy[1:, 1:, 4 + k] -
                                                 rotated_sum_x[1:, 1:] * rotated_sum_x[:-1, : -1]) /
                                                (rotated_dfactor[1:, 1:] * rotated_dfactor[:-1, :-1]))

                # Return back to original orientation
                corrs = np.rot90(rotated_corrs, k=4 - k)

            correlation_image = np.sum(corrs, axis=-1)
            norm_factor = 5 * np.ones(correlation_image.shape) # edges
            norm_factor[[0, -1, 0, -1], [0, -1, -1, 0]] = 3 # corners
            norm_factor[1:-1, 1:-1] = 8 # center
            correlation_image /= norm_factor

            # Insert
            field_key = {**key, 'channel': channel}
            self.insert1(field_key)
            self.Average.insert1({**field_key, 'average_image': average_image})
            self.L6Norm.insert1({**field_key, 'l6norm_image': l6norm_image})
            self.Correlation.insert1({**field_key, 'correlation_image': correlation_image})
