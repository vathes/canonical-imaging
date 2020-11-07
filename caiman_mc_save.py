# DataJoint Imaging Element - CaImAn Integration
# Run these commands after the CaImAn analysis has completed.
# This will save the relevant motion correction data into the 'analysis_results.hdf5' file.
# Please do not clear variables from memory prior to running these commands.
# The motion correction (mc) object will be read from memory.

# Enter the directory containing the 'analysis_results.hdf5' file
data_dir = '/'

# ---------------------------------------------------------------------------------
import h5py

filename = data_dir + '/analysis_results.hdf5'

# Load motion corrected image
mc_image = cm.load(mc.mmap_file)

# Compute motion corrected summary images
average_image = np.mean(mc_image, axis=0)
max_image = np.max(mc_image, axis=0)

# Open hdf5 file and create 'motion_correction' group
h5f = h5py.File(filename, 'a')
h5g = h5f.require_group("motion_correction")

# Write motion correction shifts and motion corrected summary images to hdf5 file
if mc.pw_rigid:
    h5g.require_dataset("x_shifts_els",shape=np.shape(mc.x_shifts_els),data=mc.x_shifts_els,dtype=mc.x_shifts_els[0][0].dtype)
    h5g.require_dataset("y_shifts_els",shape=np.shape(mc.y_shifts_els),data=mc.y_shifts_els,dtype=mc.y_shifts_els[0][0].dtype)
    h5g.require_dataset("reference_image",shape=np.shape(mc.total_template_els),data=mc.total_template_els,dtype=mc.total_template_els.dtype)
else:
    h5g.require_dataset("shifts_rig",shape=np.shape(mc.shifts_rig),data=mc.shifts_rig,dtype=mc.shifts_rig[0].dtype)
    h5g.require_dataset("reference_image",shape=np.shape(mc.total_template_rig),data=mc.total_template_rig,dtype=mc.total_template_rig.dtype)

h5g.require_dataset("correlation_image",shape=np.shape(Cn),data=Cn,dtype=Cn.dtype)
h5g.require_dataset("average_image",shape=np.shape(average_image),data=average_image,dtype=average_image.dtype)
h5g.require_dataset("max_image",shape=np.shape(max_image),data=max_image,dtype=max_image.dtype)

# Close hdf5 file
h5f.close()