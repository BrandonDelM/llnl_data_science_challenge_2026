import numpy as np
import pyvista as pv
import tifffile

# ---------- Configuration ----------
SLICE_NORMAL = [0, 1, 0]  # Slicing along the Y-axis. Change to [0, 0, 1] for Z-axis.
SLICE_ORIGIN = (0, 0, 0)  # Dead center of the aligned models

# ---------- Load & Center TIF volume ----------
print("Loading and thresholding TIF...")
volume = tifffile.imread('test/json_cropped_aligned_tif.tif')
volume = volume.copy()
volume[volume < 40000] = 0

print("TIF volume shape:", volume.shape)

grid = pv.ImageData()
grid.dimensions = np.array(volume.shape) + 1
grid.spacing = (1, 1, 1)
grid.origin = (0, 0, 0)
grid.cell_data['density'] = volume.flatten(order='F')

threshold = volume.max() * 0.5
# Use extract_surface() so slicing it creates a 2D outline like the STL, not a filled solid
tif_surface = grid.threshold(threshold).extract_surface()

# Center the TIF surface at (0,0,0)
tif_center = np.array(tif_surface.center)
tif_surface.translate(-tif_center, inplace=True)
print("TIF surface bounds after centering:", tif_surface.bounds)

print("Slicing TIF...")
tif_slice = tif_surface.slice(normal=SLICE_NORMAL, origin=SLICE_ORIGIN)


# ---------- Load & Center STL mesh ----------
print("Loading and aligning STL...")
stl_mesh = pv.read('test/json_cropped_aligned_stl.stl')

# stl_mesh.rotate_z(90, inplace=True)
# print("After rotation, STL bounds:", stl_mesh.bounds)

# Center the STL at (0,0,0)
stl_center = np.array(stl_mesh.center)
stl_mesh.translate(-stl_center, inplace=True)
print("After centering, STL bounds:", stl_mesh.bounds)
stl_mesh.rotate_y(180, inplace=True)

print("Slicing STL...")
stl_slice = stl_mesh.slice(normal=SLICE_NORMAL, origin=SLICE_ORIGIN)


# ---------- Plotting in 1x3 Subplots ----------
print("Generating plot...")
pl = pv.Plotter(shape=(1, 3))

# --- Subplot 1: TIF Slice Only ---
pl.subplot(0, 0)
pl.add_mesh(tif_slice, color='red', line_width=3)
pl.add_text("TIF Slice", font_size=10, color='black')
pl.enable_parallel_projection()
if SLICE_NORMAL == [0, 1, 0]:
    pl.view_xz()
elif SLICE_NORMAL == [0, 0, 1]:
    pl.view_xy()
else:
    pl.view_yz()

# --- Subplot 2: STL Slice Only ---
pl.subplot(0, 1)
pl.add_mesh(stl_slice, color='blue', line_width=3)
pl.add_text("STL Slice", font_size=10, color='black')
pl.enable_parallel_projection()
if SLICE_NORMAL == [0, 1, 0]:
    pl.view_xz()
elif SLICE_NORMAL == [0, 0, 1]:
    pl.view_xy()
else:
    pl.view_yz()

# --- Subplot 3: Overlaid Intersection ---
pl.subplot(0, 2)
pl.add_mesh(tif_slice, color='red', line_width=3, label='TIF')
pl.add_mesh(stl_slice, color='blue', line_width=3, label='STL')
pl.add_text("Overlaid Slices (Intersection)", font_size=10, color='black')
pl.enable_parallel_projection()
if SLICE_NORMAL == [0, 1, 0]:
    pl.view_xz()
elif SLICE_NORMAL == [0, 0, 1]:
    pl.view_xy()
else:
    pl.view_yz()

# Link cameras and display
pl.link_views()
pl.set_background('white')
pl.show()