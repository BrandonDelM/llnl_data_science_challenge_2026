import numpy as np
import pyvista as pv
import tifffile

# ---------- 1. Load TIF and Setup Grid ----------
volume = tifffile.imread('data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif')
volume = volume.copy()
volume[volume < 43400] = 0

print("TIF volume shape:", volume.shape)

grid = pv.ImageData()
grid.dimensions = np.array(volume.shape) + 1
grid.spacing = (1, 1, 1)

# Center the ENTIRE TIF grid (including the empty space) at (0,0,0)
grid.origin = (0, 0, 0) # Set to 0 to calculate true center
grid_center = np.array(grid.center)
grid.origin = -grid_center # Shift origin so the grid's center is precisely (0,0,0)

print("Full TIF Canvas Bounds (Grid):", grid.bounds)

grid.cell_data['density'] = volume.flatten(order='F')

# Because the grid is centered, the thresholded surface is inherently centered too
threshold = volume.max() * 0.5
tif_surface = grid.threshold(threshold).extract_surface()

# ---------- 2. Load and Align STL ----------
stl_mesh = pv.read('perfectly_aligned_cropped_model.stl')

# Apply the rotation that registers them
stl_mesh.rotate_z(90, inplace=True)  

# Center the STL at (0,0,0)
stl_center = np.array(stl_mesh.center)
stl_mesh.translate(-stl_center, inplace=True)

# ---------- 3. CROP THE STL ----------
# Instead of using tif_surface.bounds, we use the FULL grid.bounds
# This ensures we slice exactly at the edges of the TIF array, catching the excess metal
print("\nCropping STL using full scan bounds:", grid.bounds)

# Clip the mesh to the exact boundaries of the TIF canvas
cropped_stl = stl_mesh.clip_box(grid.bounds, invert=False)

print("Final Cropped STL bounds:", cropped_stl.bounds)

# Save the perfectly aligned and cropped STL
# cropped_stl.save('test/final_aligned_and_cropped_model_full_bounds.stl')

# ---------- 4. Plot both together to verify ----------
pl = pv.Plotter()

# Add TIF
pl.add_mesh(tif_surface, color='red', opacity=0.4, label='TIF scan')

# Add Cropped STL
pl.add_mesh(cropped_stl, color='silver', opacity=0.4, label='Cropped STL')

# Add the Grid Bounding Box to visualize the "crop box"
pl.add_mesh(grid.outline(), color='blue', line_width=2, label='TIF Canvas Bounds')
surface_mesh = cropped_stl.extract_surface()

# 2. Convert any polygon/quad cuts into strict triangles (STL requirement)
triangulated_mesh = surface_mesh.triangulate()

# 3. Save the final STL
triangulated_mesh.save("test/potential_perfect_alignment.stl")

pl.add_legend()
pl.add_axes()
pl.show_grid()
pl.show()