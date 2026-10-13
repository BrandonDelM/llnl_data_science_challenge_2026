import numpy as np
import pyvista as pv
import tifffile
import json
import os

# --- 1. Load the JSON Cropping Bounds ---
# Example JSON structure expected: 
# {"bounds": [-50, 50, -20, 20, -100, 100]}  [x_min, x_max, y_min, y_max, z_min, z_max]
json_path = 'crop_bounds.json'

# (Optional) Creating a dummy JSON file for testing if it doesn't exist
if not os.path.exists(json_path):
    with open(json_path, 'w') as f:
        # Replace these numbers with your actual desired cropping box in the centered space
        json.dump({"bounds": [-50, 50, -30, 30, -50, 50]}, f)

with open(json_path, 'r') as f:
    crop_data = json.load(f)
    
# Extract the bounds from the JSON
crop_bounds = crop_data['bounds']
print(f"Loaded cropping bounds from JSON: {crop_bounds}")


# --- 2. Load & Align the TIF ---
print("Loading and thresholding TIF...")
volume = tifffile.imread('data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif')
volume = volume.copy()
volume[volume < 43400] = 0

grid = pv.ImageData()
grid.dimensions = np.array(volume.shape) + 1
grid.spacing = (1, 1, 1)
grid.origin = (0, 0, 0)
grid.cell_data['density'] = volume.flatten(order='F')

# Threshold and extract surface for visualization
threshold = volume.max() * 0.5
tif_surface = grid.threshold(threshold).extract_surface()

# ALIGNMENT: Center the TIF at (0,0,0)
tif_center = np.array(tif_surface.center)
tif_surface.translate(-tif_center, inplace=True)


# --- 3. Load & Align the STL ---
print("Loading and aligning STL...")
stl_mesh = pv.read('registered_model.stl')

# ROTATION PROBLEM: Fix the divergence in coordinate systems
stl_mesh.rotate_z(90, inplace=True)

# ALIGNMENT: Center the STL at (0,0,0) to match the TIF
stl_center = np.array(stl_mesh.center)
stl_mesh.translate(-stl_center, inplace=True)


# --- 4. Crop Both Using the JSON Bounds ---
print("Cropping models based on JSON bounds...")
# .clip_box cuts away everything outside the [xmin, xmax, ymin, ymax, zmin, zmax] limits
cropped_tif = tif_surface.clip_box(crop_bounds, invert=False)
cropped_stl = stl_mesh.clip_box(crop_bounds, invert=False)


# --- 5. Plotting in 1x3 Subplots ---
print("Generating plot...")
pl = pv.Plotter(shape=(1, 3))

# Subplot 1: Cropped TIF
pl.subplot(0, 0)
pl.add_mesh(cropped_tif, color='red', opacity=0.7)
pl.add_bounding_box(color='black', line_width=2)
pl.add_text("JSON Cropped TIF", font_size=10, color='black')
pl.enable_parallel_projection()
pl.view_isometric()

# Subplot 2: Cropped STL
pl.subplot(0, 1)
pl.add_mesh(cropped_stl, color='blue', opacity=0.7)
pl.add_bounding_box(color='black', line_width=2)
pl.add_text("JSON Cropped STL", font_size=10, color='black')
pl.enable_parallel_projection()
pl.view_isometric()

# Subplot 3: Overlaid Alignment
pl.subplot(0, 2)
pl.add_mesh(cropped_tif, color='red', opacity=0.5, label='TIF')
pl.add_mesh(cropped_stl, color='blue', opacity=0.5, label='STL')
pl.add_bounding_box(color='black', line_width=2)
pl.add_text("Overlaid (Alignment Check)", font_size=10, color='black')
pl.enable_parallel_projection()
pl.view_isometric()

# Link cameras and display
pl.link_views()
pl.set_background('white')
pl.show()