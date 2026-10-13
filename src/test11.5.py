import numpy as np
import pyvista as pv
import tifffile
import json

# --- 1. Load the JSON Points ---
json_path = 'data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json'
with open(json_path, 'r') as f:
    data = json.load(f)

raw_points = np.array([j['position'] for j in data['junctions']])
json_center = raw_points.mean(axis=0)

print(f"Loaded {len(raw_points)} points from JSON.")

# Shift JSON to (0,0,0) to match TIF space
aligned_points = raw_points - json_center


# --- 2. Load the ALREADY REGISTERED STL ---
# Note: Load the output from your Open3D ICP script here. 
# No 90-degree Z-rotation is needed because it's baked into the file!
print("Loading ICP-aligned STL...")
stl_mesh = pv.read('registered_model_0.stl')

# Shift the STL to (0,0,0)
stl_center = np.array(stl_mesh.center)
stl_mesh.translate(-stl_center, inplace=True)


# --- 3. Calculate Bounds from Centered Space ---
x_min, x_max = aligned_points[:, 0].min(), aligned_points[:, 0].max()
y_min, y_max = aligned_points[:, 1].min(), aligned_points[:, 1].max()
z_min, z_max = aligned_points[:, 2].min(), aligned_points[:, 2].max()

crop_bounds = [x_min, x_max, y_min, y_max, z_min, z_max]
print(f"Calculated Crop Bounds from JSON: {np.round(crop_bounds, 2)}")


# --- 4. Load & Align the TIF ---
print("Loading and thresholding TIF...")
volume = tifffile.imread('data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif')
volume = volume.copy()
volume[volume < 43400] = 0

grid = pv.ImageData()
grid.dimensions = np.array(volume.shape) + 1
grid.spacing = (1, 1, 1)
grid.origin = (0, 0, 0)
grid.cell_data['density'] = volume.flatten(order='F')

threshold = volume.max() * 0.5
tif_surface = grid.threshold(threshold).extract_surface()

# Center the TIF at (0,0,0)
tif_center = np.array(tif_surface.center)
tif_surface.translate(-tif_center, inplace=True)


# --- 5. Crop Models Using the Derived Bounds ---
print("Cropping models based on dynamic JSON bounds...")
cropped_tif = tif_surface.clip_box(crop_bounds, invert=False)
cropped_stl = stl_mesh.clip_box(crop_bounds, invert=False)

# Optional: Save the cropped mesh if you specifically want the trimmed 9x9x9 version
final_stl = cropped_stl.extract_surface().triangulate()
final_stl.save("registered_model_0_v2.stl")


# --- 6. Crop and Save the Raw TIF Array ---
idx_x_min = max(0, int(np.floor(crop_bounds[0] + tif_center[0])))
idx_x_max = min(volume.shape[0], int(np.ceil(crop_bounds[1] + tif_center[0])))

idx_y_min = max(0, int(np.floor(crop_bounds[2] + tif_center[1])))
idx_y_max = min(volume.shape[1], int(np.ceil(crop_bounds[3] + tif_center[1])))

idx_z_min = max(0, int(np.floor(crop_bounds[4] + tif_center[2])))
idx_z_max = min(volume.shape[2], int(np.ceil(crop_bounds[5] + tif_center[2])))

print(f"Cropping raw array at X: {idx_x_min}-{idx_x_max}, Y: {idx_y_min}-{idx_y_max}, Z: {idx_z_min}-{idx_z_max}")

# Slice the original numpy array
raw_cropped_volume = volume[idx_x_min:idx_x_max, idx_y_min:idx_y_max, idx_z_min:idx_z_max]

# Save the numpy array
tifffile.imwrite("test/json_cropped_aligned_tif.tif", raw_cropped_volume)
print("Saved cropped TIF successfully!")


# --- 7. Plotting in 1x3 Subplots ---
print("Generating plot...")
pl = pv.Plotter(shape=(1, 3))

# Subplot 1: Cropped TIF
pl.subplot(0, 0)
pl.add_mesh(cropped_tif, color='red', opacity=0.7)
pl.add_bounding_box(color='black', line_width=2)
pl.add_text("JSON Points Cropped TIF", font_size=10, color='black')
pl.enable_parallel_projection()
pl.add_axes()
pl.show_grid()

# Subplot 2: Cropped STL
pl.subplot(0, 1)
pl.add_mesh(cropped_stl, color='blue', opacity=0.7)
pl.add_bounding_box(color='black', line_width=2)
pl.add_text("JSON Points Cropped STL", font_size=10, color='black')
pl.enable_parallel_projection()
pl.add_axes()
pl.show_grid()

# Subplot 3: Overlaid Alignment
pl.subplot(0, 2)
pl.add_mesh(cropped_tif, color='red', opacity=0.5, label='TIF')
pl.add_mesh(cropped_stl, color='blue', opacity=0.5, label='STL')
pl.add_bounding_box(color='black', line_width=2)
pl.add_text("Overlaid (Intersection)", font_size=10, color='black')
pl.enable_parallel_projection()
pl.add_axes()
pl.show_grid()

pl.link_views()
pl.set_background('white')
pl.show()