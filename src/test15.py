import numpy as np
import pyvista as pv
import tifffile

# ---------- 1. Load RAW TIF & Align Grid ----------
tif_path = 'data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif'
volume = tifffile.imread(tif_path)
volume[volume < 43400] = 0

grid = pv.ImageData()
grid.dimensions = np.array(volume.shape) + 1
grid.spacing = (1, 1, 1)

grid.origin = (0, 0, 0)
grid.origin = -np.array(grid.center)  # Shift TIF canvas to (0,0,0)

ox, oy, oz = grid.origin
sx, sy, sz = grid.spacing

# ---------- 2. Load and Align the STL ----------
stl_mesh = pv.read('perfectly_aligned_cropped_model.stl')
stl_mesh.rotate_z(90, inplace=True)  
stl_mesh.translate(-np.array(stl_mesh.center), inplace=True)
stl_bounds = stl_mesh.bounds

# ---------- 3. Map STL Bounds to TIF Indices ----------
idx_x_min = max(0, int(np.floor((stl_bounds[0] - ox) / sx)))
idx_x_max = min(volume.shape[0], int(np.ceil((stl_bounds[1] - ox) / sx)))

idx_y_min = max(0, int(np.floor((stl_bounds[2] - oy) / sy)))
idx_y_max = min(volume.shape[1], int(np.ceil((stl_bounds[3] - oy) / sy)))

idx_z_min = max(0, int(np.floor((stl_bounds[4] - oz) / sz)))
idx_z_max = min(volume.shape[2], int(np.ceil((stl_bounds[5] - oz) / sz)))

# ---------- 4. Crop and Save the TIF ----------
print("Cropping and saving TIF...")
cropped_volume = volume[idx_x_min:idx_x_max, idx_y_min:idx_y_max, idx_z_min:idx_z_max]

tifffile.imwrite('test/final_co_cropped_image.tif', cropped_volume)
print("Saved TIF! Shape:", cropped_volume.shape)

# ---------- 5. Back-Calculate Grid-Snapped Bounds ----------
# This computes the EXACT physical space the newly cropped TIF array occupies
snapped_bounds = [
    ox + (idx_x_min * sx), ox + (idx_x_max * sx),
    oy + (idx_y_min * sy), oy + (idx_y_max * sy),
    oz + (idx_z_min * sz), oz + (idx_z_max * sz)
]

print(f"\nOriginal STL Bounds: {np.round(stl_bounds, 2)}")
print(f"Snapped Grid Bounds: {np.round(snapped_bounds, 2)}")

# ---------- 6. Crop and Save the STL to the Snapped Bounds ----------
print("\nCropping STL to perfectly match the grid-snapped TIF...")
cropped_stl = stl_mesh.clip_box(snapped_bounds, invert=False)

# Convert from UnstructuredGrid back to STL-compatible triangulated surface
final_stl = cropped_stl.extract_surface().triangulate()
final_stl.save('test/final_co_cropped_model.stl')
print("Saved STL!")

print("\nSuccess! Both files are perfectly co-cropped and share the exact same bounding box.")