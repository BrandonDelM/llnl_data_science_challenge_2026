import json
import numpy as np
import trimesh
import pyvista as pv
import tifffile

# ---------- Config ----------
STL_INPUT = "data/missing_struts/stls/0.stl"
JSON_INPUT = "data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
TIF_INPUT = "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"

DENSITY_THRESHOLD = 43400

# Use the crop_bounds you already computed
crop_bounds = [-357.49203407133825, 357.49203407134115,
               -358.18517259274404, 358.18517259274864,
               -356.67550500190777, 356.6755050019068]

# ---------- Load raw TIF and derive index bounds ----------
raw_volume = tifffile.imread(TIF_INPUT)
print("raw_volume.shape:", raw_volume.shape)

# Recompute tif_center exactly as your pipeline does, so indices line up
volume_thresh = raw_volume.copy()
volume_thresh[volume_thresh < DENSITY_THRESHOLD] = 0

grid = pv.ImageData()
grid.dimensions = np.array(volume_thresh.shape) + 1
grid.spacing = (1, 1, 1)
grid.origin = (0, 0, 0)
grid.cell_data['density'] = volume_thresh.flatten(order='F')

tif_surface = grid.threshold(volume_thresh.max() * 0.5).extract_surface()
tif_center = np.array(tif_surface.center)
print("tif_center:", tif_center)

idx_x_min = max(0, int(np.floor(crop_bounds[0] + tif_center[0])))
idx_x_max = min(raw_volume.shape[0], int(np.ceil(crop_bounds[1] + tif_center[0])))

idx_y_min = max(0, int(np.floor(crop_bounds[2] + tif_center[1])))
idx_y_max = min(raw_volume.shape[1], int(np.ceil(crop_bounds[3] + tif_center[1])))

idx_z_min = max(0, int(np.floor(crop_bounds[4] + tif_center[2])))
idx_z_max = min(raw_volume.shape[2], int(np.ceil(crop_bounds[5] + tif_center[2])))

print(f"X idx: {idx_x_min}-{idx_x_max}")
print(f"Y idx: {idx_y_min}-{idx_y_max}")
print(f"Z idx: {idx_z_min}-{idx_z_max}")

# ---------- Two candidate crops: original order vs Y/Z swapped ----------
raw_cropped_v1 = raw_volume[idx_x_min:idx_x_max, idx_y_min:idx_y_max, idx_z_min:idx_z_max]
raw_cropped_v2 = raw_volume[idx_x_min:idx_x_max, idx_z_min:idx_z_max, idx_y_min:idx_y_max]

print("v1 (original order) shape:", raw_cropped_v1.shape)
print("v2 (Y/Z swapped) shape:", raw_cropped_v2.shape)

tifffile.imwrite('test_v1_original_order.tif', raw_cropped_v1)
tifffile.imwrite('test_v2_yz_swapped.tif', raw_cropped_v2)

# ---------- Build STL for comparison (with your confirmed rotation) ----------
mesh = trimesh.load(STL_INPUT)

# Apply the same alignment rotation used in your pipeline: Z by 90, then X by -90
Rz = trimesh.transformations.rotation_matrix(np.radians(90), [0, 0, 1])
Rx = trimesh.transformations.rotation_matrix(np.radians(-90), [1, 0, 0])
mesh.apply_transform(Rz)
mesh.apply_transform(Rx)

# NOTE: this test script does NOT redo full ICP/scale — it assumes you already
# have an aligned+cropped STL saved. Point this at that file instead if you have one:
# stl_mesh = pv.read('final_json_cropped_aligned.stl')
stl_mesh = pv.wrap(mesh)
stl_mesh.translate(-np.array(stl_mesh.center), inplace=True)
print("STL bounds (post-rotation, centered):", stl_mesh.bounds)


# ---------- Helper: build a pv surface from a numpy volume ----------
def volume_to_surface(volume, threshold_frac=0.5):
    v = volume.copy()
    g = pv.ImageData()
    g.dimensions = np.array(v.shape) + 1
    g.spacing = (1, 1, 1)
    g.origin = (0, 0, 0)
    g.cell_data['density'] = v.flatten(order='F')
    surf = g.threshold(v.max() * threshold_frac).extract_surface()
    surf.translate(-np.array(surf.center), inplace=True)
    return surf


v1_surface = volume_to_surface(raw_cropped_v1)
v2_surface = volume_to_surface(raw_cropped_v2)

# ---------- Plot: STL vs v1 vs v2, side by side ----------
pl = pv.Plotter(shape=(1, 3), window_size=(1800, 600))

pl.subplot(0, 0)
pl.add_mesh(stl_mesh, color='silver')
pl.add_text("STL (reference)", font_size=10)
pl.enable_parallel_projection()
pl.view_isometric()

pl.subplot(0, 1)
pl.add_mesh(v1_surface, color='red')
pl.add_text("TIF crop v1 (original order)", font_size=10)
pl.enable_parallel_projection()
pl.view_isometric()

pl.subplot(0, 2)
pl.add_mesh(v2_surface, color='blue')
pl.add_text("TIF crop v2 (Y/Z swapped)", font_size=10)
pl.enable_parallel_projection()
pl.view_isometric()

pl.link_views()
pl.set_background('white', all_renderers=True)
pl.show(screenshot='axis_order_test.png')