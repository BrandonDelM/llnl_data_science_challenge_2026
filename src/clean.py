import json
import numpy as np
import pyvista as pv
import tifffile

JSON_PATH = (
    "data/missing_struts/registered_jsons/"
    "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
)
TIF_PATH = (
    "data/missing_struts/tif_stacks/"
    "210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
)
STL_PATH = "registered_model.stl"

# --------------------------------------------------
# 1. Load registered JSON bounds
# --------------------------------------------------

with open(JSON_PATH) as f:
    data = json.load(f)

junctions = np.array([j["position"] for j in data["junctions"]])

json_min = junctions.min(axis=0)
json_max = junctions.max(axis=0)

print("Registered JSON bounds:")
print("min:", json_min)
print("max:", json_max)

# --------------------------------------------------
# 2. Load registered STL
# --------------------------------------------------

stl = pv.read(STL_PATH)
# DO NOT rotate yet! Leave it aligned with the JSON for clipping.

print("Original STL bounds:", stl.bounds)

# --------------------------------------------------
# 3. Remove the extra Y boundary geometry
# --------------------------------------------------

# Use the JSON's Y-range, with a small margin.
margin_y = 2.0

crop_bounds = [
    json_min[0] - 2.0,
    json_max[0] + 2.0,
    json_min[1] - margin_y,
    json_max[1] + margin_y,
    json_min[2] - 2.0,
    json_max[2] + 2.0,
]

# Clip the box while it still matches the JSON coordinates
stl_clean = (
    stl.clip_box(crop_bounds, invert=False)
        .extract_surface()
        .triangulate()
)

# NOW rotate the cleanly cropped mesh by 90 degrees Z
stl_clean.rotate_z(-90, inplace=True)

stl_clean.save("test/registered_model_clean.stl")

print("Clean and Rotated STL bounds:", stl_clean.bounds)

# --------------------------------------------------
# 4. Load TIFF as Z,Y,X and convert to X,Y,Z
# --------------------------------------------------

volume_xyz = tifffile.imread(TIF_PATH)
# volume_xyz = np.transpose(volume_zyx, (2, 1, 0))

volume_xyz = volume_xyz.copy()
volume_xyz[volume_xyz < 43400] = 0

grid = pv.ImageData()
grid.dimensions = np.array(volume_xyz.shape) + 1
grid.spacing = (1.0, 1.0, 1.0)
grid.origin = (0.0, 0.0, 0.0)
grid.cell_data["density"] = volume_xyz.flatten(order="F")

tif_surface = grid.threshold(volume_xyz.max() * 0.5)

print("TIFF bounds:", tif_surface.bounds)

# --------------------------------------------------
# 4.5 Align Coordinate Spaces
# --------------------------------------------------
print("Aligning STL centroid to TIF centroid...")

tif_center = np.array(tif_surface.center)
stl_center = np.array(stl_clean.center)

# Calculate the distance between the two coordinate spaces
translation_vector = tif_center - stl_center

# Shift the STL into the TIF's pixel space
stl_clean.translate(translation_vector, inplace=True)

print("New STL bounds after translation:", stl_clean.bounds)

# --------------------------------------------------
# 5. Choose one shared physical slice coordinate
# --------------------------------------------------

# Use the TIFF coordinate system, not each object's own center.
y_min, y_max = tif_surface.bounds[2], tif_surface.bounds[3]
y_slice = 0.5 * (y_min + y_max)

tif_slice = tif_surface.slice(
    normal=(0, 1, 0),
    origin=(0, y_slice, 0),
)

stl_slice = stl_clean.slice(
    normal=(0, 1, 0),
    origin=(0, y_slice, 0),
)

print("Shared Y slice:", y_slice)
print("TIFF slice points:", tif_slice.n_points)
print("STL slice points:", stl_slice.n_points)

# --------------------------------------------------
# 6. Overlay
# --------------------------------------------------

plotter = pv.Plotter()
plotter.add_mesh(tif_slice, color="red", opacity=0.55)
plotter.add_mesh(stl_slice, color="blue", opacity=0.55)
plotter.view_xz()
plotter.enable_parallel_projection()
plotter.show_grid()
plotter.show(screenshot="test/clean_center_slice.png")