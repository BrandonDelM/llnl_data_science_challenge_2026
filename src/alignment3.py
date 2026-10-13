import json
import numpy as np
import trimesh
import open3d as o3d
import pyvista as pv
import tifffile
import os

def run_unified_registration_and_cropping(stl_in, json_in, tif_in, stl_out, tif_out):
    # ==========================================
    # PHASE 1: ALIGNMENT (Trimesh + Open3D)
    # ==========================================
    print("=== PHASE 1: ICP Registration ===")
    print("Loading JSON and STL for alignment...")
    with open(json_in, 'r') as f:
        json_data = json.load(f)
    json_points = np.array([j["position"] for j in json_data["junctions"]])

    mesh = trimesh.load(stl_in)
    vertices = mesh.vertices.copy()

    # Isolate the 9x9x9 core along the Y-axis to give ICP a clean target
    y_coords = vertices[:, 1]
    y_min, y_max = y_coords.min(), y_coords.max()
    unit_cell_y = (y_max - y_min) / 11.0
    crop_min_y = y_min + (unit_cell_y * 0.95)
    crop_max_y = y_max - (unit_cell_y * 0.95)
    
    cropped_vertices = vertices[(y_coords >= crop_min_y) & (y_coords <= crop_max_y)]

    # Prepare Open3D point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(cropped_vertices)
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(json_points)

    # Estimate Scale & Centers
    source_center = source_pcd.get_center()
    target_center = target_pcd.get_center()
    scale_factor = np.mean(
        (target_pcd.get_max_bound() - target_pcd.get_min_bound()) / 
        (source_pcd.get_max_bound() - source_pcd.get_min_bound())
    )

    # Create chained rotation matrices (Z by 90, then X by -90)
    tz = np.radians(90)
    cz, sz = np.cos(tz), np.sin(tz)
    R_z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

    # tx = np.radians(-90)
    # cx, sx = np.cos(tx), np.sin(tx)
    # R_x = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])

    # R_combined = R_x @ R_z

    # Build Initial Transformation Matrix
    init_transform = np.eye(4)
    init_transform[:3, :3] = R_z * scale_factor
    init_transform[:3, 3] = target_center - (init_transform[:3, :3] @ source_center)

    print("Running ICP...")
    correspondence_threshold = unit_cell_y * scale_factor * 0.5
    icp_result = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, correspondence_threshold, init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )

    # Apply transform to the full mesh and save a temporary checkpoint
    aligned_mesh = mesh.copy()
    aligned_mesh.apply_transform(icp_result.transformation)
    
    temp_stl_path = "temp_aligned_full.stl"
    aligned_mesh.export(temp_stl_path)


    # ==========================================
    # PHASE 2: CENTERING & CROPPING (PyVista + Numpy)
    # ==========================================
    print("\n=== PHASE 2: Centering & Cropping ===")
    
    # 1. Derive zero-centered JSON bounds
    json_center = json_points.mean(axis=0)
    aligned_points = json_points - json_center

    x_min, x_max = aligned_points[:, 0].min(), aligned_points[:, 0].max()
    y_min, y_max = aligned_points[:, 1].min(), aligned_points[:, 1].max()
    z_min, z_max = aligned_points[:, 2].min(), aligned_points[:, 2].max()
    crop_bounds = [x_min, x_max, y_min, y_max, z_min, z_max]
    print(f"Target crop bounds: {np.round(crop_bounds, 2)}")

    # 2. Load aligned STL into PyVista and center it
    pv_stl = pv.read(temp_stl_path)
    stl_center = np.array(pv_stl.center)
    pv_stl.translate(-stl_center, inplace=True)

    # Crop and save final STL
    print("Cropping STL...")
    cropped_stl = pv_stl.clip_box(crop_bounds, invert=False)
    final_stl = cropped_stl.extract_surface().triangulate()
    final_stl.save(stl_out)
    print(f"Saved: {stl_out}")

    # 3. Load TIF and find its PyVista threshold center
    print("Loading and thresholding TIF...")
    raw_volume = tifffile.imread(tif_in)
    volume_thresh = raw_volume.copy()
    volume_thresh[volume_thresh < 43400] = 0

    grid = pv.ImageData()
    grid.dimensions = np.array(volume_thresh.shape) + 1
    grid.spacing = (1, 1, 1)
    grid.origin = (0, 0, 0)
    grid.cell_data['density'] = volume_thresh.flatten(order='F')

    tif_surface = grid.threshold(volume_thresh.max() * 0.5).extract_surface()
    tif_center = np.array(tif_surface.center)
    tif_surface.translate(-tif_center, inplace=True)
    
    cropped_tif_surface = tif_surface.clip_box(crop_bounds, invert=False)

    # 4. Crop raw TIF array mathematically
    print("Cropping Numpy TIF Array...")
    idx_x_min = max(0, int(np.floor(crop_bounds[0] + tif_center[0])))
    idx_x_max = min(raw_volume.shape[0], int(np.ceil(crop_bounds[1] + tif_center[0])))
    
    idx_y_min = max(0, int(np.floor(crop_bounds[2] + tif_center[1])))
    idx_y_max = min(raw_volume.shape[1], int(np.ceil(crop_bounds[3] + tif_center[1])))
    
    idx_z_min = max(0, int(np.floor(crop_bounds[4] + tif_center[2])))
    idx_z_max = min(raw_volume.shape[2], int(np.ceil(crop_bounds[5] + tif_center[2])))

    print("raw_volume.shape:", raw_volume.shape)
    print("crop_bounds (X,Y,Z):", crop_bounds)
    raw_cropped_volume = raw_volume[idx_x_min:idx_x_max, idx_y_min:idx_y_max, idx_z_min:idx_z_max]
    tifffile.imwrite(tif_out, raw_cropped_volume)
    print(f"Saved: {tif_out}")

    # Cleanup temporary file
    if os.path.exists(temp_stl_path):
        os.remove(temp_stl_path)


    # ==========================================
    # PHASE 3: VERIFICATION PLOT
    # ==========================================
    print("\n=== PHASE 3: Rendering Overlay ===")
    pl = pv.Plotter(shape=(1, 3))

    pl.subplot(0, 0)
    pl.add_mesh(cropped_tif_surface, color='red', opacity=0.7)
    pl.add_bounding_box(color='black', line_width=2)
    pl.add_text("Cropped TIF", font_size=10, color='black')
    pl.enable_parallel_projection()

    pl.subplot(0, 1)
    pl.add_mesh(cropped_stl, color='blue', opacity=0.7)
    pl.add_bounding_box(color='black', line_width=2)
    pl.add_text("Cropped STL", font_size=10, color='black')
    pl.enable_parallel_projection()

    pl.subplot(0, 2)
    pl.add_mesh(cropped_tif_surface, color='red', opacity=0.5, label='TIF')
    pl.add_mesh(cropped_stl, color='blue', opacity=0.5, label='STL')
    pl.add_bounding_box(color='black', line_width=2)
    pl.add_text("Overlaid (Intersection)", font_size=10, color='black')
    pl.enable_parallel_projection()

    pl.link_views()
    pl.set_background('white')
    pl.show()

if __name__ == "__main__":
    # Standardized paths for data execution
    STL_INPUT = "data/missing_struts/stls/0.stl"
    JSON_INPUT = "data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
    TIF_INPUT = "data/missing_struts/tif_stacks/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.tif"
    
    STL_OUTPUT = "final_json_cropped_aligned.stl"
    TIF_OUTPUT = "final_json_cropped_aligned.tif"
    
    run_unified_registration_and_cropping(
        stl_in=STL_INPUT,
        json_in=JSON_INPUT,
        tif_in=TIF_INPUT,
        stl_out=STL_OUTPUT,
        tif_out=TIF_OUTPUT
    )