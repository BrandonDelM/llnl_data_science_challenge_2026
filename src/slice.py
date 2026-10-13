import numpy as np
import pyvista as pv
import tifffile

def map_tif_page_to_stl(stl_path, tif_path, target_page):
    print(f"Loading files...")
    # Load the final cropped/aligned files
    mesh = pv.read(stl_path)
    volume = tifffile.imread(tif_path)
    
    # Assuming volume shape is (X, Y, Z). If your TIF loads as (Z, Y, X), 
    # change volume.shape[2] to volume.shape[0] below.
    z_total_pages = volume.shape[2] 
    tif_z_center = z_total_pages / 2.0

    if target_page < 0 or target_page >= z_total_pages:
        raise ValueError(f"target_page must be between 0 and {z_total_pages-1}")

    # ==========================================
    # THE MAPPING CALCULATION
    # ==========================================
    z_physical = target_page - tif_z_center
    print(f"TIF Page [{target_page}] maps exactly to PyVista Z-Height: {z_physical}")

    # 1. Extract the 2D image array for this specific TIF page
    # (Applying a threshold so the struts are highly visible)
    tif_slice_image = volume[:, :, target_page].copy()
    tif_slice_image[tif_slice_image < 43400] = 0 

    # 2. Slice the 3D STL mesh at this exact physical Z-height
    # This creates a 2D cross-section of the mesh
    mesh_slice = mesh.slice(normal='z', origin=(0, 0, z_physical))

    # ==========================================
    # VISUALIZATION
    # ==========================================
    pl = pv.Plotter(shape=(1, 2))

    # Subplot 1: The Cross-Section of the STL at z_physical
    pl.subplot(0, 0)
    pl.add_mesh(mesh, color='lightblue', opacity=0.2, style='wireframe')
    pl.add_mesh(mesh_slice, color='red', line_width=5)
    
    # Draw a plane to visualize the slice height natively in 3D
    plane = pv.Plane(center=(0, 0, z_physical), direction=(0, 0, 1), 
                     i_size=volume.shape[0], j_size=volume.shape[1])
    pl.add_mesh(plane, color='gray', opacity=0.3)
    
    pl.add_text(f"STL Sliced at Z = {z_physical}", font_size=10, color='black')
    pl.view_isometric()

    # Subplot 2: The Raw TIF Page Data mapped to a 2D grid
    pl.subplot(0, 1)
    grid2d = pv.ImageData()
    grid2d.dimensions = (volume.shape[0], volume.shape[1], 1)
    grid2d.spacing = (1, 1, 1)
    grid2d.point_data['density'] = tif_slice_image.flatten(order='F')
    
    # Threshold the 2D grid just to show the strut pixels
    thresh2d = grid2d.threshold(volume.max() * 0.5)
    pl.add_mesh(thresh2d, color='red')
    pl.add_text(f"Raw TIF Page Index: {target_page}", font_size=10, color='black')
    pl.view_xy()

    pl.set_background('white')
    pl.show()

if __name__ == "__main__":
    STL_FILE = "registered_model_0_v2.stl"
    TIF_FILE = "final_json_cropped_aligned.tif"
    
    # Pick a specific page from your TIF stack to test the mapping
    PAGE_TO_TEST = 45
    
    map_tif_page_to_stl(STL_FILE, TIF_FILE, target_page=PAGE_TO_TEST)