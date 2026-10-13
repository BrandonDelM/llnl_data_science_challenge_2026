import json
import numpy as np
import pyvista as pv

def plot_centered_lattice_and_stl(json_path, stl_path, rotation_degrees=90, tube_radius=0.5):
    print("1. Loading JSON Lattice (Junctions and Struts)...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    junctions = data['junctions']
    struts = data['struts']

    # --- Build Points Array ---
    max_id = max(j['id'] for j in junctions)
    # Initialize array. Note: Unused IDs will remain [0,0,0], which is fine 
    # as long as the struts array doesn't reference them.
    points = np.zeros((max_id + 1, 3))
    
    # Track valid positions strictly for calculating the true center
    valid_positions = []
    
    for j in junctions:
        pos = j['position']
        points[j['id']] = pos
        valid_positions.append(pos)
        
    # Calculate the centroid of the JSON lattice
    json_center = np.mean(valid_positions, axis=0)
    
    # Shift ALL points so the lattice centroid sits at (0, 0, 0)
    centered_points = points - json_center

    # --- Build Strut Tubes ---
    # PyVista line format: [n_points_in_line, p0, p1, n_points_in_line, p0, p1, ...]
    lines = []
    for s in struts:
        lines.extend([2, s['junction0'], s['junction1']])
    lines = np.array(lines)

    # Create PolyData of line segments
    lattice_mesh = pv.PolyData(centered_points, lines=lines)
    
    # Convert lines -> tubes (cylinders) using the provided thickness
    lattice_tubes = lattice_mesh.tube(radius=tube_radius)

    # --- Optional: Extract Junctions for plotting as Spheres ---
    centered_valid_positions = np.array(valid_positions) - json_center
    junctions_pcd = pv.PolyData(centered_valid_positions)

    print("2. Loading and Aligning STL mesh...")
    mesh = pv.read(stl_path)
    mesh_center = np.array(mesh.center)
    
    # Shift STL to (0,0,0) to perfectly match the JSON center
    mesh.translate(-mesh_center, inplace=True)
    
    # Rotate 90 degrees around Z-axis, spinning on its new center
    # mesh.rotate_z(rotation_degrees, point=(0, 0, 0), inplace=True)

    print("3. Rendering 3D overlay...")
    plotter = pv.Plotter()
    
    # Plot the STL mesh (Semi-transparent so we can see inside)
    plotter.add_mesh(
        mesh, 
        color="lightblue", 
        opacity=0.4, 
        label="Centered STL Mesh"
    )
    
    # Plot the Struts (Silver Tubes)
    plotter.add_mesh(
        lattice_tubes, 
        color='red', 
        smooth_shading=True,
        label="JSON Struts"
    )

    # Plot the Junctions (Red Spheres) 
    # You can comment this out if you only want the tubes!
    plotter.add_mesh(
        junctions_pcd, 
        color="red", 
        point_size=10, 
        render_points_as_spheres=True, 
        label="JSON Nodes"
    )
    
    # Add your reference plane (optional)
    plane = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=5, j_size=5)
    plotter.add_mesh(plane, show_edges=True, color='lightblue', opacity=0.3)

    # Set camera and UI
    plotter.camera_position = 'xz'
    plotter.add_legend()
    plotter.set_background("white")
    
    plotter.show()

if __name__ == "__main__":
    # Ensure these paths point to your actual files
    JSON_PATH = "data/missing_struts/registered_jsons/210127_Brian_Tran_strut_lattices_0point5dash1 1 Slices.json"
    STL_PATH = "registered_model_0.stl"
    
    plot_centered_lattice_and_stl(JSON_PATH, STL_PATH, rotation_degrees=90, tube_radius=1.5)