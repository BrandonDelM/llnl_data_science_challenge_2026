import pyvista as pv

def create_rotation_gif(stl_path, output_gif_path):
    print(f"Loading {stl_path}...")
    mesh = pv.read(stl_path)

    # Set up an off-screen plotter specifically for rendering animations
    # off_screen=True prevents the interactive window from popping up and slowing down the render
    pl = pv.Plotter(off_screen=True)
    
    # Add the mesh with some specular lighting for better 3D depth perception
    pl.add_mesh(mesh, color='lightblue', opacity=0.8, specular=0.5)
    pl.set_background('white')

    # Orient the camera to face the XY plane
    pl.view_xy()
    pl.camera.elevation = 30 
    
    # NOTE: view_xy() looks perfectly straight down the Z-axis (like a 2D top-down view).
    # If you want to see the 3D depth while it spins, uncomment the line below to angle the camera slightly:
    # pl.camera.elevation = 30 

    print(f"Opening {output_gif_path} for writing 360 frames...")
    # fps=30 means 360 frames will result in a 12-second animation
    pl.open_gif(output_gif_path, fps=30)  

    # Loop to generate exactly 360 frames
    for i in range(360):
        # Rotate the camera by 1 degree around the focus point
        pl.camera.azimuth += 1
        
        # Capture and save the current view as a frame in the GIF
        pl.write_frame()
        
        # Console progress update
        if (i + 1) % 60 == 0:
            print(f"Rendered {i + 1}/360 frames...")

    # Close the plotter to finalize and save the GIF file
    pl.close()
    print("GIF successfully generated!")

if __name__ == "__main__":
    # Use the final output from the unified pipeline
    STL_INPUT = "data/missing_struts/stls/0.stl"
    GIF_OUTPUT = "mesh_xy_rotation.gif"
    
    create_rotation_gif(STL_INPUT, GIF_OUTPUT)