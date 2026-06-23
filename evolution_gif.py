
import os
import re
import pyvista as pv

def extract_iteration_number(filename):
    """
    Extracts the numerical iteration from filenames like 'temp_5.stl' or 'geometry_12.stl'.
    """
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else -1

def create_optimization_animation(folder_path, output_filename="optimization_history.gif", fps=2):
    """
    Reads STL files from a folder chronologically and compiles a 3D animation.
    """
    # 1. Gather and sort all STL files numerically by iteration
    if not os.path.exists(folder_path):
        print(f"Error: The folder '{folder_path}' does not exist.")
        return

    all_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.stl')]
    # Filter files that contain a number (to skip non-iteration templates if any exist)
    stl_files = [f for f in all_files if extract_iteration_number(f) != -1]
    stl_files.sort(key=extract_iteration_number)

    if not stl_files:
        print(f"No valid iteration STL files found in '{folder_path}'.")
        return

    print(f"Found {len(stl_files)} iteration files. Preparing animation sequence...")

    # 2. Initialize the PyVista plotter in off-screen mode (runs in the background)
    plotter = pv.Plotter(off_screen=True)
    plotter.background_color = 'white'

    # --- FIX APPLIED HERE ---
    # Check if the output is a GIF; use open_gif directly to prevent the imageio fps conflict.
    if output_filename.lower().endswith('.gif'):
        plotter.open_gif(output_filename, fps=fps)
    else:
        plotter.open_movie(output_filename, fps=fps)

    # 3. Pre-read the first frame to establish bounding boxes and setup the camera view
    first_mesh_path = os.path.join(folder_path, stl_files[0])
    current_mesh = pv.read(first_mesh_path)
    
    # Add mesh to the plotter. We use a generic 'actor' label so we can swap the underlying data later.
    actor = plotter.add_mesh(
        current_mesh, 
        color='lightblue', 
        show_edges=True, 
        edge_color='navy',
        lighting=True
    )
    
    # Add an active text overlay to trace the iteration status dynamically
    iter_num = extract_iteration_number(stl_files[0])
    text_actor = plotter.add_text(f"Iteration: {iter_num}", color='black', font_size=14)

    # Position the camera cleanly looking down at the aircraft configuration
    plotter.camera_position = 'iso'  # Isometric view; options: 'xy', 'xz', 'yz', or custom coordinates
    plotter.camera.focal_point = current_mesh.center
    plotter.reset_camera()

    # 4. Loop through every STL file, update the geometry, and capture frames
    for index, filename in enumerate(stl_files):
        file_path = os.path.join(folder_path, filename)
        current_iter = extract_iteration_number(filename)
        print(f"[{index + 1}/{len(stl_files)}] Processing {filename} (Iteration {current_iter})...")

        # Read the new geometry step
        new_mesh = pv.read(file_path)
        
        # Smoothly update the current mesh points and face structures inside the canvas
        actor.mapper.dataset.copy_from(new_mesh)
        
        # Update the live progress text indicator
        text_actor.SetText(2, f"Iteration: {current_iter}")
        
        # Render the updated graphics and record it as a video frame
        plotter.write_frame()

    # 5. Clean up and save the completed video file
    plotter.close()
    print(f"\nSuccess! Animation successfully saved to: {os.path.abspath(output_filename)}")


if __name__ == "__main__":
    # Point this path to your optimization script's output directory
    TARGET_FOLDER = r"C:\Users\Maria\Documents\DSE\DSE\Final_analysis_optimization\intermediate_results_sensitivity" 
    
    # You can change the extension to ".mp4" if you prefer a standard movie clip
    OUTPUT_FILE = "wing_optimization_sensitivity.gif" 
    
    # Frames per second (higher number makes transitions faster)
    FRAMES_PER_SECOND = 10 

    create_optimization_animation(TARGET_FOLDER, OUTPUT_FILE, fps=FRAMES_PER_SECOND)