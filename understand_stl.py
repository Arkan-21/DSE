# This code studies structure of STL files and how to extract geometry info from them, using the stl library.
import os

import numpy as np
from stl import mesh
def load_stl(filename):
    """Load an STL file and return its vertices and faces."""
    stl_mesh = mesh.Mesh.from_file(filename)
    vertices = stl_mesh.vectors.reshape(-1, 3)  # Flatten to (N*3, 3)
    faces = np.arange(len(vertices)).reshape(-1, 3)  # Each face is a triplet of vertex indices
    return vertices, faces

def analyze_stl(filename):
    vertices, faces = load_stl(filename)
    print(f"Loaded STL file: {filename}")
    print(f"Number of vertices: {len(vertices)}")
    print(f"Number of faces: {len(faces)}")
    print(f"Vertex coordinates (first 5):\n{vertices[:5]}")
    print(f"Face indices (first 5):\n{faces[:5]}")

def plot_3D(vertices, faces, filename):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Create a list of vertex coordinates for each face
    face_vertices = vertices[faces]

    # Create a 3D polygon collection
    # poly_collection = Poly3DCollection(face_vertices, alpha=0.0)
    # poly_collection.set_facecolor('lightblue')
    # ax.add_collection3d(poly_collection)

    # find vertices inside the fuselage (assuming it's centered around x-axis and has a certain radius)
    fuselage_bounds = (10.5,50.0,2)  # Example bounds: x_min, x_max, radius
    inside_vertices, outside_vertices, bool_position = find_vertices_inside_fuselage(filename, fuselage_bounds)
    print(f"Number of vertices inside fuselage bounds: {len(inside_vertices)}")
    print(f"Number of vertices outside fuselage bounds: {len(outside_vertices)}")


    # Draw mesh boundary edges (wireframe)
    edges = []
    for tri in face_vertices:
        edges.append([tri[0], tri[1]])
        edges.append([tri[1], tri[2]])
        edges.append([tri[2], tri[0]])
    # edge_collection = Line3DCollection(edges, colors='black', linewidths=0.3, alpha=0.5)
    # ax.add_collection3d(edge_collection)

    #show all vertices as points (optional)
    #ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], color='red', s=10, label='Vertices')
    if len(inside_vertices) > 0:
        ax.scatter(inside_vertices[:, 0], inside_vertices[:, 1], inside_vertices[:, 2],
                   color='blue', s=20, label='Vertices inside fuselage')
    if len(outside_vertices) > 0:
        ax.scatter(outside_vertices[:, 0], outside_vertices[:, 1], outside_vertices[:, 2],
                   color='red', s=20, label='Vertices outside fuselage')
    # show
    # Auto scale to the mesh size
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(f"3D Visualization of STL Mesh: {filename}")
    plt.tight_layout()
    plt.show()

def show_structure(filename,fuselage_bounds=(10.5,53.0,2)):
    vertices, faces = load_stl(filename)
    mean_x = np.mean(vertices[:, 0])
    mean_y = np.mean(vertices[:, 1])
    mean_z = np.mean(vertices[:, 2])
    print(f"Mean vertex position: x={mean_x:.2f}, y={mean_y:.2f}, z={mean_z:.2f}")
    analyze_stl(filename)
    plot_3D(vertices, faces, filename)
    _,_, bool_position = find_faces_inside_fuselage(filename, fuselage_bounds)
    plot_faces(vertices, faces, bool_position)

def mesh_volume(filename):
    #If the file doesn't exist return -1000
    if not os.path.isfile(filename):
        print(f"File {filename} not found. Returning volume = -1000.")
        return -1000

    stl_mesh = mesh.Mesh.from_file(filename)
    # vectors shape: (N_faces, 3, 3)
    v0 = stl_mesh.vectors[:, 0, :]
    v1 = stl_mesh.vectors[:, 1, :]
    v2 = stl_mesh.vectors[:, 2, :]
    # Signed volume of tetrahedron for each face
    signed_vols = np.einsum('ij,ij->i', v0, np.cross(v1, v2)) / 6.0
    return abs(np.sum(signed_vols))

def find_vertices_inside_fuselage(stl_file, fuselage_bounds, x_offset=0.0, y_offset=0.0, z_offset=0.0):
    vertices, _ = load_stl(stl_file)
    x_min, x_max,   R = fuselage_bounds
    inside_vertices = []
    outside_vertices = []
    bool_position = [] # True if inside, False if outside
    for v in vertices:
        v = v - np.array([x_offset, y_offset, z_offset]) # move vertices to origin if needed
        R_v = np.sqrt(v[0]**2 + v[2]**2)  # Radial distance from x-axis
        if (x_min <= v[0] <= x_max and
            R_v < R):
            inside_vertices.append(v)
            bool_position.append(True)
        else:
            outside_vertices.append(v)
            bool_position.append(False)
    return np.array(inside_vertices), np.array(outside_vertices), np.array(bool_position)

def plot_faces(vertices, faces, bool_position):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')


    for i, face in enumerate(faces):
        tri_vertices = vertices[face]
        if bool_position[i]:
            color = 'blue' if bool_position[i] else 'red'
            poly_collection = Poly3DCollection([tri_vertices], alpha=0.5)
            poly_collection.set_facecolor(color)
            ax.add_collection3d(poly_collection)

    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Faces colored by position (inside fuselage = blue, outside = red)")
    plt.tight_layout()
    plt.show()

def find_faces_inside_fuselage(stl_file, fuselage_bounds, x_offset=0.0, y_offset=0.0, z_offset=0.0):
    vertices, faces = load_stl(stl_file)
    x_min, x_max,   R = fuselage_bounds
    inside_faces = []
    outside_faces = []
    bool_position = np.zeros(len(faces), dtype=bool) # True if inside, False if outside
    for i, face in enumerate(faces):
        v0, v1, v2 = vertices[face]
        v0 = v0 - np.array([x_offset, y_offset, z_offset])
        v1 = v1 - np.array([x_offset, y_offset, z_offset])
        v2 = v2 - np.array([x_offset, y_offset, z_offset])
        if all((x_min <= v[0] <= x_max and np.sqrt(v[1]**2 + v[2]**2) < R-1e-3) for v in [v0, v1, v2]):
            bool_position[i] = True
        else:          
            bool_position[i] = False
    
    inside_faces = faces[bool_position]
    outside_faces = faces[~bool_position]
    
    return np.array(inside_faces), np.array(outside_faces), np.array(bool_position)

def exclude_fuselage_faces(stl_file, fuselage_bounds, output_stl):
    vertices, faces = load_stl(stl_file)
    inside_faces, outside_faces, _ = find_faces_inside_fuselage(stl_file, fuselage_bounds)
    new_mesh = mesh.Mesh(np.zeros(len(outside_faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(outside_faces):
        new_mesh.vectors[i] = vertices[face]
    new_mesh.save(output_stl)


def stl_to_ascii(filename, output_filename=None):
    """
    Convert a binary or ASCII STL file to ASCII STL text format.

    Parameters
    ----------
    filename        : path to input STL file
    output_filename : if given, write ASCII STL to this path and return None;
                      otherwise return the ASCII content as a string
    """
    stl_mesh = mesh.Mesh.from_file(filename)

    def _lines():
        yield "solid"
        for i in range(len(stl_mesh.vectors)):
            v0, v1, v2 = stl_mesh.vectors[i]
            n = np.cross(v1 - v0, v2 - v0)
            length = np.linalg.norm(n)
            if length > 0:
                n = n / length
            yield f" facet normal  {n[0]:.10e} {n[1]:.10e} {n[2]:.10e}"
            yield "   outer loop"
            yield f"     vertex {v0[0]:.10e} {v0[1]:.10e} {v0[2]:.10e}"
            yield f"     vertex {v1[0]:.10e} {v1[1]:.10e} {v1[2]:.10e}"
            yield f"     vertex {v2[0]:.10e} {v2[1]:.10e} {v2[2]:.10e}"
            yield "   endloop"
            yield " endfacet"
        yield "endsolid"

    if output_filename:
        with open(output_filename, "w") as f:
            for line in _lines():
                f.write(line + "\n")
        return None
    return "\n".join(_lines()) + "\n"



if __name__ == "__main__":
    file = "baseline_geometry.stl"  # Replace with your STL file path
    exclude_fuselage_faces(file, fuselage_bounds=(0.0, 60.5, 2.0), output_stl="wings_only.stl")
    show_structure("wings_only.stl")
    stl_to_ascii("wings_only.stl", output_filename="wings_only_ascii.stl")
    input("Press Enter to continue...")
    volume = mesh_volume(file)
    print(f"Estimated volume of the mesh: {volume:.3f} cubic units")
    fuselage = "fuselage_only.stl"
    whole_plane ="Love_u_Simran__and_Alex.stl"
    fuselage_volume = mesh_volume(fuselage)
    whole_volume = mesh_volume(whole_plane)
    print(f"Estimated volume of the fuselage: {fuselage_volume:.3f} cubic units")
    print(f"Estimated volume of the whole plane: {whole_volume:.3f} cubic units")
    print(f"Estimated volume of the wings (whole - fuselage): {whole_volume - fuselage_volume:.3f} cubic units")