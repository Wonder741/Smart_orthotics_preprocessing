import numpy as np
from stl import mesh
import matplotlib.pyplot as plt


# Load STL file
your_stl_file = 'Scanning_R.stl'
mesh_data = mesh.Mesh.from_file(your_stl_file)

def stl_properties(mesh_data):
    # Load the STL mesh    
    # Number of triangles
    num_triangles = len(mesh_data.vectors)
    
    # Bounding Box
    min_coords = mesh_data.v0.min(axis=0)
    max_coords = mesh_data.v0.max(axis=0)
    for v in [mesh_data.v1, mesh_data.v2]:
        min_coords = np.minimum(min_coords, v.min(axis=0))
        max_coords = np.maximum(max_coords, v.max(axis=0))
    bounding_box = (min_coords, max_coords)
    
    # Volume (requires the mesh to be watertight/closed)
    volume = mesh_data.get_mass_properties()[0]
    
    # Surface Area
    areas = mesh_data.areas
    surface_area = np.sum(areas)
    
    # Centroid
    centroid = mesh_data.get_mass_properties()[1]
    
    return {
        "Number of Triangles": num_triangles,
        "Bounding Box": bounding_box,
        "Volume": volume,
        "Surface Area": surface_area,
        "Centroid": centroid
    }



def sample_points_from_triangle(triangle, num_points,):
    """Sample points from a triangle using barycentric coordinates."""
    points = []
    for _ in range(num_points):
        s, t = sorted([np.random.rand(), np.random.rand()])
        f = lambda i: s * triangle[0][i] + (t-s) * triangle[1][i] + (1-t) * triangle[2][i]
        points.append([f(0), f(1), f(2)])
    return points

def stl_to_point_cloud(mesh_data, output_file_path, points_per_unit_area=100):
    # Load the STL mesh
    
    # Compute the area of each triangle and determine number of points to sample from each
    areas = mesh_data.areas
    total_points = (areas * points_per_unit_area).astype(int)
    
    # Sample points from each triangle
    all_points = []
    for triangle, num_points in zip(mesh_data.vectors, total_points):
        all_points.extend(sample_points_from_triangle(triangle, int(num_points)))
    
    # Save the points to the output file
    with open(output_file_path, 'w') as out_file:
        for point in all_points:
            out_file.write(f"{point[0]} {point[1]} {point[2]}\n")
    
    print(f"Saved point cloud with {len(all_points)} points to {output_file_path}")


def point_cloud_to_grayscale_image(point_cloud_file_path, resolution=1.0):
    # Load the point cloud
    points = np.loadtxt(point_cloud_file_path)
    x, y = points[:, 0], points[:, 1]

    # Determine bounds for the histogram based on point cloud extents
    x_min, y_min = np.min(x), np.min(y)
    x_max, y_max = np.max(x), np.max(y)
    
    # Number of bins based on resolution
    x_bins = int((x_max - x_min) / resolution)
    y_bins = int((y_max - y_min) / resolution)

    # Create a 2D histogram of the projected points
    histogram, _, _ = np.histogram2d(x, y, bins=(x_bins, y_bins), range=[[x_min, x_max], [y_min, y_max]])

    # Normalize to the range [0, 255]
    normalized = np.interp(histogram, (histogram.min(), histogram.max()), (0, 255)) 
    # Change values that are 0 (not projected) to 255
    normalized[normalized != 0] *= 1.2
    normalized[normalized > 255] = 255
    normalized[normalized == 0] = 255
    
    # Save the normalized object to a CSV file
    np.savetxt("output1.csv", normalized, delimiter=",")
    plt.imsave("output1.png", normalized.T, cmap='viridis', origin='lower')

    # Display the grayscale image
    plt.imshow(normalized.T, cmap='viridis', origin='lower', extent=[x_min, x_max, y_min, y_max])
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Grayscale Image from Point Cloud')
    plt.show()


properties = stl_properties(mesh_data)
for key, value in properties.items():
    print(f"{key}: {value}")

output_path = 'output_point_cloud.txt'
stl_to_point_cloud(mesh_data, output_path)

# Example usage
point_cloud_to_grayscale_image(output_path, resolution=1.0)