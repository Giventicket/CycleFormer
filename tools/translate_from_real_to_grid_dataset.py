import numpy as np
import torch

class TSPImageDataset(torch.utils.data.Dataset):
  def __init__(self, data_file):
    self.data_file = data_file
    
    self.file_lines = open(data_file).read().splitlines()
    print(f'Loaded "{data_file}" with {len(self.file_lines)} lines')

  def __len__(self):
    return len(self.file_lines)

  def rasterize(self, idx):
    # Select sample
    line = self.file_lines[idx]
    # Clear leading/trailing characters
    line = line.strip()

    # Extract points
    points = line.split(' output ')[0]
    points = points.split(' ')
    points = np.array([[float(points[i]), float(points[i + 1])] for i in range(0, len(points), 2)])
    # Extract tour
    tour = line.split(' output ')[1]
    tour = tour.split(' ')
    tour = np.array([int(t) for t in tour])
    
    return points, tour

  def __getitem__(self, idx):
    points, tour = self.rasterize(idx)
    return points, tour

data_path = "./tsp50-50_concorde(val).txt"

images = TSPImageDataset(
    data_file = data_path, 
)

def point2grid(nodes_coord, grid_size = 100):
    nodes_coord_scaled = (nodes_coord * grid_size).astype(int)
    grid_indices = nodes_coord_scaled[:, 0] + nodes_coord_scaled[:, 1] * grid_size
    return grid_indices

with open("./tsp50_grid100_val.txt", "w") as file:
    for image in images:
        points, tour = image
        tour_points = points[tour - 1]
        
        points_indices = point2grid(points)
        tour_indices = point2grid(tour_points)
        
        for idx in points_indices:
            file.write(str(idx))
            file.write(" ")
            
        file.write("output ")
        
        for idx in tour_indices:
            file.write(str(idx))
            file.write(" ")
        file.write("\n")
