# https://github.com/Edward-Sun/DIFUSCO/blob/main/data/generate_tsp_data.py

import argparse
import pprint as pp
import time
import warnings
from multiprocessing import Pool
import numpy as np
import tqdm
import os
import glob

from concorde.tsp import TSPSolver  # https://github.com/jvkersch/pyconcorde

# pip install 'pyconcorde @ git+https://github.com/jvkersch/pyconcorde'

warnings.filterwarnings("ignore")

def return_grid_coord(nodes_coord, grid_size):
    grid_coord = ((nodes_coord * grid_size).astype(int) / grid_size) + 1 / (2 * grid_size)
    return grid_coord

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_nodes", type=int, default=50)
    parser.add_argument("--max_nodes", type=int, default=50)
    parser.add_argument("--num_samples", type=int, default=100000)
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--filename", type=str, default=None)
    parser.add_argument("--solver", type=str, default="concorde")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid_size", type=int, default=100)
    opts = parser.parse_args()

    assert opts.num_samples % opts.batch_size == 0, "Number of samples must be divisible by batch size"

    np.random.seed(opts.seed)

    if opts.filename is None:
        opts.filename = f"tsp{opts.min_nodes}-{opts.max_nodes}_concorde.txt"
        # opts.filename = f"tsp{opts.min_nodes}-{opts.max_nodes}_concorde(val).txt"

    # Pretty print the run args
    pp.pprint(vars(opts))

    with open(opts.filename, "w") as f:
        start_time = time.time()
        for b_idx in tqdm.tqdm(range(opts.num_samples // opts.batch_size)):
            num_nodes = np.random.randint(low=opts.min_nodes, high=opts.max_nodes + 1)
            assert opts.min_nodes <= num_nodes <= opts.max_nodes

            batch_nodes_coord = []
            for _ in range(opts.batch_size):
                nodes_coord = np.random.random([num_nodes, 2])
                grid_coord = return_grid_coord(nodes_coord, opts.grid_size)
                batch_nodes_coord.append(grid_coord)
            batch_nodes_coord = np.array(batch_nodes_coord)    

            def solve_tsp(nodes_coord):
                if opts.solver == "concorde":
                    scale = 1e6
                    solver = TSPSolver.from_data(nodes_coord[:, 0] * scale, nodes_coord[:, 1] * scale, norm="EUC_2D")
                    solution = solver.solve(verbose=False)
                    tour = solution.tour
                else:
                    raise ValueError(f"Unknown solver: {opts.solver}")

                return tour

            with Pool(opts.batch_size) as p:
                tours = p.map(solve_tsp, [batch_nodes_coord[idx] for idx in range(opts.batch_size)])

            for idx, tour in enumerate(tours):
                if (np.sort(tour) == np.arange(num_nodes)).all():
                    f.write(" ".join(str(x) + str(" ") + str(y) for x, y in batch_nodes_coord[idx]))
                    f.write(str(" ") + str("output") + str(" "))
                    f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
                    f.write(str(" ") + str(tour[0] + 1) + str(" "))
                    f.write("\n")

        end_time = time.time() - start_time

        assert b_idx == opts.num_samples // opts.batch_size - 1

    print(f"Completed generation of {opts.num_samples} samples of TSP{opts.min_nodes}-{opts.max_nodes}.")
    print(f"Total time: {end_time / 60:.1f}m")
    print(f"Average time: {end_time / opts.num_samples:.1f}s")

    res_files = glob.glob(os.path.join("./", "*.res"))
    for res_file in res_files:
        os.remove(res_file)

    res_files = glob.glob(os.path.join("./", "*.pul"))
    for res_file in res_files:
        os.remove(res_file)

    res_files = glob.glob(os.path.join("./", "*.sav"))
    for res_file in res_files:
        os.remove(res_file)

    res_files = glob.glob(os.path.join("./", "*.sol"))
    for res_file in res_files:
        os.remove(res_file)
