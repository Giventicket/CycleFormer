import os

def delete_files_except(keep_files):
    current_files = os.listdir('.')
    for file_name in current_files:
        if file_name not in keep_files:
            try:
                os.remove(file_name)
                print(f"Deleted: {file_name}")
            except FileNotFoundError:
                print(f"File not found: {file_name}")
            except PermissionError:
                print(f"Permission denied: {file_name}")

if __name__ == "__main__":
    keep_files = [
        "make_tsp_dataset.py",
        "run.sh",
        "tsp50_test_concorde.txt",
        "tsp50_train_concorde.txt",
        "tsp100_test_concorde.txt",
        "tsp100_train_concorde.txt",
        "tsp500_test_concorde.txt",
        "tsp500_train_concorde.txt",
        "tsp1000_test_concorde.txt",
        "delete_files.py"
    ]
    delete_files_except(keep_files)
