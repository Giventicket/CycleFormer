# python3 make_tsp_dataset.py --min_nodes 50 --max_nodes 50 --num_samples 1502000 --batch_size 100 --filename "tsp50_train_concorde.txt" --seed 1234
# python3 make_tsp_dataset.py --min_nodes 50 --max_nodes 50 --num_samples 10000 --batch_size 100 --filename "tsp50_test_concorde.txt" --seed 4321
# python3 make_tsp_dataset.py --min_nodes 100 --max_nodes 100 --num_samples 1502000 --batch_size 100 --filename "tsp100_train_concorde.txt" --seed 1234
# python3 make_tsp_dataset.py --min_nodes 100 --max_nodes 100 --num_samples 10000 --batch_size 100 --filename "tsp100_test_concorde.txt" --seed 4321
# python3 make_tsp_dataset.py --min_nodes 500 --max_nodes 500 --num_samples 128000 --batch_size 100 --filename "tsp500_train_concorde.txt" --seed 1234
# python3 make_tsp_dataset.py --min_nodes 500 --max_nodes 500 --num_samples 128 --batch_size 64 --filename "tsp500_test_concorde.txt" --seed 4321
# python3 make_tsp_dataset.py --min_nodes 1000 --max_nodes 1000 --num_samples 128 --batch_size 64 --filename "tsp1000_test_concorde.txt" --seed 4321
# python3 make_tsp_dataset.py --min_nodes 10000 --max_nodes 10000 --num_samples 6400 --batch_size 100 --filename "tsp10000_train_concorde.txt" --seed 1234
# python3 make_tsp_dataset.py --min_nodes 10000 --max_nodes 10000 --num_samples 16 --batch_size 16 --filename "tsp10000_test_concorde.txt" --seed 4321
python3 make_tsp_dataset.py --min_nodes 1000 --max_nodes 1000 --num_samples 64000 --batch_size 100 --filename "tsp1000_train_lkh3.txt" --seed 1234 --solver lkh
python3 make_tsp_dataset.py --min_nodes 10000 --max_nodes 10000 --num_samples 6400 --batch_size 100 --filename "tsp10000_train_lkh3.txt" --seed 1234 --solver lkh
