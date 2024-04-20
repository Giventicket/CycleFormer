docker run -it --gpus all --ipc host -v /home/jpseo99/Geoformer:/home/Geoformer 9ef37be4ff59
# nvidia/cuda   12.3.1-devel-ubuntu20.04   9ef37be4ff59   2 weeks ago   7.03GB
python3 eval.py val_data_path=./trained_model_and_data/data/tsp100_test_concorde.txt load_checkpoint_path=./outputs/TSP-N100G100-Erevmha-Tnone-1230T1419/checkpoint.ckpt encoder_use_position_encoding=false encoder_position_encoding_type=none
