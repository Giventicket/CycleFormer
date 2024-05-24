# CycleFormer: TSP Solver Based on Transformer Models

We introduce CycleFormer, a novel transformer model tailored specifically for the Traveling Salesman Problem (TSP). Unlike conventional transformer models designed for natural language processing, TSP presents unique challenges due to its dynamic and unlimited node set. To address this, we have innovatively integrated key elements into CycleFormer to harness the full potential of transformers for TSP.

CycleFormer enhances the encoder-decoder interaction by directly connecting the encoder output with the decoder linear layer, thereby facilitating seamless information flow. Additionally, we have introduced positional encodings tailored to TSP's two-dimensional nature. The encoder tokens are equipped with positional encodings reflecting spatial coordinates, while circular positional encodings are employed for the decoder tokens to capture the cyclic properties of a tour.

Our experimental results demonstrate the superiority of CycleFormer over state-of-the-art transformer models across various TSP problem sizes, ranging from TSP-50 to TSP-500. Notably, on TSP-500, CycleFormer achieves a remarkable reduction in the optimality gap, from 3.09% to 1.10%, outperforming existing state-of-the-art solutions.

### Model Architecture
![CycleFormer Architecture](https://github.com/Giventicket/CycleFormer/assets/39179946/afd33960-7937-4b3e-9912-440db50a439e)

### Random TSP Results
Results on large-scale Random TSP.

*(Results marked with an asterisk (*) are directly sourced from Sun and Yang \cite{Sun2023}. DACT \cite{Ma2021}, incorporating extra heuristics like 2-opt, was excluded from the comparison.)*
![Random TSP Results](https://github.com/Giventicket/CycleFormer/assets/39179946/2069e2a0-8c37-4744-8304-981112704718)

### Environment
- CUDA 12.1
- Python 3.10.12
- Torch 2.2.0+cu121

### Pretrained Checkpoints
*(To be uploaded)*

### Train Command
```bash
python3 train.py \
    --seed 1 \
    --train_data_path ./tsp{node_size}_train_concorde.txt \
    --val_data_path ./tsp{node_size}_test_concorde.txt \
    --node_size {node_size} \
    --train_batch_size 80 \
    --val_batch_size 1280 \
    --test_batch_size 1280 \
    --G 1 \
    --resume_checkpoint None \
    --gpus 3 \
    --max_epochs 100 \
    --enc_num_layers 6 \
    --dec_num_layers 6 \
    --d_model 128 \
    --d_ff 512 \
    --h 8 \
    --dropout 0.1 \
    --smoothing 0.1 \
    --lr 0.5 \
    --betas 0.9 0.98 \
    --eps 1e-9 \
    --factor 1.0 \
    --warmup 400 \
    --encoder_pe 2D \
    --decoder_pe "circular PE" \
    --decoder_lut memory \
    --comparison_matrix memory
```

### Inference Command
Greedy decoding
```bash
python inference.py \
    --seed 1 \
    --test_data_path ./tsp{node_size}_test_concorde.txt \
    --test_batch_size 1 \
    --G 1 \ 
    --checkpoint_path ./Cycleformer_{node_size}.pth \
    --gpus 0
```

Multi_start decoding
```bash
python inference.py \
    --seed 1 \
    --test_data_path ./tsp{node_size}_test_concorde.txt \
    --test_batch_size 1 \
    --G {node_size} \ 
    --checkpoint_path ./Cycleformer_{node_size}.pth \
    --gpus 0
```

### Build and Run Docker Container
```bash
docker build -t cycleformer-image .
docker run -it --gpus all --ipc host cycleformer-image
```
