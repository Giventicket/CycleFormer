import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from TSPmodel import TSPModel

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="Seed for random number generation.")
    parser.add_argument("--train_data_path", default="./tsp50_train_concorde.txt", help="Path to the training data.")
    parser.add_argument("--val_data_path", default="./tsp50_test_concorde.txt", help="Path to the validation data.")
    parser.add_argument("--node_size", type=int, default=50, help="Node size for TSP instances")
    parser.add_argument("--train_batch_size", type=int, default=80, help="Batch size for training. [tsp50, tsp100, tsp500, tsp1000]: 80")
    parser.add_argument("--val_batch_size", type=int, default=1280, help="Batch size for validation. [tsp50, tsp100]: 1280, [tsp500, tsp1000]: 128")
    parser.add_argument("--test_batch_size", type=int, default=1280, help="Batch size for testing.")
    parser.add_argument("--G", type=int, default=1, help="The number of multistart nodes.")
    parser.add_argument("--resume_checkpoint", default=None, help="Path to the checkpoint for resuming training.")
    parser.add_argument("--gpus", type=int, nargs="+", default=[3], help="List of GPU indices to use.")
    parser.add_argument("--max_epochs", type=int, default=100, help="Maximum number of epochs for training.")
    parser.add_argument("--enc_num_layers", type=int, default=6, help="Number of layers in the encoder.")
    parser.add_argument("--dec_num_layers", type=int, default=6, help="Number of layers in the decoder.")
    parser.add_argument("--d_model", type=int, default=128, help="Dimension of model.")
    parser.add_argument("--d_ff", type=int, default=512, help="Dimension of feedforward networks.")
    parser.add_argument("--h", type=int, default=8, help="Number of attention heads.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout probability.")
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing factor.")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning rate.")
    parser.add_argument("--betas", type=float, nargs=2, default=[0.9, 0.98], help="AdamW optimizer betas.")
    parser.add_argument("--eps", type=float, default=1e-9, help="Adam optimizer epsilon.")
    parser.add_argument("--factor", type=float, default=1.0, help="Factor for learning rate schedule.")
    parser.add_argument("--warmup", type=int, default=400, help="Warmup steps for learning rate schedule.")
    parser.add_argument("--encoder_pe", default="2D", help="Type of positional encoding for encoder. [None, 2D]")
    parser.add_argument("--decoder_pe", default="circular PE", help="Type of positional encoding for decoder. [None, 1D, circular]")
    parser.add_argument("--decoder_lut", default="memory", help="Type of lookup table for decoder. [shared, unshared, memory]")
    parser.add_argument("--comparison_matrix", default="memory", help="Type of comparison matrix. [encoder_lut, decoder_lut, memory]")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    cfg = parse_arguments()

    pl.seed_everything(cfg.seed)
    tsp_model = TSPModel(cfg)

    checkpoint_callback = ModelCheckpoint(
        monitor = "opt_gap",
        filename = f'TSP{cfg.node_size}-' + "{epoch:02d}-{opt_gap:.4f}",
        save_top_k=3,
        mode="min",
        every_n_epochs=1,
    )

    loggers = []
    tb_logger = TensorBoardLogger("logs")
    
    trainer = pl.Trainer(
        default_root_dir="./",
        devices=cfg.gpus,
        accelerator="cuda",
        precision="16-mixed",
        max_epochs=cfg.max_epochs,
        reload_dataloaders_every_n_epochs=0,
        num_sanity_val_steps=0,
        logger=[tb_logger],
        callbacks=[checkpoint_callback],
        strategy="ddp_find_unused_parameters_true",
    )
    trainer.fit(tsp_model)
    
