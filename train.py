import os
import traceback
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import config
from models.snn import SNN
from models.srnn import SRNN
from models.cnn import CNN
from models.crnn import CRNN
from dataloaders import get_preloaded_data_loaders
from train_eval_func import train_func, evaluate_func
from early_stopping import EarlyStopping
from utils import filter_data, split_data

import sys

def setup(rank, world_size):
    """Set up the environment for distributed training."""
    os.environ['MASTER_ADDR'] = config["env"]["ip"]  # Master IP Address
    os.environ['MASTER_PORT'] = config["env"]["port"]  # Master Port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed training environment."""
    dist.destroy_process_group()

def get_model(model_type):
    if model_type == "SNN":
        return SNN()
    elif model_type == "SRNN":
        return SRNN()
    elif model_type == "CNN":
        return CNN()
    elif model_type == "CRNN":
        return CRNN()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def main(rank, world_size):
    setup(rank, world_size)

    try:
        print(f"Setup complete for rank {rank}")

        data = filter_data()
        if not data:
            print("No data returned from filter_data.")
            return

        train_data, val_data, _ = split_data(data)
        if not train_data or not val_data:
            print("Training or validation data sets are empty.")
            return

        train_loader = get_preloaded_data_loaders(train_data, shuffle=True, rank=rank, world_size=world_size)
        val_loader = get_preloaded_data_loaders(val_data, shuffle=False, rank=rank, world_size=world_size)

        model_type = config["model_config"]["model_type"]
        model = get_model(model_type).to(rank)
        model = DDP(model, device_ids=[rank])

        if not model:
            print("Failed to initialize the model.")
            return

        print(f"Using {model_type} model")

        if rank == 0:
            model_path = config["paths"]["model_path"]
            if os.path.exists(model_path):
                state_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(rank))
                new_state_dict = {f'module.{k}' if not k.startswith('module.') else k: v for k, v in state_dict.items()}
                model.load_state_dict(new_state_dict)
                print(f"Loaded {os.path.basename(model_path)} for further training\n")
            else:
                print(f"{model_path} Not Found... Restarting Training...\n")
                
        device = torch.device(f"cuda:{rank}")
        loss_func = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), 
                                     lr=config["training"]["learning_rate"], 
                                     #weight_decay=config["training"]["l2"]
                                     )
        
        lr_scheduler = CosineAnnealingLR(optimizer, 
                                         T_max=config["training"]["lr_scheduler"]["t_max"], 
                                         eta_min=config["training"]["lr_scheduler"]["eta_min"])
        
        early_stopping = EarlyStopping(patience=config["training"]["early_stopping"]["patience"], 
                                       verbose=True if rank == 0 else False, 
                                       delta=config["training"]["early_stopping"]["delta"])    
        
        for epoch in range(config["training"]["num_epochs"]):
            if rank == 0:
                print(f'Epoch {epoch+1}/{config["training"]["num_epochs"]}')
                print('-' * 10)

            results = train_func(model, train_loader, optimizer, loss_func, device, rank)
            if rank == 0:
                train_loss, train_low, train_median, train_high = results
                print(f"Train Loss: {train_loss:.4f}, Low: {train_low:.2f}°, Median: {train_median:.2f}°, High: {train_high:.2f}°")

            results = evaluate_func(model, val_loader, loss_func, device, rank)
            if rank == 0:
                val_loss, val_low, val_median, val_high = results
                print(f"Val Loss: {val_loss:.4f}, Low: {val_low:.2f}°, Median: {val_median:.2f}°, High: {val_high:.2f}°\n")

                early_stopping(val_median, model.module if isinstance(model, DDP) else model)
                if early_stopping.early_stop:
                    print("Early stopping\n")
                    break
                
                print(f"Current LR: {optimizer.param_groups[0]['lr']:.7f}\n")
                
                torch.cuda.empty_cache()  # Clear unused memory

            lr_scheduler.step()

    except KeyboardInterrupt:
        print(f"Interrupted by user, cleaning up process {rank}.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
    finally:
        print(f"Started cleanup for rank {rank}")
        cleanup()
        print(f"Finished cleanup for rank {rank}")
        sys.exit(0)

if __name__ == '__main__':
    world_size = torch.cuda.device_count()  # Number of available GPUs
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
