import os
import traceback
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from config import config
from dataloaders import get_preloaded_data_loaders
from train_eval_func import evaluate_func
from utils import filter_data, split_data
from train import setup, cleanup, get_model

import sys

def main(rank, world_size):
    try:
        setup(rank, world_size)

        data = filter_data()
        _, _, test_data = split_data(data)

        test_loader = get_preloaded_data_loaders(test_data, shuffle=False, rank=rank, world_size=world_size)

        model_type = config["model_config"]["model_type"]
        model = get_model(model_type).to(rank)
        model = DDP(model, device_ids=[rank])

        if rank == 0:
            model_path = config["paths"]["model_path"]
            state_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(rank))
            new_state_dict = {f'module.{k}' if not k.startswith('module.') else k: v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            print(f"Model loaded from {os.path.basename(model_path)}")

        model.eval()

        loss_func = nn.MSELoss()
        device = torch.device(f"cuda:{rank}")

        if rank == 0:
            test_loss, test_low, test_median, test_high = evaluate_func(model, test_loader, loss_func, device, rank)
            print(f"Testing completed. Loss: {test_loss:.4f}, Lowest Error: {test_low:.2f}°, Median Error: {test_median:.2f}°, Highest Error: {test_high:.2f}°")
    
    except KeyboardInterrupt:
        print(f"Interrupted by user, cleaning up process {rank}.")
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        print(f"Starting cleanup for rank {rank}")
        cleanup()
        print(f"Finished cleanup for rank {rank}")
        
        sys.exit(0)

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)
