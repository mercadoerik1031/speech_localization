import os
import optuna
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from snn import SNN  # Ensure this is correctly imported
from early_stopping import EarlyStopping
from config import config
from utils import filter_data, split_data
from dataloaders import get_preloaded_data_loaders
from train_eval_func import train_func, evaluate_func

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '10.182.0.2'
    os.environ['MASTER_PORT'] = '49152'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def objective(trial, rank, world_size):
    # Define hyperparameters
    dropout_rates = [trial.suggest_float(f'dropout_rate_{i}', 0.0, 0.75) for i in range(8)]
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    thresholds = [trial.suggest_float(f'threshold_{i}', 1.0, 30.0) for i in range(8)]
    l2 = trial.suggest_float('l2', 0, 1e-1)
    #features = trial.suggest_categorical('feature', ["log_mel_spectrogram", "hilbert_transform", "gcc_phat", "active_reactive_intensities"])
    #apply_spacial_rotation = trial.suggest_categorical('apply_spacial_rotation', [True, False])
    t_max = trial.suggest_int('t_max', 10, 20)
    
    # Update config with trial's suggestions
    config["model_config"]["dropout_rates"] = dropout_rates
    # config["training"]["learning_rate"] = learning_rate
    config["model_config"]["thresholds"] = thresholds
    config["training"]["l2_scheduler"]["init"] = l2
    #config["model_config"]["features"] = features
    #config["data_augmentation"]["apply_spherical_rotation"] = apply_spacial_rotation
    config["training"]["lr_scheduler"]["t_max"] = t_max

    batch_size = config["model_config"]["batch_size"]

    train_data, val_data, _ = split_data(filter_data())
    train_loader = get_preloaded_data_loaders(train_data, False, True, batch_size, rank, world_size)
    val_loader = get_preloaded_data_loaders(val_data, False, False, batch_size, rank, world_size)

    model = SNN().to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=l2)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=config["training"]["lr_scheduler"]["t_max"], eta_min=config["training"]["lr_scheduler"]["eta_min"])
    device = torch.device(f"cuda:{rank}")

    early_stopper = EarlyStopping(patience=6, verbose=True if rank == 0 else False, delta=0.1, path='best_model.pth')

    for epoch in range(config["training"]["num_epochs"]):
        train_results = train_func(model, train_loader, optimizer, nn.MSELoss(), device, rank)
        val_results = evaluate_func(model, val_loader, nn.MSELoss(), device, rank)
        
        train_loss, train_low, train_median, train_high = train_results
        val_loss, val_low, val_median, val_high = val_results
        
        # Report to Optuna and check if the trial should be pruned
        trial.report(val_median, epoch)
        early_stopper(val_median, model)

        if rank == 0:
            print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Median: {train_median} Val Median: {val_median:.2f}°")

        if early_stopper.early_stop:
            if rank == 0:
                print("Early stopping triggered.")
            raise optuna.exceptions.TrialPruned()

        lr_scheduler.step()

    return val_median if rank == 0 else float('inf')


def study(rank, world_size):
    try:
        setup(rank, world_size)
        if rank == 0:
            storage_url = "sqlite:///snn_study.db"
            study = optuna.create_study(direction="minimize", study_name="snn", storage=storage_url, load_if_exists=True)
            study.optimize(lambda trial: objective(trial, rank, world_size), n_trials=100)
            study._storage.commit()

            # Fetch and print the best trial's information
            best_trial = study.best_trial
            print(f"Best trial ID: {best_trial.number}")
            print(f"Best trial validation median error: {best_trial.value:.2f}°")
            print("Best trial parameters:")
            for key, value in best_trial.params.items():
                print(f"{key}: {value}")

    except KeyboardInterrupt:
        print(f"Optimization was interrupted manually on rank {rank}.")
    except Exception as e:
        print(f"An error occurred on rank {rank}: {e}")
    finally:
        print(f"Starting cleanup for rank {rank}.")
        cleanup()

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(study, args=(world_size,), nprocs=world_size, join=True)
