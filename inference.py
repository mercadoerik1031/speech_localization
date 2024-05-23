import os
import time
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np
import matplotlib.pyplot as plt

from config import config
from dataloaders import get_preloaded_data_loaders
from utils import filter_data, split_data
from mae import calc_angular_errors, to_cartesian
from train import get_model, setup, cleanup

def count_parameters(model):
    """Count the number of parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def run_inference(rank, world_size, model_type, output_dir):
    setup(rank, world_size)

    config["model_config"]["use_snn"] = model_type in ["SNN", "SRNN"]

    data = filter_data()
    _, _, test_data = split_data(data)

    test_loader = get_preloaded_data_loaders(test_data, apply_augmentations=False, shuffle=False, rank=rank, world_size=world_size)

    model = get_model(model_type).to(rank)
    model = DDP(model, device_ids=[rank])

    model_paths = {
        "CNN": "/home/erikmercado1031/pretrained_models/cnn_gcc_hilbert_20_46.pth",
        "CRNN": "/home/erikmercado1031/pretrained_models/crnn_gcc_hilbert_21_37.pth",
        "SNN": "/home/erikmercado1031/pretrained_models/snn_gcc_hilbert_32_66.pth",
        "SRNN": "/home/erikmercado1031/pretrained_models/srnn_gcc_hilbert_32_06.pth"
    }

    if rank == 0:
        model_path = model_paths[model_type]
        state_dict = torch.load(model_path, map_location=lambda storage, loc: storage.cuda(rank))
        new_state_dict = {f'module.{k}' if not k.startswith('module.') else k: v for k, v in state_dict.items()}
        model.load_state_dict(new_state_dict)
        print(f"Model {model_type} loaded from {os.path.basename(model_path)}")

    model.eval()
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        total_time = 0
        num_parameters = count_parameters(model)
        print(f"{model_type} number of parameters: {num_parameters}")
        results = {'outputs': [], 'labels': [], 'num_parameters': num_parameters}
        with torch.no_grad():
            for batch in test_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(device), labels.to(device)
                
                start_time = time.time()
                outputs = model(inputs)
                end_time = time.time()
                
                total_time += (end_time - start_time)
                results['outputs'].append(outputs.cpu().numpy())
                results['labels'].append(labels.cpu().numpy())

        avg_inference_time = total_time / len(test_loader)
        results['avg_inference_time'] = avg_inference_time
        print(f"Inference completed for {model_type}. Average inference time: {avg_inference_time:.4f} seconds per batch")

        # Save results to a temporary file
        torch.save(results, os.path.join(output_dir, f"{model_type}_results.pth"))

    cleanup()

def aggregate_results(output_dir, model_types):
    results = {}
    for model_type in model_types:
        model_results = torch.load(os.path.join(output_dir, f"{model_type}_results.pth"))
        results[model_type] = model_results
    return results

def plot_results(results_path, num_samples=10):
    results = torch.load(results_path)
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    # Colors for different models
    colors = {
        "CNN": "r",
        "CRNN": "g",
        "SNN": "b",
        "SRNN": "c"
    }

    # Calculate median errors and find samples closest to these median errors
    errors_dict = {}
    median_errors = {}
    avg_inference_times = {}
    num_parameters = {}
    for model_type, result in results.items():
        outputs = np.concatenate(result['outputs'], axis=0)
        labels = np.concatenate(result['labels'], axis=0)
        azimuth_gt, elevation_gt = labels[:, 0], labels[:, 1]
        azimuth_pred, elevation_pred = outputs[:, 0], outputs[:, 1]
        
        # Convert numpy arrays to torch tensors
        azimuth_gt_tensor = torch.tensor(azimuth_gt)
        elevation_gt_tensor = torch.tensor(elevation_gt)
        azimuth_pred_tensor = torch.tensor(azimuth_pred)
        elevation_pred_tensor = torch.tensor(elevation_pred)

        errors = calc_angular_errors(azimuth_gt_tensor, elevation_gt_tensor, azimuth_pred_tensor, elevation_pred_tensor)
        errors_dict[model_type] = np.array(errors).astype(float)  # Ensure errors are float type
        median_errors[model_type] = float(np.median(errors))
        avg_inference_times[model_type] = result['avg_inference_time']
        num_parameters[model_type] = result['num_parameters']
        print(f"{model_type} median error: {median_errors[model_type]}")
        print(f"{model_type} average inference time: {avg_inference_times[model_type]:.4f} seconds per batch")
        print(f"{model_type} number of parameters: {num_parameters[model_type]}")

    # Ensure that both errors and median_errors are float type
    for model_type in errors_dict.keys():
        errors_dict[model_type] = np.array(errors_dict[model_type]).astype(float)
        median_errors[model_type] = float(median_errors[model_type])

    # Find the samples where all models are within ±5 degrees of their median error
    common_samples = []
    for idx in range(len(errors_dict["CNN"])):
        within_threshold = all(
            np.abs(errors_dict[model_type][idx] - median_errors[model_type]) <= 5
            for model_type in model_types
        )
        if within_threshold:
            common_samples.append(idx)

    # Limit to the specified number of samples
    common_samples = common_samples[:num_samples]

    # Create plots for the samples where all models are within ±5 degrees of their median error
    for sample_idx in common_samples:
        fig = plt.figure(figsize=(14, 10))  # Increase figure size
        ax = fig.add_subplot(111, projection='3d')

        labels = np.concatenate(list(results.values())[0]['labels'], axis=0)
        ground_truth = labels[sample_idx]  # Get ground truth for the sample index

        azimuth_gt, elevation_gt = ground_truth
        x_gt, y_gt, z_gt = to_cartesian(torch.tensor(azimuth_gt), torch.tensor(elevation_gt))
        ax.scatter(x_gt, y_gt, z_gt, c='k', marker='o', s=200, label='Ground Truth')
        ax.scatter(0, 0, 0, c='m', marker='x', s=200, label='Reference Point (Center of Room)')

        for model_type in model_types:
            outputs = np.concatenate(results[model_type]['outputs'], axis=0)
            avg_inference_time = results[model_type]['avg_inference_time']
            azimuth_pred, elevation_pred = outputs[sample_idx]
            x_pred, y_pred, z_pred = to_cartesian(torch.tensor(azimuth_pred), torch.tensor(elevation_pred))
            error = calc_angular_errors(torch.tensor([azimuth_gt]), torch.tensor([elevation_gt]), torch.tensor([azimuth_pred]), torch.tensor([elevation_pred]))[0]

            ax.scatter(x_pred, y_pred, z_pred, c=colors[model_type], marker='^', s=200, label=f'{model_type} Prediction ({error:.2f}°) - Avg Inf Time: {avg_inference_time:.4f}s')

        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)

        # Adjust legend placement and size
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=17)  # Place legend further above the plot and increase font size

        # Adjust the top margin to make space for the legend and move the title down slightly
        plt.subplots_adjust(top=0.8)
        ax.set_title(f'Sample ID: {sample_idx}', y=1.25, fontsize=20)

        plt.savefig(os.path.join(output_dir, f'sample_{sample_idx}_results.png'))
        plt.close()







if __name__ == "__main__":
    output_dir = "/home/erikmercado1031/speech_localization/results"
    os.makedirs(output_dir, exist_ok=True)
    model_types = ["CNN", "CRNN", "SNN", "SRNN"]
    world_size = torch.cuda.device_count()

    results_path = os.path.join(output_dir, "all_results.pth")
    
    if not os.path.exists(results_path):
        from multiprocessing import Process

        processes = []
        for rank in range(world_size):
            for model_type in model_types:
                p = Process(target=run_inference, args=(rank, world_size, model_type, output_dir))
                p.start()
                processes.append(p)

        for p in processes:
            p.join()

        aggregated_results = aggregate_results(output_dir, model_types)
        torch.save(aggregated_results, results_path)
    else:
        print(f"Results file found at {results_path}. Skipping inference.\n")

    plot_results(results_path, num_samples=10)

