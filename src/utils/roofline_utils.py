import numpy as np
import matplotlib.pyplot as plt

def plot_points_from_results(results, color=None):
    """
    Plots each entry in 'results' as a point on the Roofline chart:
      - X-axis: arithmetic_intensity
      - Y-axis: performance in TFLOP/s
    If 'color' is provided, that color is used for all points in the dataset.
    Otherwise, a colormap is used to assign a different color to each point.
    """
    if color is not None:
        for entry in results:
            x = entry['arithmetic_intensity']
            y = (entry['total_flops'] / entry['avg_time_seconds']) / 1e12
            label = f"WS={entry['world_size']},B={entry['batch_size']},D={entry['D']},F={entry['F']},L={entry['num_layers']}"
            plt.scatter(x, y, s=80, color=color)
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(5, -5),
                textcoords="offset points",
                fontsize=8
            )
    else:
        # Use a colormap to assign different colors to each point
        cmap = plt.get_cmap('tab10')
        for i, entry in enumerate(results):
            x = entry['arithmetic_intensity']
            y = (entry['total_flops'] / entry['avg_time_seconds']) / 1e12
            label = f"WS={entry['world_size']},B={entry['batch_size']},D={entry['D']},F={entry['F']},L={entry['num_layers']}"
            plt.scatter(x, y, s=80, color=cmap(i % 10))
            plt.annotate(
                label,
                xy=(x, y),
                xytext=(5, -5),
                textcoords="offset points",
                fontsize=8
            )

def plot_roofline_chart(gpu_info, gemm_results=None):
    """
    Constructs a Roofline chart using GPU specs (peak FLOPS & memory bandwidth) 
    and optionally overlays GEMM benchmark results.
    
    If gemm_results is a single dataset (list of dicts), each point is plotted with a different color.
    If gemm_results is a list of datasets (each a list of dicts), then each dataset is plotted in a fixed color.
    """
    # Peak theoretical performance of the GPU (tensor cores)
    peak_flops_tflops_single = 242 / 2
    peak_flops_tflops = peak_flops_tflops_single * len(gpu_info) # Adjust as necessary for your hardware
    
    # Memory bandwidth in TB/s (taken from gpu_info)
    peak_bandwidth_tb_s = gpu_info[0]['memory_bandwidth_tb_s'] * len(gpu_info)
    
    # Generate a range for arithmetic intensity
    arithmetic_intensity = np.linspace(0.1, 1000, num=1000)
    
    # Compute the compute and memory performance limits
    compute_limit = np.full_like(arithmetic_intensity, peak_flops_tflops)
    compute_limit_single = np.full_like(arithmetic_intensity, peak_flops_tflops_single)
    memory_limit = arithmetic_intensity * peak_bandwidth_tb_s
    memory_limit_single = arithmetic_intensity * peak_flops_tflops_single
    performance = np.minimum(memory_limit, compute_limit)
    performance_single = np.minimum(memory_limit, compute_limit_single)

    plt.figure(figsize=(10, 6))
    plt.plot(arithmetic_intensity, performance, label='Roofline Performance', linewidth=2, zorder=10)
    plt.axhline(
        y=peak_flops_tflops,
        linestyle='--',
        color="red",
        label=f'Compute Bound: {peak_flops_tflops} TFLOPS'
    )

    plt.plot(arithmetic_intensity, performance_single, label='Roofline Performance 1 GPU', linewidth=2, zorder=10)
    plt.axhline(
        y=peak_flops_tflops_single,
        linestyle='--',
        color="red",
        label=f'Compute Bound Single GPU: {peak_flops_tflops_single} TFLOPS'
    )
    
    # Compute-memory boundary
    cross_point = peak_flops_tflops / peak_bandwidth_tb_s
    plt.axvline(
        x=cross_point,
        linestyle='--',
        color="green",
        label=f'Compute/Memory Boundary: {cross_point:.2f} FLOPs/Byte'
    )

    # Plot GEMM results if available
    if gemm_results:
        # Check if gemm_results is a single dataset (list of dicts) or multiple datasets (list of lists)
        if isinstance(gemm_results, list) and gemm_results and isinstance(gemm_results[0], dict):
            # Single dataset: use different colors for each point.
            plot_points_from_results(gemm_results, color=None)
        elif isinstance(gemm_results, list) and gemm_results and isinstance(gemm_results[0], list):
            # Multiple datasets: assign fixed colors per dataset.
            fixed_colors = ['blue', 'orange', "red", "purple", "green", "gray", "pink"]  # Extend if more than 2 datasets are needed.
            for i, dataset in enumerate(gemm_results):
                plot_points_from_results(dataset, color=fixed_colors[i % len(fixed_colors)])
    
    plt.xlabel('Arithmetic Intensity (FLOPs/Byte)')
    plt.ylabel('Performance (TFLOP/s)')
    plt.title(f'Realistic Roofline Model for {len(gpu_info)} X {gpu_info[0]["name"]} GPU(s)')
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()

    # Adjust x/y limits for better visibility
    plt.xlim(0, cross_point * 2)
    plt.ylim(0, peak_flops_tflops * 1.5)

    plt.show()
