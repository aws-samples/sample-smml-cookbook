import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import random
import json

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
            x = entry["arithmetic_intensity"]
            y = (entry["total_flops"] / entry["avg_time_seconds"]) / 1e12
            label = f"WS={entry.get('world_size')},B={entry.get('batch_size')},D={entry.get('D')},F={entry.get('F')},L={entry.get('num_layers')}"
            plt.scatter(x, y, s=80, color=color)
            plt.annotate(
                label, xy=(x, y), xytext=(5, -5), textcoords="offset points", fontsize=8
            )
    else:
        # Use a colormap to assign different colors to each point
        cmap = plt.get_cmap("tab10")
        for i, entry in enumerate(results):
            x = entry["arithmetic_intensity"]
            y = (entry["total_flops"] / entry["avg_time_seconds"]) / 1e12
            label = f"WS={entry.get('world_size')},B={entry.get('batch_size')},D={entry.get('D')},F={entry.get('F')},L={entry.get('num_layers')}"
            plt.scatter(x, y, s=80, color=cmap(i % 10))
            plt.annotate(
                label, xy=(x, y), xytext=(5, -5), textcoords="offset points", fontsize=8
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
    peak_flops_tflops = peak_flops_tflops_single * len(
        gpu_info
    )  # Adjust as necessary for your hardware

    # Memory bandwidth in TB/s (taken from gpu_info)
    peak_bandwidth_tb_s = gpu_info[0]["memory_bandwidth_tb_s"] * len(gpu_info)

    # Generate a range for arithmetic intensity
    arithmetic_intensity = np.linspace(0.1, 1000, num=1000)

    # Compute the compute and memory performance limits
    compute_limit = np.full_like(arithmetic_intensity, peak_flops_tflops)
    compute_limit_single = np.full_like(arithmetic_intensity, peak_flops_tflops_single)
    memory_limit = arithmetic_intensity * peak_bandwidth_tb_s
    memory_limit_single = arithmetic_intensity * gpu_info[0]["memory_bandwidth_tb_s"]
    performance = np.minimum(memory_limit, compute_limit)
    performance_single = np.minimum(memory_limit_single, compute_limit_single)

    plt.figure(figsize=(10, 6))
    plt.plot(
        arithmetic_intensity,
        performance,
        label="Roofline Performance",
        linewidth=2,
        zorder=10,
    )
    plt.axhline(
        y=peak_flops_tflops,
        linestyle="--",
        color="red",
        label=f"Compute Bound: {peak_flops_tflops} TFLOPS",
    )

    plt.plot(
        arithmetic_intensity,
        performance_single,
        label="Roofline Performance 1 GPU",
        linewidth=2,
        zorder=10,
    )
    plt.axhline(
        y=peak_flops_tflops_single,
        linestyle="--",
        color="red",
        label=f"Compute Bound Single GPU: {peak_flops_tflops_single} TFLOPS",
    )

    # Compute-memory boundary
    cross_point = peak_flops_tflops / peak_bandwidth_tb_s
    plt.axvline(
        x=cross_point,
        linestyle="--",
        color="green",
        label=f"Compute/Memory Boundary: {cross_point:.2f} FLOPs/Byte",
    )

    # Plot GEMM results if available
    if gemm_results:
        # Check if gemm_results is a single dataset (list of dicts) or multiple datasets (list of lists)
        if (
            isinstance(gemm_results, list)
            and gemm_results
            and isinstance(gemm_results[0], dict)
        ):
            # Single dataset: use different colors for each point.
            plot_points_from_results(gemm_results, color=None)
        elif (
            isinstance(gemm_results, list)
            and gemm_results
            and isinstance(gemm_results[0], list)
        ):
            # Multiple datasets: assign fixed colors per dataset.
            fixed_colors = [
                "blue",
                "orange",
                "red",
                "purple",
                "green",
                "gray",
                "pink",
            ]  # Extend if more than 2 datasets are needed.
            for i, dataset in enumerate(gemm_results):
                plot_points_from_results(
                    dataset, color=fixed_colors[i % len(fixed_colors)]
                )

    plt.xlabel("Arithmetic Intensity (FLOPs/Byte)")
    plt.ylabel("Performance (TFLOP/s)")
    plt.title(
        f'Realistic Roofline Model for {len(gpu_info)} X {gpu_info[0]["name"]} GPU(s)'
    )
    plt.grid(True, which="both", ls="--", linewidth=0.5)
    plt.legend()

    # Adjust x/y limits for better visibility
    plt.xlim(0, cross_point * 2)
    plt.ylim(0, peak_flops_tflops * 1.5)

    plt.show()


def plot_speedup_vs_theory(
    gpu_counts,
    actual_times,
    amdahl_ps=[0.5, 0.8, 1.00],
    gustafson_alphas=[0.10, 0.05, 0.01, 0.00],
    title="Parallel Scaling: Actual vs Amdahl vs Gustafson",
    max_y=4.2,
    figsize=(14, 5),
):
    """
    Plot actual speedup vs Amdahl's and Gustafson's Law with multiple parameter values.

    Parameters:
    - gpu_counts: list or np.array of GPU counts used
    - actual_times: list or np.array of execution times corresponding to gpu_counts
    - amdahl_ps: list of fixed p values to plot for Amdahl's Law
    - gustafson_alphas: list of fixed α values to plot for Gustafson's Law
    - title: suptitle for the plot
    - max_y: y-axis limit for speedup
    - figsize: tuple for figure size
    """

    gpu_counts = np.array(gpu_counts)
    actual_times = np.array(actual_times)
    base_time = actual_times[0]
    actual_speedup = base_time / actual_times

    def amdahl(n, p):
        return 1 / ((1 - p) + p / n)

    def gustafson(n, alpha):
        return n - alpha * (n - 1)

    x_vals = np.linspace(gpu_counts[0], gpu_counts[-1], 100)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    # Amdahl
    ax1.plot(gpu_counts, actual_speedup, "o-", label="Actual Speedup", linewidth=2)
    for p in amdahl_ps:
        ax1.plot(
            x_vals, amdahl(x_vals, p), "--", label=f"Amdahl (p={p:.2f})", linewidth=1.5
        )
    ax1.set_title("Amdahl's Law (Fixed Parallel Fractions)")
    ax1.set_xlabel("Number of GPUs")
    ax1.set_ylabel("Speedup")
    ax1.grid(True)
    ax1.set_ylim(1, max_y)
    ax1.legend()

    # Gustafson
    ax2.plot(gpu_counts, actual_speedup, "o-", label="Actual Speedup", linewidth=2)
    for alpha in gustafson_alphas:
        ax2.plot(
            x_vals,
            gustafson(x_vals, alpha),
            ":",
            label=f"Gustafson (α={alpha:.2f})",
            linewidth=1.5,
        )
    ax2.set_title("Gustafson's Law (Fixed Serial Fractions)")
    ax2.set_xlabel("Number of GPUs")
    ax2.grid(True)
    ax2.set_ylim(1, max_y)
    ax2.legend()

    plt.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def plot_shards_aligned(
    result, full_shape=(4, 4), title_prefix=None, show_pattern=True
):
    """
    Plot aligned shards for a single pattern result in a 2x2 GPU mesh layout.

    Parameters:
    - result: dict containing keys 'all_ranks', 'local_shapes', 'mesh_shape', and optionally 'shard_pattern'
    - full_shape: global tensor shape, default (4, 4)
    - title_prefix: optional string prefix for the row title
    - show_pattern: whether to show the shard pattern in the plot
    """
    shard_data = result["all_ranks"]
    shard_shapes = result["local_shapes"]
    mesh_shape = result["mesh_shape"]
    pattern = result.get("shard_pattern", "") or "replicated"

    rows, cols = full_shape
    mesh_rows, mesh_cols = mesh_shape

    fig, axs = plt.subplots(2, 2, figsize=(8, 6))
    axs = axs.flatten()

    for rank in range(4):
        ax = axs[rank]
        shard = np.array(shard_data[rank])

        # Estimate the start row/col from the first element in the shard
        val = int(shard[0, 0])
        row_start, col_start = divmod(val, cols)
        row_start = row_start - (row_start % shard.shape[0])
        col_start = col_start - (col_start % shard.shape[1])

        ax.set_xlim(0, cols)
        ax.set_ylim(rows, 0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")

        for i in range(shard.shape[0]):
            for j in range(shard.shape[1]):
                val = int(shard[i, j])
                r = row_start + i
                c = col_start + j
                ax.add_patch(
                    plt.Rectangle(
                        (c, r), 1, 1, color=plt.cm.Blues(val / 15), ec="black"
                    )
                )
                ax.text(
                    c + 0.5,
                    r + 0.5,
                    f"{val}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="black",
                )

        ax.set_title(f"Rank {rank}", fontsize=10)

    if show_pattern and title_prefix:
        axs[0].text(
            -0.5,
            1.15,
            f"{title_prefix}: {pattern}",
            transform=axs[0].transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )
    elif title_prefix:
        axs[0].text(
            -0.5,
            1.15,
            f"{title_prefix}",
            transform=axs[0].transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )

    plt.tight_layout()
    plt.show()


def launch_flashcard_quiz(results):
    """
    Launch an interactive flashcard quiz from a list of DTensor result dicts.
    """
    valid_results = [r for r in results if "error" not in r]

    score = 0
    round_number = 0
    current_result = None
    current_pattern = None

    input_box = widgets.Text(placeholder="e.g. Ix_J, Ix_Jy, I_Jxy, replicated")
    submit_button = widgets.Button(description="Submit", button_style="primary")
    feedback_label = widgets.Label()
    score_label = widgets.Label()
    output_area = widgets.Output()

    def next_question():
        nonlocal current_result, current_pattern, round_number
        round_number += 1
        current_result = random.choice(valid_results)
        current_pattern = current_result["shard_pattern"] or "replicated"

        with output_area:
            clear_output(wait=True)
            plot_shards_aligned(
                current_result,
                title_prefix=f"Flashcard #{round_number}",
                show_pattern=False,
            )

    def handle_submit(_):
        nonlocal score
        guess = input_box.value.strip()
        normalized = guess if guess != "replicated" else ""

        if normalized == current_pattern:
            score += 1
            feedback_label.value = f"✅ Correct! It was {current_pattern}"
        else:
            feedback_label.value = f"❌ Incorrect. It was {current_pattern}"

        score_label.value = f"Score: {score}/{round_number}"
        input_box.value = ""
        next_question()

    submit_button.on_click(handle_submit)

    quiz_ui = widgets.VBox(
        [
            output_area,
            widgets.HBox([input_box, submit_button]),
            feedback_label,
            score_label,
        ]
    )

    next_question()
    display(quiz_ui)

def plot_cost_vs_batch(results):
    # Extract data
    batch_sizes = [entry['batch_size'] for entry in results]
    costs = [entry['cost_per_1m_tokens'] for entry in results]

    # Create the plot
    plt.figure()
    plt.plot(batch_sizes, costs, marker='o')
    plt.xlabel('Batch Size')
    plt.ylabel('Cost per 1M Tokens ($)')
    plt.title('Cost vs Batch Size')
    plt.grid(True)
    plt.xticks(batch_sizes)  # ensure all batch sizes are marked
    plt.show()
