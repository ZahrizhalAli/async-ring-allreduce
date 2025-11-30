import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import os

def format_bytes(x, pos):
    """Formats axis ticks into readable sizes (KB, MB, GB)."""
    if x == 0:
        return '0 B'
    # 1 KB = 1024 Bytes logic
    if x >= 1024**3:
        return f'{x / 1024**3:.0f} GB'
    elif x >= 1024**2:
        return f'{x / 1024**2:.0f} MB'
    elif x >= 1024:
        return f'{x / 1024:.0f} KB'
    else:
        return f'{x:.0f} B'

def plot_scurve(input_file):
    # 1. Setup Theme
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    plt.rcParams['font.family'] = 'sans-serif'
    
    # 2. Load Data
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        return
        
    data = pd.read_csv(input_file)

    # 3. FILTER OUT NCCL
    data = data[data['impl'] != 'nccl']

    # 4. Data Preprocessing
    # Assuming 'throughput' in CSV is MB/s (Bytes/microsecond)
    # Convert to GB/s (1000 MB/s = 1 GB/s for standard networking reporting)
    data['Bandwidth (GB/s)'] = data['throughput'] / 1000.0

    # 5. Create Plot
    plt.figure(figsize=(10, 6))
    
    p = sns.lineplot(
        data=data,
        x='input_bytes',
        y='Bandwidth (GB/s)',
        hue='impl',
        markers=True,
        dashes=True,
        linewidth=2.5,
        markersize=8,
        palette="viridis"
    )

    # 6. Beautify Axes
    p.set_xscale('log')
    p.xaxis.set_major_formatter(ticker.FuncFormatter(format_bytes))
    
    plt.grid(True, which="minor", ls="--", alpha=0.3)
    plt.grid(True, which="major", ls="-", alpha=0.8)

    plt.title("Ring AllReduce: Naive vs (Async) 4GPUS", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Message Size", fontsize=12, labelpad=10)
    plt.ylabel("Effective Bandwidth (GB/s)", fontsize=12, labelpad=10)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Implementation')

    # 7. Save
    output_file = "s_curve_comparison.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot S-Curve from benchmark CSV")
    parser.add_argument("input_file", type=str, nargs='?', default="results/output.csv", help="Path to the CSV file (default: results/output.csv)")
    
    args = parser.parse_args()
    plot_scurve(args.input_file)