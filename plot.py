import json
import matplotlib.pyplot as plt

# Function to read the JSON file and extract the 'Result' data
def read_result_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data['Result']  # Assuming 'Result' is the key in the JSON file

# Function to plot the accuracy and surprisal curves
def plot_curves(accuracy_data, surprisal_data, save_path=None):
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 7))

    x = range(1, len(accuracy_data) + 1)

    # Plot the accuracy curve
    ax.plot(x, accuracy_data, marker='o', color='#1f77b4', linestyle='-', 
            linewidth=2.5, markersize=8, markerfacecolor='white', label="Accuracy Curve", zorder=5)

    # Plot the surprisal curve
    ax.plot(x, surprisal_data, marker='s', color='#ff7f0e', linestyle='--',
            linewidth=2.5, markersize=7, markerfacecolor='white', label="Surprisal Curve", zorder=4)

    # Labels
    ax.set_xlabel("Layer ID", fontsize=16, fontweight='bold', family='DejaVu Sans')
    ax.set_ylabel("Value", fontsize=16, fontweight='bold', family='DejaVu Sans')

    # Grid and ticks
    ax.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=0.7)
    plt.xticks(fontsize=14, fontweight='light', family='DejaVu Sans')
    plt.yticks(fontsize=14, fontweight='light', family='DejaVu Sans')

    # Legend
    ax.legend(loc='lower right', fontsize=14, frameon=True, facecolor='white',
              edgecolor='gray', framealpha=0.9, borderpad=1, labelspacing=1.5,
              title="Legend", title_fontsize=16)

    # Save and show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


# Main function to run the program
if __name__ == "__main__":
    # Provide the path to your JSON file
    dataset = "GSM8K"
    model = "Qwen2.5-Math-1.5B-Instruct"
    eval_type = "Right"
    accuracy_json_file = f"outputs/output/{dataset.lower()}/qwen25-math-cot_seed42_{model}_{eval_type}_-1_-1_accuracy.json"
    surprisal_json_file = f"outputs/output/{dataset.lower()}/qwen25-math-cot_seed42_{model}_{eval_type}_-1_-1_surprisal.json"
    save_path = 'plots/gsm8k_1.5B_Instruct.png'

    # Read data
    accuracy_data = read_result_from_json(accuracy_json)
    surprisal_data = read_result_from_json(surprisal_json)

    # Plot the accuracy curve
    plot_accuracy_curve(result_data, save_path, save_path)
