import json
import matplotlib.pyplot as plt

# Function to read the JSON file and extract the 'Result' data
def read_result_from_json(json_file):
    with open(json_file, 'r') as file:
        data = json.load(file)
    return data['Result']  # Assuming 'Result' is the key in the JSON file

# Function to plot the accuracy curve
def plot_accuracy_curve(result_data, save_path, dataset, model):
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(10, 7))

    # Plot the accuracy curve with a refined line style, color, and markers
    ax.plot(range(1, len(result_data) + 1), result_data, marker='o', color='#1f77b4', linestyle='-', 
            linewidth=2.5, markersize=8, markerfacecolor='white', label="Accuracy Curve", zorder=5)

    # Add title and labels with a clean, modern font style
    # ax.set_title(f"{model} on {dataset}", fontsize=20, fontweight='bold')
    ax.set_xlabel("Layer ID", fontsize=18)
    ax.set_ylabel("Accuracy", fontsize=18)

    # Set grid with subtle gray lines
    ax.grid(True, linestyle='--', alpha=0.7, color='gray', linewidth=0.7)

    # Customize ticks with a refined font size
    plt.xticks(fontsize=14, fontweight='light')
    plt.yticks(fontsize=14, fontweight='light')

    # Create a customized legend
    ax.legend(loc='lower right', fontsize=14, frameon=True, facecolor='white', edgecolor='#1f77b4', 
              framealpha=0.8, borderpad=1, labelspacing=1.5)

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  # High-quality output with tight layout

    # Show the plot
    plt.tight_layout()
    plt.show()

# Main function to run the program
if __name__ == "__main__":
    # Provide the path to your JSON file
    dataset = "GSM8K"
    model = "Qwen2.5-Math-1.5B-Instruct"
    eval_type = "Right"
    json_file = f"outputs/output/{dataset.lower()}/qwen25-math-cot_seed42_{model}_{eval_type}_-1_-1_accuracy.json"
    save_path = 'plots/gsm8k_1.5B_Instruct.png'
    # Read the result data from the JSON file
    result_data = read_result_from_json(json_file)

    # Plot the accuracy curve
    plot_accuracy_curve(result_data, save_path, dataset, model)