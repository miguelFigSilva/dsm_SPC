import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set a random seed for reproducibility
np.random.seed(42)

# Define the dataset parameters
num_samples = 120000
segments = [0, 15000, 30000, 45000, 60000, 75000, 90000, 105000, 120000]
num_features = 3
relevant_feature_indices = [0, 1]
concepts = [8, 9, 7, 10]

# Generate synthetic features
features = np.random.uniform(0, 10, size=(num_samples, num_features))

# Initialize lists to store scatter plots and legend handles
scatter_plots = []
legend_handles = []

# Generate scatter plots for each concept segment
fig, axs = plt.subplots(2, len(segments) // 2, figsize=(15, 10))
for i in range(1, len(segments)):
    # Generate labels based on concepts and introduce noise
    labels = (features[segments[i - 1]:segments[i], relevant_feature_indices].sum(axis=1) > concepts[(i - 1) % len(concepts)]).astype(int)
    n = segments[i] - segments[i - 1]
    noise_indices = np.random.choice(n, size=int(0.1 * n), replace=False)
    labels[noise_indices] = 1 - labels[noise_indices]
    
    # Create a DataFrame for the current segment
    segment_df = pd.DataFrame(np.column_stack((features[segments[i - 1]:segments[i], :], labels)), columns=[f'feature_{j}' for j in range(num_features)] + ['label'])
    
    # Calculate the subplot position
    row_index = (i - 1) // (len(segments) // 2)
    col_index = (i - 1) % (len(segments) // 2)
    
    # Scatter plot for the current segment
    scatter_plot = axs[row_index, col_index].scatter(segment_df['feature_0'], segment_df['feature_1'], c=segment_df['label'].map({0: 'cornflowerblue', 1: 'lightsalmon'}), marker='o', alpha=0.7)
    scatter_plots.append(scatter_plot)
    
    # Set labels and title for the current plot
    axs[row_index, col_index].set_xlabel('Relevant feature 1')
    axs[row_index, col_index].set_ylabel('Relevant feature 2')
    axs[row_index, col_index].set_title(f'Concept {i}')
    
    # Create legend handles for the scatter plot
    legend_handles.append(scatter_plot)

# Add legend to the last plot
legend_labels = ['Class 0', 'Class 1']
fig.legend(legend_handles, labels=legend_labels, loc='upper right')

# Adjust layout and display the plot
plt.tight_layout()
plt.suptitle('Synthetic dataset with multiple concepts', y=1.05)
plt.show()



# Convert to Pandas DataFrame for convenience
column_names = [f"feature_{i}" for i in range(num_features)] + ['label']
df = pd.DataFrame(dataset, columns=column_names)

# Save the data
PATH = os.path.dirname(os.path.abspath(__file__))
folder_path = f"{PATH}/data"
os.makedirs(folder_path, exist_ok=True)
file_name = "SEAconcepts.csv"
df.to_csv(f"{folder_path}/{file_name}", index=False)