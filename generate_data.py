import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Set a random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 10000
half_samples = num_samples // 2

# Number of relevant features
num_features = 3
num_relevant_features = 2

# Define relevant features indices
relevant_feature_indices = [0,1]

# Generate synthetic features
features = np.random.uniform(0, 10, size=(num_samples, num_features))

# Generate synthetic labels based on concepts
labels_concept1 = (features[:half_samples, relevant_feature_indices].sum(axis=1) > 7).astype(int)
labels_concept2 = (features[half_samples:, relevant_feature_indices].sum(axis=1) > 9).astype(int)

# Introduce noise (10% of samples)
noise_indices = np.random.choice(num_samples//2, size=int(0.1 * num_samples), replace=False)
labels_concept1[noise_indices] = 1 - labels_concept1[noise_indices]
noise_indices = np.random.choice(num_samples//2, size=int(0.1 * num_samples), replace=False)
labels_concept2[noise_indices] = 1 - labels_concept2[noise_indices]

# Combine features and labels for each concept
data_concept1 = np.column_stack((features[:half_samples, :], labels_concept1))
data_concept2 = np.column_stack((features[half_samples:, :], labels_concept2))

# Combine both concepts into a single dataset
synthetic_dataset = np.vstack((data_concept1, data_concept2))

# Shuffle the dataset
#np.random.shuffle(synthetic_dataset)

# Convert to Pandas DataFrame for convenience
column_names = [f"feature_{i}" for i in range(num_features)] + ['label']
synthetic_df = pd.DataFrame(synthetic_dataset, columns=column_names)



# Save the data
PATH = os.path.dirname(os.path.abspath(__file__))
folder_path = f"{PATH}/data"
os.makedirs(folder_path, exist_ok=True)
file_name = "synthetic_dataset.csv"
synthetic_df.to_csv(f"{folder_path}/{file_name}", index=False)


# Display the synthetic dataset
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
scatter_concept1 = ax1.scatter(synthetic_df.iloc[:half_samples, 0], synthetic_df.iloc[:half_samples, 1],
            c=synthetic_df['label'][:half_samples].map({0: 'cornflowerblue', 1: 'lightsalmon'}),
            label=synthetic_df['label'] , marker='o', alpha=0.7)

#ax1.legend(labels=['Class 0', 'Class 1'])
ax1.set_xlabel('Relevant feature 1')
ax1.set_ylabel('Relevant feature 2')
ax1.set_title('Concept 1')

scatter_concept2 = ax2.scatter(synthetic_df.iloc[half_samples:, 0], synthetic_df.iloc[half_samples:, 1],
            c=synthetic_df['label'][half_samples:].map({0: 'cornflowerblue', 1: 'lightsalmon'}),
            marker='o', alpha=0.7)

#ax2.legend(labels=['Class 0', 'Class 1'])
ax2.set_xlabel('Relevant feature 1')
ax2.set_ylabel('Relevant feature 2')
ax2.set_title('Concept 2')

ax1.legend(handles=[scatter_concept1, scatter_concept2], labels=['Class 0', 'Class 1'])
ax2.legend(handles=[scatter_concept1, scatter_concept2], labels=['Class 0', 'Class 1'])

leg = ax1.get_legend()
leg.legendHandles[0].set_color('cornflowerblue')
leg.legendHandles[1].set_color('lightsalmon')

leg = ax2.get_legend()
leg.legendHandles[0].set_color('cornflowerblue')
leg.legendHandles[1].set_color('lightsalmon')
plt.suptitle('Synthetic dataset with 2 concepts')
plt.show()










