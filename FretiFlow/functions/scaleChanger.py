import pandas as pd
import numpy as np

# Define sliding window size
window_size = 3  # Adjust as per your preference

# Read train data
train_data = pd.read_csv(r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\output\train_data_v3.csv')

# Define column names for input points and labels
input_columns = [f'input_{i+1}' for i in range(window_size)]
label_columns = [f'label_{i+1}' for i in range(window_size)]

# Initialize input and label columns with NaN values
for col in input_columns + label_columns:
    train_data[col] = np.nan

# Iterate over each row in train data
for index, row in train_data.iterrows():
    section_start_str = row['song.sections_start']

    try:
        if isinstance(section_start_str, float):
            section_start = np.array([section_start_str])
        else:
            section_start = np.array([float(value) for value in section_start_str.split(',')])

        # Randomly select a point from section_start to be included in the window
        selected_point = np.random.choice(section_start)

        # Get indices of the sliding window points
        window_indices = np.arange(len(section_start))[-window_size:]

        # Fill input columns with the window points
        train_data.loc[index, input_columns] = section_start[window_indices]

        # Check if each window point corresponds to a scale change
        for i, idx in enumerate(window_indices):
            label = 1 if np.abs(selected_point - section_start[idx]) <= 1 else 0
            train_data.at[index, label_columns[i]] = label

    except ValueError:
        # Handle invalid values
        train_data.loc[index, input_columns + label_columns] = np.nan

# Read test data
test_data = pd.read_csv(r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\output\test_data_v3.csv')

# Initialize input and label columns with NaN values for test data
for col in input_columns + label_columns:
    test_data[col] = np.nan

# Iterate over each row in test data
for index, row in test_data.iterrows():
    section_start_str = row['song.sections_start']

    try:
        if isinstance(section_start_str, float):
            section_start = np.array([section_start_str])
        else:
            section_start = np.array([float(value) for value in section_start_str.split(',')])

        # Randomly select a point from section_start to be included in the window
        selected_point = np.random.choice(section_start)

        # Get indices of the sliding window points
        window_indices = np.arange(len(section_start))[-window_size:]

        # Fill input columns with the window points
        test_data.loc[index, input_columns] = section_start[window_indices]

        # Check if each window point corresponds to a scale change
        for i, idx in enumerate(window_indices):
            label = 1 if np.abs(selected_point - section_start[idx]) <= 1 else 0
            test_data.at[index, label_columns[i]] = label

    except ValueError:
        # Handle invalid values
        test_data.loc[index, input_columns + label_columns] = np.nan

# Save the modified train and test data
train_data.to_csv(r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\output\modified_train_data.csv', index=False)
test_data.to_csv(r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\output\modified_test_data.csv', index=False)
