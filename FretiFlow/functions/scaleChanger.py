import pandas as pd
import numpy as np

# Read train data
train_data = pd.read_csv(r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\output\train_data_v3.csv')

# Define column names for input and label
input_column = 'input'
label_column = 'label'

# Iterate over each row in train data
for index, row in train_data.iterrows():
    section_start_str = row['song.sections_start']

    try:
        if isinstance(section_start_str, float):
            section_start = np.array([section_start_str])
        else:
            section_start = np.array([float(value) for value in section_start_str.split(',')])

        num_points = len(section_start)

        # Generate input array
        lower_bound = 1.0  # Define lower bound for random point
        upper_bound = 600.0  # Define upper bound for random point
        # Generate input array
        random_point = np.random.choice(section_start)
        input_array = [random_point, random_point + 1] + np.random.uniform(lower_bound, upper_bound, size=num_points-2).tolist()


        # Convert input_array and section_start to NumPy arrays
        input_array = np.array(input_array)
        section_start = np.array(section_start)

        # Generate label array
        label_array = np.zeros_like(input_array)
        for i, input_value in enumerate(input_array):
            label_array[i] = 1.0 if np.any(np.abs(section_start - input_value) <= 1.0) else 0.0

        # Update the input and label columns
        train_data.at[index, input_column] = ', '.join(map(str, input_array))
        train_data.at[index, label_column] = ', '.join(map(str, label_array.tolist()))

    except ValueError:
        # Handle invalid values
        train_data.at[index, input_column] = np.nan
        train_data.at[index, label_column] = np.nan

# Read test data
test_data = pd.read_csv(r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\output\test_data_v3.csv')

# Iterate over each row in test data
for index, row in test_data.iterrows():
    section_start_str = row['song.sections_start']

    try:
        if isinstance(section_start_str, float):
            section_start = np.array([section_start_str])
        else:
            section_start = np.array([float(value) for value in section_start_str.split(',')])

        num_points = len(section_start)

        # Generate input array
        lower_bound = 1.0  # Define lower bound for random point
        upper_bound = 600.0  # Define upper bound for random point
        random_point = np.random.choice(section_start)
        input_array = [random_point, random_point + 1] + np.random.uniform(lower_bound, upper_bound, size=num_points-2).tolist()

        # Convert input_array and section_start to NumPy arrays
        input_array = np.array(input_array)
        section_start = np.array(section_start)

        # Generate label array
        label_array = np.zeros_like(input_array)
        for i, input_value in enumerate(input_array):
            label_array[i] = 1.0 if np.any(np.abs(section_start - input_value) <= 1.0) else 0.0

        # Update the input and label columns
        test_data.at[index, input_column] = ', '.join(map(str, input_array))
        test_data.at[index, label_column] = ', '.join(map(str, label_array.tolist()))

    except ValueError:
        # Handle invalid values
        test_data.at[index, input_column] = np.nan
        test_data.at[index, label_column] = np.nan

# Save the modified train and test data
train_data.to_csv(r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\output\modified_train_data.csv', index=False)
test_data.to_csv(r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\output\modified_test_data.csv', index=False)
