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

        # Generate input value
        if np.random.rand() < 0.3:  # 30% probability to choose from section starts
            input_value = np.random.choice(section_start)
        else:
            input_value = np.random.uniform(0, 600)  # Randomly choose a value within the range

        # Generate output label
        label_value = 1.0 if input_value in section_start else 0.0

        # Update the input and label columns
        train_data.at[index, input_column] = input_value
        train_data.at[index, label_column] = label_value

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

        # Generate input value
        if np.random.rand() < 0.5:  # 50% probability to choose from section starts
            input_value = np.random.choice(section_start)
        else:
            input_value = np.random.uniform(0, 600)  # Randomly choose a value within the range

        # Generate output label
        label_value = 1.0 if input_value in section_start else 0.0

        # Update the input and label columns
        test_data.at[index, input_column] = input_value
        test_data.at[index, label_column] = label_value

    except ValueError:
        # Handle invalid values
        test_data.at[index, input_column] = np.nan
        test_data.at[index, label_column] = np.nan

# Save the modified train and test data
train_data.to_csv(r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\output\cleaned_train_data.csv', index=False)
test_data.to_csv(r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\output\cleaned_test_data.csv', index=False)
