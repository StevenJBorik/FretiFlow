import csv
import re 

input_file = r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\output\test_data_v2.csv'
output_file = r'C:\Users\SBD2RP\OneDrive - MillerKnoll\installs\Desktop\output\test_data_v3.csv'

section_start_column = None

# Open the input and output CSV files
with open(input_file, 'r') as csv_in, open(output_file, 'w', newline='') as csv_out:
    reader = csv.reader(csv_in)
    writer = csv.writer(csv_out)

    # Find the column index of "song.sections_start"
    header = next(reader)  # Read the header row
    for i, column_name in enumerate(header):
        if column_name == 'song.sections_start':
            section_start_column = i
            break

    # Check if "song.sections_start" column was found
    if section_start_column is None:
        print("Column 'song.sections_start' not found in the input file.")
        exit()

    # Write the header row to the output file
    writer.writerow(header)

    # Iterate over each row in the input file
    for row in reader:
        section_start_str = row[section_start_column]  # Get the section start string

        # Use regular expression to extract numerical values
        section_start_values = re.findall(r"[-+]?\d*\.\d+|\d+", section_start_str)

        # Join the extracted values with commas
        section_start_formatted = ", ".join(section_start_values)

        # Update the section start value in the row
        row[section_start_column] = section_start_formatted

        # Write the modified row to the output file
        writer.writerow(row)

print('CSV file successfully modified.')