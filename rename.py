import os

# Directory containing the files
directory = 'Images/Seema1'

# Define the renaming range
old_prefix = 'Seema1'
new_prefix = 'Seema'

# Iterate over files in the directory
for filename in os.listdir(directory):
    if filename.startswith(old_prefix):
        # Extract the number part of the filename
        number_str = filename[len(old_prefix):]

        try:
            # Convert number string to integer
            number = int(number_str)

            # Check if the number falls within the range 111-130
            if 111 <= number <= 130:
                # Calculate new number for renaming
                new_number = number - 79  # Adjusting the range from 111-130 to 32-50

                # Form new filename
                new_filename = f"{new_prefix}{new_number}"

                # Rename file
                os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
                print(f"Renamed {filename} to {new_filename}")
        except ValueError:
            print(f"Skipping {filename} as it does not match expected pattern")
