import os

# Specify the source folder where your images are located
source_folder = '../check/v4/B'

# Specify the destination folder where you want to move the renamed images
destination_folder = '../check/v4/B2A/generated'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True)

# Function to rename files
def rename_files(source_folder, destination_folder, suffix='B2A'):
    file_list = os.listdir(source_folder)
    count = 1
    for filename in file_list:
        if filename.endswith(suffix + ".jpg"):
            # Construct the new filename
            new_filename = f'G_{count}.jpg'
            
            # Full path to the source file
            src_path = os.path.join(source_folder, filename)
            
            # Full path to the destination file
            dest_path = os.path.join(destination_folder, new_filename)
            
            # Rename the file and move it to the destination folder
            os.rename(src_path, dest_path)
            
            count += 1

# Rename the A2B images
rename_files(source_folder, destination_folder)

# Rename the realA images
# rename_files(source_folder, destination_folder, 'B_real')
