import os
import re

def delete_specific_files(folder_path, pattern):
    for filename in os.listdir(folder_path):
        if re.match(pattern, filename):
            os.remove(os.path.join(folder_path, filename))
            print(f"Deleted {filename}")

# Usage
folder_path = './input/B'  # Replace with the path to your folder
pattern = r'B_\d+_realB\.jpg'  # Regular expression for the pattern A_n_realA.jpg
delete_specific_files(folder_path, pattern)



