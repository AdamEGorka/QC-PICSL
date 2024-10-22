import os
import shutil

source_dir = r"C:\Users\adame\Desktop\QC Project\chunkt1_20230512zip\chunkt1_20230512"

# Destination directory where files with "_sr_" will be copied
destination_dir = r"C:\Users\adame\Desktop\QC Project\chunkt1_20230512zip\sr_only"

# Loop through files in the source directory
for filename in os.listdir(source_dir):
    if filename.endswith(".nii.gz") and "_sr_" in filename:
        # Construct the full paths for the source and destination files
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)

        # Copy the file from the source directory to the destination directory
        shutil.copy(source_file, destination_file)
        print(f"Copied: {source_file} -> {destination_file}")
