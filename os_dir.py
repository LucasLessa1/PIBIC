import os
import shutil

# List of source folders containing files to move
source_folders = ['C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/images_001/imagens', 'C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/images_002/imagens', 
                  'C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/images_003/imagens', 'C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/images_004/imagens',
                  'C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/images_005/imagens', 'C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/images_006/imagens',
                  'C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/images_007/imagens', 'C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/images_008/imagens',
                  'C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/images_009/imagens', 'C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/images_010/imagens',
                  'C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/images_011/imagens', 'C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/images_012/imagens']

# source_folders = ['C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/test_flder']
# Destination folder where you want to move the files
destination_folder = r'C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/images/'

# Iterate through the source folders
for source_folder in source_folders:
    # Get a list of files in the source folder
    files_to_move = os.listdir(source_folder)

    # Iterate through the files and move them to the destination folder
    for file_to_move in files_to_move:
        source_path = os.path.join(source_folder, file_to_move)
        destination_path = os.path.join(destination_folder, file_to_move)
        
        # Check if the file already exists in the destination folder
        if os.path.exists(destination_path):
            # You can handle the case where a file with the same name already exists here
            # For example, you can rename the file before moving it.
            pass
        
        # Move the file to the destination folder
        shutil.move(source_path, destination_path)
        
        print(f"Moved '{file_to_move}' to '{destination_folder}'")

print("All files moved successfully.")




source_folder = os.path.normpath('C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/images')

# Paths to the destination folders
train_folder = 'train'
test_folder = 'test'
validation_folder = 'validation'
flag = False

# Check if the destination folders already exist, and create them if not
for folder in [train_folder, test_folder, validation_folder]:
    destination = os.path.join('C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/', folder)
    if not os.path.exists(destination):
        os.mkdir(destination)
    else:
        print(f"Folder '{folder}' already exists. Skipping creation.")
        
# Get a list of file names in the source folder
file_names = os.listdir(source_folder)

# Shuffle the list to ensure randomness
random.shuffle(file_names)

# Set the proportion of images for each folder
train_proportion = 0.9
test_proportion = 0.05
validation_proportion = 0.05

# Calculate the number of images for each folder
total_images = len(file_names)
num_train = int(total_images * train_proportion)
num_test = int(total_images * test_proportion)
num_validation = total_images - num_train - num_test

# Move the images to the corresponding folders
for i, file_name in enumerate(file_names):
    source = os.path.normpath(os.path.join(source_folder, file_name))
    
    if i < num_train:
        destination = os.path.normpath(os.path.join('C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/', train_folder, file_name))
    elif i < num_train + num_test:
        destination = os.path.normpath(os.path.join('C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/', test_folder, file_name))
    else:
        destination = os.path.normpath(os.path.join('C:/Users/Lucas/Documents/PIBIC/DATASET/NIH-CHEST/archive/', validation_folder, file_name))
    
    # Use shutil.move to move the images
    shutil.move(source, destination)

print("Images moved successfully!")



# import os
# import shutil
# import random

# # Set the paths for the source and destination directories

# new_dir_test = 'C:/Users/Lucas/PIBIC/new_dir_test'

# # Define the number of images you want to copy for each subfolder
# train_images_per_subfolder = 100
# valid_images_per_subfolder = 5

# # Function to copy a specified number of random images from a source folder to a destination folder
# def copy_images(source_dir, dest_dir, num_images):
#     if not os.path.exists(dest_dir):
#         os.makedirs(dest_dir)

#     files = os.listdir(source_dir)
#     random.shuffle(files)
#     selected_files = files[:num_images]

#     for file in selected_files:
#         src_path = os.path.join(source_dir, file)
#         dest_path = os.path.join(dest_dir, file)
#         shutil.copy(src_path, dest_path)

# # Loop through subfolders in dir_train and dir_valid
# for subfolder in os.listdir(dir_train):
#     subfolder_path_train = os.path.join(dir_test, subfolder)

#     # Copy images from dir_train to new_dir_test
#     copy_images(subfolder_path_train, os.path.join(new_dir_test, subfolder), 5)


# print("Image copying completed.")
