import os

# Define the root directory where your images are stored
root_directory = 'MSLD/'  # Replace with the actual path

# Create a list of tuples to store the directory name and image paths
directory_image_paths = []

# Walk through the directory structure and collect image paths
for dirpath, dirnames, filenames in os.walk(root_directory):
    image_paths = [os.path.join(dirpath, filename) for filename in filenames if filename.endswith('.jpg')]
    if image_paths:
        directory_name = os.path.basename(dirpath)
        directory_image_paths.append((directory_name, image_paths))

# Print the list of directory names and image paths
for directory_name, folder_images in directory_image_paths:
    print(f"Images in Directory '{directory_name}':")
    for image_path in folder_images:
        print(image_path)

