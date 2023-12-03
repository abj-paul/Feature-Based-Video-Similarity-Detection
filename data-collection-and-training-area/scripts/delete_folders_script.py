import os
import shutil

def delete_folders():
    # Get the current directory
    current_directory = os.getcwd()

    # List all items in the directory
    items = os.listdir(current_directory)

    # Iterate over items
    for item in items:
        # Check if the item is a directory
        if os.path.isdir(item):
            # Delete the directory and its contents
            try:
                shutil.rmtree(item)
                print(f'Deleted folder: {item}')
            except Exception as e:
                print(f'Error deleting folder {item}: {e}')

# Run the function to delete folders
delete_folders()

