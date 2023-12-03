#!/bin/bash

# Loop through each mp4 file in the current directory
for file in *.mp4
do
    # Check if the file is not a noisy video
    if [[ $file != *noisy* ]]
    then
        # Extract the category name (e.g., Kuthar) from the filename
        category=$(echo $file | awk -F '_' '{print $1}')

        # Create the category folder if it doesn't exist
        mkdir -p $category

        # Move the file to the category folder
        mv "$file" "$category/"
    fi
done

