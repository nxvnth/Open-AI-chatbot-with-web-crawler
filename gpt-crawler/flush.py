import os
import shutil
# Function to change the working directory to the script's directory
def change_to_script_dir():
    # Get the absolute path to the script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Change the working directory to the script directory
    os.chdir(script_dir)

# Call the function to ensure we are in the correct directory
change_to_script_dir()

# Define the relative paths from the script's directory to the target directories
paths = [
    "./storage/datasets/default",
    "./storage/request_queues"
]

for path in paths:
    # Check if the path exists
    if os.path.exists(path):
        # List all files and directories in the path
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            
            # Check if it's a file and delete it
            if os.path.isfile(item_path):
                os.remove(item_path)
                print(f"File {item_path} has been deleted.")
            # If it's a directory, delete the directory and all its contents
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
                print(f"Directory {item_path} and all its contents have been deleted.")
    else:
        print(f"Path {path} does not exist.")
