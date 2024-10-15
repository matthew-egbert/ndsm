import os

def create_log_file(name):
    file_path = f"/home/mde/repos/ndsm/logs/{name}.txt"

    # Check if the file exists
    if not os.path.exists(file_path):
        # Create the file if it doesn't exist
        with open(file_path, 'a') as file:
            pass
    else:
        # Clear the file if it exists
        with open(file_path, 'w') as file:
            file.write("")

    return open(file_path, 'a')