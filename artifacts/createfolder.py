import os
import tempfile


class CreateFolder:
    def __init__(self):
        self.current_directory = os.getcwd()

    def CreateFolder(self, folder_name):
        if not isinstance(folder_name, str) or not folder_name.strip():
            raise ValueError("invalid_name")

        full_path = os.path.join(self.current_directory, folder_name)
        if os.path.exists(full_path):
            raise ValueError("folder_exists")

        try:
            os.mkdir(full_path)
            return True
        except Exception as e:
            raise RuntimeError(f"An error occurred: {e}")
