"""Helper functions for main program"""

import os
import sys


def check_file_exists(path: str, action: str = "exit") -> None:
    """Check if a file already exists and if it is to be overwritten

    Args:
        path   (str): path to check
        action (str): should the program exit if answered no or not
    """
    if os.path.exists(path):
        confirm = input("File already exists. Overide? [Y/n] ")
        if confirm not in ("Y", "y"):
            if action == "return":
                print("Returning...")
                return False
            print("Exiting...")
            sys.exit()

    return True
