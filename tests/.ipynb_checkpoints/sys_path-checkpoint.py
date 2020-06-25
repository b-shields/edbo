import sys
import os

def main():
    """
    Script to temporarily add package to system path.
    """
    
    # Current working directory
    cwd = os.getcwd()
    path = cwd.replace('tests','edbo')
    
    if path not in sys.path:
        sys.path.append(path)

    return path
    
main()