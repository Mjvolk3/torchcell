#!/usr/bin/env python3
import argparse
import glob
import random
import subprocess
from typing import List

def open_random_notes(notes_dir: str, num_files: int):
    # Find all .md files in the specified directory (non-recursive)
    md_files = glob.glob(f"{notes_dir}/*.md")
    
    # Select a random subset of files
    selected_files = random.sample(md_files, k=min(num_files, len(md_files)))
    
    # Open each file with VSCode
    for file in selected_files:
        subprocess.run(["code", file])

def main():
    parser = argparse.ArgumentParser(description='Open a random set of Markdown notes with VSCode.')
    parser.add_argument('-n', '--number', type=int, default=1, help='Number of random notes to open')
    parser.add_argument('-d', '--directory', type=str, default='notes', help='Directory containing the notes')

    args = parser.parse_args()

    open_random_notes(args.directory, args.number)

if __name__ == '__main__':
    main()
