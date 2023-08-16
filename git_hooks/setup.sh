#!/bin/bash

# Get the directory of the current script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Make the pre-commit script executable
chmod +x "$DIR/pre-commit"

# Create a symbolic link for the pre-commit hook
ln -s -f "$DIR/pre-commit" .git/hooks/pre-commit
