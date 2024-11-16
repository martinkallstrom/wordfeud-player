#!/bin/bash

# Add all changes
git add .

# Get the commit message from command line argument, default to "Update"
commit_message=${1:-"Update"}

# Commit with the provided or default message
git commit -m "$commit_message"

# Push to origin main
git push origin main

echo "Changes pushed to origin/main"
