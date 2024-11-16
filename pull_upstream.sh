#!/bin/bash

# Fetch updates from upstream
git fetch upstream

# Merge upstream changes into main branch
git merge upstream/master main

echo "Changes pulled from upstream/master to main"
