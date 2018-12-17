#!/bin/bash

# creating a symbolic link
ln -sf $PWD/maximask.py $PWD/maximask

# adding current directory to the path
export PATH=$PATH:$PWD
