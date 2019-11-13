#!/bin/sh

files=(./debug/*)

if [ ${#files[@]} -gt 1 ]; then 
    rm ./debug/*
    echo "Cleaning debug files!"
else
    echo "Nothing to clean!"
fi