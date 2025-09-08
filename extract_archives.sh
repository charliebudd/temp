#!/bin/bash
for f in sarrarp/*/video*.zip; do
    dir="${f%.zip}"
    mkdir -p "$dir"
    unzip "$f" -d "$dir"
done
