#!/bin/bash

source thesis/bin/activate
rm -rf grid_results
mkdir grid_results
python two-encoder-state-ray.py
deactivate