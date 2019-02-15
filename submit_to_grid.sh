#!/bin/bash

git pull
qsub -cwd -l long -l gpus=2 run_grid.sh
