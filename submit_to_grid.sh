#!/bin/bash

git pull
qsub -t 1-9 -cwd -l short -l gpus=2 run_grid.sh
