#!/bin/bash

git pull
qsub -t 1-9 -l short -l gpus=2 -cwd run_grid.sh
