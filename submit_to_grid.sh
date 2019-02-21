#!/bin/bash

git pull
qsub -t 1-10 -l short -l gpus=2 -cwd run_grid.sh
