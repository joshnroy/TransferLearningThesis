#!/bin/bash

for i in {1..8}
do
    CUDA_VISIBLE_DEVICES=0 python darla_a3c.py $i > darla_log_$i.txt
done
