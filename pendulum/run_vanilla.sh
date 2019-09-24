#!/bin/bash

for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=0 python vanilla_a3c.py $i > vanilla_log_$i.txt
done
