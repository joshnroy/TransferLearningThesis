#!/bin/bash

for i in {6..10}
do
    CUDA_VISIBLE_DEVICES=1 python attention_a3c.py $i > patt_log_$i.txt
done
