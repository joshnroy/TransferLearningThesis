#!/bin/bash

for i in {1..5}
do
    CUDA_VISIBLE_DEVICES=1 python darla_a3c.py $i > darla_log_train_$i.txt
done
