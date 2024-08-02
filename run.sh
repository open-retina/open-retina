#!/usr/bin/env bash

for cells in "all" "on" "off" "on-off"; do
    echo "Run training for $cells with Acs"
    saveFolder="models/rgcs_subsets/all"; mkdir -p ${saveFolder}; ./scripts/hoefling_2022_train.py --data_folder /gpfs01/euler/data/SharedFiles/projects/TP12/ --save_folder $saveFolder --datasets "natural" | tee ${saveFolder}/train.log ;
    echo "Run training for $cells without Acs"
    saveFolder="models/rgcs_subsets/all_no_acs"; mkdir -p ${saveFolder}; ./scripts/hoefling_2022_train.py --data_folder /gpfs01/euler/data/SharedFiles/projects/TP12/ --save_folder $saveFolder --datasets "natural" --max_id 32 | tee ${saveFolder}/train.log ;
done

