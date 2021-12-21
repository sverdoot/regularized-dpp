#!/usr/bin/bash
declare -a DATASETS=(
    # "triazines_scale"
    "mg_scale" 
    # "mpg_scale"
    "housing_scale" 
    "bodyfat_scale"
    "space_ga_scale"
    )

for dataset in ${DATASETS[@]}
do
    echo "$dataset"
    python regdpp/main.py configs/${dataset}.yaml
done