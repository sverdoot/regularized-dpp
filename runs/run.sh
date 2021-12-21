#!/usr/bin/bash
declare -a DATASETS=(
    "space_ga_scale"
    # "triazines_scale"
    "mg_scale" 
    # "mpg_scale"
    "housing_scale" 
    "bodyfat_scale"
    )

for dataset in ${DATASETS[@]}
do
    echo "$dataset"
    python regdpp/main.py configs/${dataset}.yaml
done