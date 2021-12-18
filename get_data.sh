#!/usr/bin/bash
declare -a DATASETS=(
    "triazines_scale"
    "mg_scale" 
    "mpg_scale"
    "housing_scale" 
    "bodyfat_scale"
    )

for dataset in ${DATASETS[@]}
do
    echo "$dataset"
    wget -nc https://www.csie.ntu.edu.tw/\~cjlin/libsvmtools/datasets/regression/"${dataset}" -P data
done