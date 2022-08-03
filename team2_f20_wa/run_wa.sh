#!/bin/bash
echo ================ Creating directories... ================

files=(Processed_Data/)
for file in ${files[@]}
do
    echo mkdir -p $file
    mkdir -p $file
done
printf "================ Finished creating directories ================\n\n"

pyfiles=(Cleaning_Pipeline.py fix_wa_sales_duplicates.py Area_Variables/area_variables.py linking_wmaps_pipeline.py
  Demographics/demographics.py Tag_Products/tag_products.py Competitive_Measure/add_competitive_measure.py)
for file in ${pyfiles[@]}
do
    echo ================ Running python $file ================
    python $file
    if [ $? -eq 1 ]; then
        echo Failed. Make sure you import all the correct libraries
        exit 1
        break
    fi
    echo ================ Finished python $file ================
done
printf "\nProcess succeeded. Output files stored in Processed_Data"
printf "\n\npipeline_final_output.csv is the result of the entire pipeline\n"
