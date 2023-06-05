#!/bin/bash

# Expand the template into multiple files, one for each item to be processed.
BATCH_JOBS_DIR=./batch-jobs
rm -r $BATCH_JOBS_DIR
mkdir $BATCH_JOBS_DIR

BLOBS_DIR="AAA"
COUNTS_PREFIXES_LIST=("cropland_2020_01_01" "cropland_2020_01_02" "cropland_2020_01_03")

for i in ${!COUNTS_PREFIXES_LIST[@]}
do
    COUNTS_PREFIXES=${COUNTS_PREFIXES_LIST[$i]}
    cat job-tmpl.yaml | sed "s/\$ID/$i/" | sed "s/\$BLOBS_DIR/$BLOBS_DIR/" | sed "s/\$COUNTS_PREFIXES/$COUNTS_PREFIXES/"  > $BATCH_JOBS_DIR/job-$i.yaml
done
