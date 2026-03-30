#!/bin/bash
SRC_DIR="/hpcwork/ab123456/mimic_cxr_jpg_subset_with_labs"
DST_DIR="./data"
AMOUNT=101

head -n $AMOUNT "$SRC_DIR/annotations.csv" > "$DST_DIR/annotations.csv"
zcat "$SRC_DIR/mimic-cxr-2.0.0-metadata.csv.gz" | head -n $AMOUNT > "$DST_DIR/metadata.csv"
tail -n+2 "$DST_DIR/annotations.csv" | cut -d',' -f18 | awk "{print \"$SRC_DIR/images/\" \$0}" | xargs cp -t "$DST_DIR/img"