#!/bin/bash
source /Users/michaelvolk/miniconda3/bin/activate torchcell

FILE_PATH=$1
OUTPUT_FORMAT=${2:-png}  # Default to png if no format specified
FILE_NAME=$(basename -- "$FILE_PATH")
FILE_NAME_WITHOUT_EXT="${FILE_NAME%.*}"
DATE_TIME=$(date "+%Y.%m.%d.%H.%M.%S")
OUTPUT_DIR="notes/assets/images"

pyreverse -o "$OUTPUT_FORMAT" -p MyProjectName "$FILE_PATH"

# Move classes diagram if it exists
if [ -f "classes_MyProjectName.$OUTPUT_FORMAT" ]; then
    mv "classes_MyProjectName.$OUTPUT_FORMAT" "$OUTPUT_DIR/${FILE_NAME_WITHOUT_EXT}-${DATE_TIME}-classes.$OUTPUT_FORMAT"
    echo "Created: $OUTPUT_DIR/${FILE_NAME_WITHOUT_EXT}-${DATE_TIME}-classes.$OUTPUT_FORMAT"
else
    echo "Warning: classes_MyProjectName.$OUTPUT_FORMAT not generated"
fi

# Move packages diagram if it exists
if [ -f "packages_MyProjectName.$OUTPUT_FORMAT" ]; then
    mv "packages_MyProjectName.$OUTPUT_FORMAT" "$OUTPUT_DIR/${FILE_NAME_WITHOUT_EXT}-${DATE_TIME}-packages.$OUTPUT_FORMAT"
    echo "Created: $OUTPUT_DIR/${FILE_NAME_WITHOUT_EXT}-${DATE_TIME}-packages.$OUTPUT_FORMAT"
else
    echo "Warning: packages_MyProjectName.$OUTPUT_FORMAT not generated"
fi
