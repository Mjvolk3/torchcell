#!/bin/bash

echo "Checking for large PNG files..."
large_files=0
compressed_files=0

find . -name "*.png" -print0 | while IFS= read -r -d $'\0' file; do
    size=$(stat -f%z "$file")         # macOS uses -f%z
    if [ "$size" -gt 52428800 ]; then # 50MB in bytes
        large_files=$((large_files + 1))
        size_mb=$(echo "scale=2; $size/1048576" | bc)
        echo "Found large file: $file ($size_mb MB)"
        echo "Compressing..."
        convert "$file" -quality 85 "${file%.png}_tmp.png"
        mv "${file%.png}_tmp.png" "$file"
        git add "$file"

        new_size=$(stat -f%z "$file")
        new_size_mb=$(echo "scale=2; $new_size/1048576" | bc)
        echo "✓ Compressed from $size_mb MB to $new_size_mb MB"
        compressed_files=$((compressed_files + 1))
    fi
done

echo "Summary:"
echo "- Files checked: $(find . -name "*.png" | wc -l)"
echo "- Large files found: $large_files"
echo "- Files compressed: $compressed_files"

# Check for any remaining large files
remaining_large=$(find . -name "*.png" -size +50M -print | wc -l)
if [[ $remaining_large -gt 0 ]]; then
    echo "❌ Error: $remaining_large files still exceed 50MB. Commit aborted."
    exit 1
fi

if [[ $compressed_files -gt 0 ]]; then
    echo "✓ Successfully compressed all large PNG files"
fi

exit 0
