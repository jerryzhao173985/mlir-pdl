#!/bin/bash

# PDL Pattern Testing Script for xDSL
# Usage: ./run_pdl.sh [input_file] [pattern_file]

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default files - adjust paths for repo structure
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

INPUT_FILE="${1:-$REPO_DIR/examples/input.mlir}"
PATTERN_FILE="${2:-$REPO_DIR/patterns/patterns.pdl}"
OUTPUT_FILE="output.mlir"
COMBINED_FILE="combined.mlir"

echo -e "${BLUE}=== xDSL PDL Pattern Testing ===${NC}"
echo ""

# Check if files exist
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file '$INPUT_FILE' not found${NC}"
    exit 1
fi

if [ ! -f "$PATTERN_FILE" ]; then
    echo -e "${RED}Error: Pattern file '$PATTERN_FILE' not found${NC}"
    exit 1
fi

# Display input
echo -e "${YELLOW}Input IR ($INPUT_FILE):${NC}"
cat "$INPUT_FILE"
echo ""

# Display patterns being applied
echo -e "${YELLOW}PDL Patterns ($PATTERN_FILE):${NC}"
grep "pdl.pattern @" "$PATTERN_FILE" | while read -r line; do
    echo "  - $line"
done
echo ""

# Combine pattern and input files
echo -e "${BLUE}Combining pattern and input files...${NC}"
cat "$PATTERN_FILE" > "$COMBINED_FILE"
echo "" >> "$COMBINED_FILE"
cat "$INPUT_FILE" >> "$COMBINED_FILE"

# Apply PDL patterns using xdsl-opt
echo -e "${BLUE}Applying PDL patterns...${NC}"
xdsl-opt "$COMBINED_FILE" -p apply-pdl -o "$OUTPUT_FILE" 2>&1

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Patterns applied successfully${NC}"
    echo ""
    echo -e "${YELLOW}Output IR ($OUTPUT_FILE):${NC}"
    cat "$OUTPUT_FILE"
    echo ""
    
    # Show diff
    echo -e "${YELLOW}Changes:${NC}"
    diff -u "$INPUT_FILE" "$OUTPUT_FILE" --color=always || true
else
    echo -e "${RED}✗ Error applying patterns${NC}"
    exit 1
fi

# Cleanup
rm -f "$COMBINED_FILE"

echo ""
echo -e "${GREEN}Done!${NC}"