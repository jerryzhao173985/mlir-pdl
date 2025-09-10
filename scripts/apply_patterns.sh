#!/bin/bash

# Apply PDL Patterns from Separate Files
# Usage: ./apply_patterns.sh <input.mlir> <patterns.mlir> [output.mlir]

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Not enough arguments${NC}"
    echo "Usage: $0 <input.mlir> <patterns.mlir> [output.mlir]"
    echo ""
    echo "Example:"
    echo "  $0 program.mlir patterns.mlir optimized.mlir"
    exit 1
fi

INPUT_FILE="$1"
PATTERN_FILE="$2"
OUTPUT_FILE="${3:-optimized.mlir}"

# Check if files exist
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file '$INPUT_FILE' not found${NC}"
    exit 1
fi

if [ ! -f "$PATTERN_FILE" ]; then
    echo -e "${RED}Error: Pattern file '$PATTERN_FILE' not found${NC}"
    exit 1
fi

echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo -e "${MAGENTA}            PDL Pattern Application Tool${NC}"
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo ""

# Display input file info
echo -e "${YELLOW}📄 Input File:${NC} $INPUT_FILE"
echo -e "${BLUE}Content Preview:${NC}"
head -n 10 "$INPUT_FILE" | sed 's/^/  /'
if [ $(wc -l < "$INPUT_FILE") -gt 10 ]; then
    echo "  ..."
fi
echo ""

# Display pattern file info
echo -e "${YELLOW}🔧 Pattern File:${NC} $PATTERN_FILE"
echo -e "${BLUE}Patterns Found:${NC}"
grep "pdl.pattern @" "$PATTERN_FILE" | while read -r line; do
    pattern_name=$(echo "$line" | sed -n 's/.*@\([^ ]*\).*/\1/p')
    benefit=$(echo "$line" | sed -n 's/.*benefit(\([0-9]*\)).*/\1/p')
    echo -e "  • ${GREEN}$pattern_name${NC} (benefit: $benefit)"
done
echo ""

# Apply patterns using xdsl-opt
echo -e "${YELLOW}⚙️  Applying Patterns...${NC}"
echo -e "${BLUE}Command:${NC} xdsl-opt \"$INPUT_FILE\" -p 'apply-pdl{pdl_file=\"$PATTERN_FILE\"}' -o \"$OUTPUT_FILE\""
echo ""

# Run the optimization
if xdsl-opt "$INPUT_FILE" -p "apply-pdl{pdl_file=\"$PATTERN_FILE\"}" -o "$OUTPUT_FILE" 2>&1; then
    echo -e "${GREEN}✓ Patterns applied successfully!${NC}"
    echo ""
    
    # Show output preview
    echo -e "${YELLOW}📄 Output File:${NC} $OUTPUT_FILE"
    echo -e "${BLUE}Optimized Content Preview:${NC}"
    head -n 15 "$OUTPUT_FILE" | sed 's/^/  /'
    if [ $(wc -l < "$OUTPUT_FILE") -gt 15 ]; then
        echo "  ..."
    fi
    echo ""
    
    # Count operations before and after
    ops_before=$(grep -c "arith\." "$INPUT_FILE" || echo 0)
    ops_after=$(grep -c "arith\." "$OUTPUT_FILE" || echo 0)
    
    if [ $ops_before -gt $ops_after ]; then
        reduction=$((ops_before - ops_after))
        percent=$((reduction * 100 / ops_before))
        echo -e "${GREEN}📊 Optimization Statistics:${NC}"
        echo -e "  Operations before: $ops_before"
        echo -e "  Operations after:  $ops_after"
        echo -e "  Reduced by:       ${GREEN}$reduction operations ($percent%)${NC}"
    else
        echo -e "${BLUE}📊 Statistics:${NC}"
        echo -e "  Operations: $ops_before → $ops_after"
    fi
    
else
    echo -e "${RED}✗ Error applying patterns${NC}"
    exit 1
fi

echo ""
echo -e "${MAGENTA}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Done! Output saved to: $OUTPUT_FILE${NC}"