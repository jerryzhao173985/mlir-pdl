# The Proper Way to Use PDL in xDSL

## The Power of Separate Files

**This is where PDL truly shines!** You can keep your patterns separate from your input code and apply them dynamically.

## The Two-File Approach

### 1. Pattern File (patterns.mlir)
Contains only PDL patterns:
```mlir
// patterns.mlir - Your rewrite rules
pdl.pattern @optimize_add_zero : benefit(2) {
    %t = pdl.type
    %x = pdl.operand
    %zero_attr = pdl.attribute = 0 : i32
    %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%t : !pdl.type)
    %zero_res = pdl.result 0 of %zero_op
    %add = pdl.operation "arith.addi" (%x, %zero_res : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    
    pdl.rewrite %add {
        pdl.replace %add with (%x : !pdl.value)
    }
}
```

### 2. Input File (program.mlir)
Contains the code to optimize:
```mlir
// program.mlir - Your actual program
func.func @main() -> i32 {
    %x = arith.constant 42 : i32
    %zero = arith.constant 0 : i32
    %result = arith.addi %x, %zero : i32
    func.return %result : i32
}
```

## Command-Line Usage

### The Magic Command
```bash
xdsl-opt program.mlir -p 'apply-pdl{pdl_file="patterns.mlir"}' -o optimized.mlir
```

This command:
1. Loads `program.mlir` as input
2. Loads patterns from `patterns.mlir`
3. Applies the patterns
4. Outputs optimized code to `optimized.mlir`

## Real-World Workflow

### Step 1: Create Your Pattern Library
```bash
# arithmetic_patterns.mlir
pdl.pattern @add_zero : benefit(2) {
    # x + 0 → x
}

pdl.pattern @mul_one : benefit(2) {
    # x * 1 → x
}

pdl.pattern @sub_self : benefit(3) {
    # x - x → 0
}
```

### Step 2: Apply to Any Program
```bash
# Apply to different programs using the same patterns
xdsl-opt program1.mlir -p 'apply-pdl{pdl_file="arithmetic_patterns.mlir"}'
xdsl-opt program2.mlir -p 'apply-pdl{pdl_file="arithmetic_patterns.mlir"}'
xdsl-opt program3.mlir -p 'apply-pdl{pdl_file="arithmetic_patterns.mlir"}'
```

## Advanced Usage

### Multiple Pattern Files
You can organize patterns by category:

```bash
# Create specialized pattern libraries
strength_reduction.mlir    # x * 2 → x + x
constant_folding.mlir      # 2 + 3 → 5
identity_elimination.mlir  # x + 0 → x
algebraic_simplify.mlir    # (a + b) - b → a
```

### Pattern Application Pipeline
```bash
# Apply multiple optimization passes in sequence
xdsl-opt input.mlir \
    -p 'apply-pdl{pdl_file="constant_folding.mlir"}' \
    -p 'apply-pdl{pdl_file="identity_elimination.mlir"}' \
    -p 'apply-pdl{pdl_file="strength_reduction.mlir"}' \
    -o optimized.mlir
```

## Complete Example

### 1. Create Pattern File
```bash
cat > my_patterns.mlir << 'EOF'
// Optimize x + 0 → x
pdl.pattern @add_identity : benefit(2) {
    %t = pdl.type
    %x = pdl.operand
    %zero_attr = pdl.attribute = 0 : i32
    %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%t : !pdl.type)
    %zero = pdl.result 0 of %zero_op
    %add = pdl.operation "arith.addi" (%x, %zero : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    
    pdl.rewrite %add {
        pdl.replace %add with (%x : !pdl.value)
    }
}

// Optimize x - x → 0
pdl.pattern @sub_self : benefit(3) {
    %t = pdl.type
    %x = pdl.operand
    %sub = pdl.operation "arith.subi" (%x, %x : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    
    pdl.rewrite %sub {
        %zero_attr = pdl.attribute = 0 : i32
        %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%t : !pdl.type)
        %zero = pdl.result 0 of %zero_op
        pdl.replace %sub with (%zero : !pdl.value)
    }
}
EOF
```

### 2. Create Input Program
```bash
cat > my_program.mlir << 'EOF'
func.func @compute(%arg0: i32) -> i32 {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    
    // These will be optimized
    %t1 = arith.addi %arg0, %zero : i32  // Will become %arg0
    %t2 = arith.subi %t1, %t1 : i32      // Will become 0
    %t3 = arith.addi %t2, %one : i32     // Will become 0 + 1 = 1
    
    func.return %t3 : i32
}
EOF
```

### 3. Apply Patterns
```bash
xdsl-opt my_program.mlir -p 'apply-pdl{pdl_file="my_patterns.mlir"}' -o optimized.mlir
```

### 4. View Results
```bash
cat optimized.mlir
# Should show optimized code with patterns applied
```

## Why This Approach is Powerful

### 1. **Separation of Concerns**
- Patterns are reusable across projects
- Input code stays clean
- Easy to version control patterns separately

### 2. **Pattern Libraries**
- Build domain-specific optimization libraries
- Share patterns across teams
- Mix and match pattern sets

### 3. **Testing and Debugging**
- Test patterns independently
- Apply subsets of patterns for debugging
- Compare results with different pattern sets

### 4. **Composability**
- Chain multiple pattern files
- Build optimization pipelines
- Conditional pattern application

## Integration with Build Systems

### Makefile Example
```makefile
PATTERNS = patterns/arithmetic.mlir patterns/strength.mlir
SOURCES = $(wildcard src/*.mlir)
OPTIMIZED = $(SOURCES:src/%.mlir=build/%.opt.mlir)

build/%.opt.mlir: src/%.mlir $(PATTERNS)
    xdsl-opt $< -p 'apply-pdl{pdl_file="patterns/arithmetic.mlir"}' -o $@

optimize: $(OPTIMIZED)
```

### Python Script
```python
import subprocess

def optimize_mlir(input_file, pattern_file, output_file):
    cmd = [
        'xdsl-opt',
        input_file,
        '-p', f'apply-pdl{{pdl_file="{pattern_file}"}}',
        '-o', output_file
    ]
    subprocess.run(cmd, check=True)

# Apply patterns to multiple files
for input_file in glob.glob('*.mlir'):
    optimize_mlir(input_file, 'patterns.mlir', f'opt_{input_file}')
```

## Best Practices

### 1. **Organize Patterns by Purpose**
```
patterns/
├── arithmetic/
│   ├── identity.mlir      # Identity eliminations
│   ├── constant.mlir       # Constant folding
│   └── strength.mlir       # Strength reduction
├── memory/
│   ├── load_store.mlir     # Load/store optimizations
│   └── alias.mlir          # Alias analysis patterns
└── domain_specific/
    └── dsp.mlir            # DSP-specific patterns
```

### 2. **Document Pattern Files**
```mlir
// patterns/arithmetic/identity.mlir
// Purpose: Eliminate arithmetic identity operations
// Author: Your Name
// Date: 2024-01-10

// Pattern: x + 0 → x
// Benefit: 2 (medium priority)
// Applicable to: Integer and floating-point additions
pdl.pattern @add_zero : benefit(2) {
    // ... pattern definition
}
```

### 3. **Test Pattern Files**
```bash
# test_patterns.sh
#!/bin/bash

# Test each pattern file individually
for pattern in patterns/*.mlir; do
    echo "Testing $pattern..."
    xdsl-opt test_input.mlir -p "apply-pdl{pdl_file=\"$pattern\"}" > /dev/null
    if [ $? -eq 0 ]; then
        echo "✓ $pattern passed"
    else
        echo "✗ $pattern failed"
    fi
done
```

## Summary

The **proper way** to use PDL in xDSL is:
1. **Keep patterns in separate files** (.mlir files with PDL patterns)
2. **Use the `pdl_file` parameter** in the apply-pdl pass
3. **Apply patterns dynamically** to any input program
4. **Build pattern libraries** for reuse

This is how PDL was designed to be used - as a **modular, reusable pattern system** that can transform any MLIR program!