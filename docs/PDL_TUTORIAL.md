# Complete PDL (Pattern Description Language) Tutorial for xDSL

## Table of Contents
1. [Basic xdsl-opt Usage](#basic-xdsl-opt-usage)
2. [PDL Core Concepts](#pdl-core-concepts)
3. [Pattern Matching Operations](#pattern-matching-operations)
4. [Rewrite Operations](#rewrite-operations)
5. [Advanced Patterns](#advanced-patterns)
6. [Testing Workflow](#testing-workflow)

## Basic xdsl-opt Usage

### Direct Command Line Usage

```bash
# Basic syntax
xdsl-opt [input_file] -p [pass_name] -o [output_file]

# Apply PDL patterns
xdsl-opt combined.mlir -p apply-pdl -o output.mlir

# Multiple passes in sequence
xdsl-opt input.mlir -p "pass1,pass2,pass3" -o output.mlir

# Print intermediate results
xdsl-opt input.mlir -p apply-pdl --print-between-passes

# Available PDL-related passes
xdsl-opt --help | grep pdl
# - apply-pdl: Apply PDL patterns
# - apply-pdl-interp: Apply PDL patterns with interpreter
# - arith-add-fastmath: Add fastmath flags to arith ops
```

### File Combination for PDL

PDL patterns must be in the same file as the IR to transform:
```bash
# Combine files
cat patterns.pdl input.mlir > combined.mlir

# Apply patterns
xdsl-opt combined.mlir -p apply-pdl -o output.mlir
```

## PDL Core Concepts

### 1. Pattern Structure

```mlir
pdl.pattern @pattern_name : benefit(priority) {
    // MATCH SECTION - What to look for
    %type = pdl.type                    // Match any type
    %operand = pdl.operand               // Match any operand
    %attribute = pdl.attribute = value   // Match specific attribute
    %operation = pdl.operation "op.name" // Match specific operation
    
    // REWRITE SECTION - How to transform
    pdl.rewrite %operation {
        // Transformation logic
    }
}
```

### 2. Benefit Priority

- Higher benefit = higher priority
- Range: typically 1-10
- When multiple patterns match, highest benefit wins

## Pattern Matching Operations

### pdl.type
Matches any MLIR type (f64, i32, tensor<>, etc.)
```mlir
%t = pdl.type  // Will match f64, i32, or any other type
```

### pdl.operand
Matches operation inputs
```mlir
%x = pdl.operand  // Single operand
// Use same %x multiple times to match same value
%op = pdl.operation "arith.subf" (%x, %x : !pdl.value, !pdl.value)
```

### pdl.attribute
Matches or creates attributes (constants)
```mlir
// Match specific constant value
%zero = pdl.attribute = 0.0 : f64
%one = pdl.attribute = 1.0 : f64
%neg_one = pdl.attribute = -1.0 : f64
```

### pdl.operation
Matches specific operations with constraints
```mlir
// Match operation with specific name
%add_op = pdl.operation "arith.addf"

// Match with operands
%mul_op = pdl.operation "arith.mulf" (%x, %y : !pdl.value, !pdl.value) -> (%t : !pdl.type)

// Match with attributes
%const_op = pdl.operation "arith.constant" {"value" = %attr} -> (%t : !pdl.type)
```

### pdl.result
Extracts results from matched operations
```mlir
%const_op = pdl.operation "arith.constant" {"value" = %zero} -> (%t : !pdl.type)
%const_result = pdl.result 0 of %const_op  // Get first (0-indexed) result
```

## Rewrite Operations

### pdl.replace
Replace matched operation with new values
```mlir
pdl.rewrite %matched_op {
    // Replace with existing value
    pdl.replace %matched_op with (%x : !pdl.value)
    
    // Replace with newly created operation's result
    %new_op = pdl.operation "arith.constant" {"value" = %attr} -> (%t : !pdl.type)
    %new_result = pdl.result 0 of %new_op
    pdl.replace %matched_op with (%new_result : !pdl.value)
}
```

### pdl.erase
Remove operations without replacement
```mlir
pdl.rewrite %dead_op {
    pdl.erase %dead_op
}
```

### Creating New Operations in Rewrite
```mlir
pdl.rewrite %op {
    // Create new attribute
    %new_attr = pdl.attribute = 2.0 : f64
    
    // Create new operation
    %new_op = pdl.operation "arith.mulf" (%x, %x : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    
    // Get result and use it
    %result = pdl.result 0 of %new_op
    pdl.replace %op with (%result : !pdl.value)
}
```

## Advanced Patterns

### 1. Multiple Operand Matching
```mlir
// Match different operands
%x = pdl.operand
%y = pdl.operand
%add = pdl.operation "arith.addf" (%x, %y : !pdl.value, !pdl.value)

// Match same operand twice (x + x)
%x = pdl.operand
%double = pdl.operation "arith.addf" (%x, %x : !pdl.value, !pdl.value)
```

### 2. Commutative Patterns
For commutative operations, you may need two patterns:
```mlir
// Pattern 1: x + 0
pdl.pattern @add_zero_left : benefit(2) {
    %t = pdl.type
    %x = pdl.operand
    %zero_attr = pdl.attribute = 0.0 : f64
    %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%t : !pdl.type)
    %zero = pdl.result 0 of %zero_op
    %add = pdl.operation "arith.addf" (%x, %zero : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    pdl.rewrite %add {
        pdl.replace %add with (%x : !pdl.value)
    }
}

// Pattern 2: 0 + x
pdl.pattern @add_zero_right : benefit(2) {
    %t = pdl.type
    %x = pdl.operand
    %zero_attr = pdl.attribute = 0.0 : f64
    %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%t : !pdl.type)
    %zero = pdl.result 0 of %zero_op
    %add = pdl.operation "arith.addf" (%zero, %x : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    pdl.rewrite %add {
        pdl.replace %add with (%x : !pdl.value)
    }
}
```

### 3. Chain Patterns
Match operations that use results of other operations:
```mlir
pdl.pattern @fold_double_neg : benefit(4) {
    %t = pdl.type
    %x = pdl.operand
    
    // First negation
    %neg1 = pdl.operation "arith.negf" (%x : !pdl.value) -> (%t : !pdl.type)
    %neg1_result = pdl.result 0 of %neg1
    
    // Second negation using first's result
    %neg2 = pdl.operation "arith.negf" (%neg1_result : !pdl.value) -> (%t : !pdl.type)
    
    pdl.rewrite %neg2 {
        pdl.replace %neg2 with (%x : !pdl.value)
    }
}
```

## Testing Workflow

### 1. Create Test Input
```mlir
// test_input.mlir
func.func @test(%x : f64, %y : f64) -> f64 {
    %0 = arith.subf %x, %x : f64
    %1 = arith.addf %y, %0 : f64
    func.return %1 : f64
}
```

### 2. Create Pattern File
```mlir
// my_patterns.pdl
pdl.pattern @simplify : benefit(3) {
    // Your pattern here
}
```

### 3. Test Pattern
```bash
# Using the script
./run_pdl.sh test_input.mlir my_patterns.pdl

# Or manually
cat my_patterns.pdl test_input.mlir > combined.mlir
xdsl-opt combined.mlir -p apply-pdl -o result.mlir
cat result.mlir
```

### 4. Debug Pattern Matching
```bash
# See intermediate transformations
xdsl-opt combined.mlir -p apply-pdl --print-between-passes

# Check if pattern syntax is valid
xdsl-opt my_patterns.pdl --verify-diagnostics
```

## Common Pitfalls

1. **Type Mismatches**: Ensure constants match the operation types (0.0 : f64 not 0 : f64)
2. **Operation Names**: Use correct operation names (arith.addf not add)
3. **Operand Count**: Match the exact number of operands the operation expects
4. **Result Indexing**: Results are 0-indexed
5. **Pattern Order**: Higher benefit patterns apply first, potentially preventing others

## Practice Exercises

1. Write a pattern for: x * 2 → x + x
2. Write a pattern for: x - 0 → x
3. Write a pattern for: 0 / x → 0
4. Write a pattern for: x * -1 → -x
5. Write a pattern to fold: (x + y) - y → x

Each pattern teaches different PDL concepts!