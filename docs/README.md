# PDL Pattern Testing Framework for xDSL

This directory contains everything you need to experiment with PDL (Pattern Description Language) rewrites in xDSL.

## Files

- `input.mlir` - Your original test input IR
- `patterns.pdl` - Collection of common arithmetic optimization patterns
- `custom_pattern.pdl` - Your specific x + 0 -> x pattern
- `test_examples.mlir` - Various test cases for different patterns
- `run_pdl.sh` - Script to apply PDL patterns to MLIR files
- `output.mlir` - Generated output after pattern application

## Quick Start

### 1. Test your custom pattern on the original input:
```bash
./run_pdl.sh input.mlir custom_pattern.pdl
```

### 2. Test all patterns on the original input:
```bash
./run_pdl.sh input.mlir patterns.pdl
```

### 3. Test patterns on different examples:
```bash
./run_pdl.sh test_examples.mlir patterns.pdl
```

## Manual Testing

You can also run xdsl-opt directly:

```bash
# Combine pattern and input files
cat custom_pattern.pdl input.mlir > combined.mlir

# Apply patterns
xdsl-opt combined.mlir -p apply-pdl-patterns -o output.mlir

# View output
cat output.mlir
```

## Writing New Patterns

Create a new PDL pattern following this structure:

```mlir
pdl.pattern @pattern_name : benefit(priority) {
  // Match section
  %type = pdl.type
  %operand = pdl.operand
  %operation = pdl.operation "op.name" (...) -> (...)
  
  // Rewrite section
  pdl.rewrite %operation {
    // Transformation logic
    pdl.replace %operation with (...)
  }
}
```

### Key Points:
- Higher benefit values = higher priority
- Use %x for the same operand in multiple places (e.g., x - x)
- Match specific constants with pdl.attribute
- Create new constants in the rewrite section

## Available Patterns in patterns.pdl

1. `@x_plus_zero` - x + 0 → x
2. `@x_minus_x` - x - x → 0
3. `@x_div_x` - x / x → 1.0
4. `@x_times_zero` - x * 0 → 0
5. `@x_times_one` - x * 1 → x
6. `@x_div_one` - x / 1 → x

## Debugging Tips

- Check pattern syntax: Ensure all pdl operations are properly formed
- Verify type matching: f64 constants should use `0.0 : f64` not `0 : f64`
- Pattern priority: Higher benefit patterns apply first
- Use `--print-between-passes` flag for detailed transformation steps

## Expected Results

For your original input:
```mlir
func.func @main(%a : f64, %b : f64, %c : f64) -> f64 {
  %0 = arith.divf %b, %b : f64    // b/b → 1.0
  %1 = arith.subf %a, %a : f64    // a-a → 0.0
  %2 = arith.addf %c, %1 : f64    // c+0 → c
  %3 = arith.divf %2, %0 : f64    // c/1 → c
  func.return %3 : f64
}
```

After applying all patterns, it should simplify to returning %c directly.