# PDL Dialect Reference for xDSL

Based on the official xDSL documentation and MLIR PDL dialect specification.

## PDL Operations

### Core Matching Operations

#### `pdl.type`
Defines a type placeholder that matches any MLIR type.
```mlir
%t = pdl.type
```
Can also specify concrete types:
```mlir
%t = pdl.type : i32
%t = pdl.type : f64
```

#### `pdl.operand`
Matches an SSA value that will be used as an operand.
```mlir
%x = pdl.operand
```
- Same operand name used multiple times matches the same value
- Example: `(%x, %x)` matches operations where both operands are identical

#### `pdl.attribute`
Matches or creates an attribute (typically constants).
```mlir
%attr = pdl.attribute = 0.0 : f64
%attr = pdl.attribute = 42 : i32
%attr = pdl.attribute = -1.0 : f64
```

#### `pdl.operation`
Matches a specific operation with constraints.
```mlir
// Basic operation match
%op = pdl.operation "arith.addf"

// With operands
%op = pdl.operation "arith.mulf" (%x, %y : !pdl.value, !pdl.value) -> (%t : !pdl.type)

// With attributes
%op = pdl.operation "arith.constant" {"value" = %attr} -> (%t : !pdl.type)

// With both operands and results
%op = pdl.operation "arith.divf" (%x, %y : !pdl.value, !pdl.value) -> (%result_type : !pdl.type)
```

#### `pdl.result`
Extracts a result from a matched operation.
```mlir
%res = pdl.result 0 of %op  // Get first result (0-indexed)
```

### Rewrite Operations

#### `pdl.rewrite`
Begins the rewrite section of a pattern.
```mlir
pdl.rewrite %matched_op {
  // Rewrite operations here
}
```

#### `pdl.replace`
Replaces the matched operation with new values.
```mlir
// Replace with existing value
pdl.replace %op with (%x : !pdl.value)

// Replace with multiple values
pdl.replace %op with (%x, %y : !pdl.value, !pdl.value)

// Replace with result of new operation
pdl.replace %op with (%new_result : !pdl.value)
```

#### `pdl.erase`
Removes an operation without replacement.
```mlir
pdl.erase %op
```

### Pattern Definition

#### `pdl.pattern`
Defines a complete rewrite pattern.
```mlir
pdl.pattern @pattern_name : benefit(priority) {
  // Matching region
  %type = pdl.type
  %operand = pdl.operand
  %op = pdl.operation "op.name" (...) -> (...)
  
  // Rewrite region
  pdl.rewrite %op {
    // Transformation
    pdl.replace %op with (...)
  }
}
```

- `@pattern_name`: Optional pattern identifier
- `benefit(priority)`: Higher values = higher priority (typically 1-10)

## Type System

### PDL Types

- `!pdl.type` - Represents an MLIR type placeholder
- `!pdl.value` - Represents an SSA value placeholder
- `!pdl.operation` - Represents an operation placeholder
- `!pdl.attribute` - Represents an attribute placeholder

## Pattern Writing Best Practices

### 1. Type Consistency
Ensure constant types match operation types:
```mlir
// Correct for f64 operations
%c0_attr = pdl.attribute = 0.0 : f64

// Correct for i32 operations
%c0_attr = pdl.attribute = 0 : i32
```

### 2. Operation Names
Use fully qualified operation names:
- `arith.addf` - Floating-point addition
- `arith.addi` - Integer addition
- `arith.subf` - Floating-point subtraction
- `arith.subi` - Integer subtraction
- `arith.mulf` - Floating-point multiplication
- `arith.muli` - Integer multiplication
- `arith.divf` - Floating-point division
- `arith.divi` - Integer division
- `arith.negf` - Floating-point negation

### 3. Benefit Values
Common benefit value guidelines:
- 1-2: Simple identity transformations (x + 0 → x)
- 3-4: Self-operations (x - x → 0, x / x → 1)
- 4-5: Complex multi-operation patterns
- 5+: Critical optimizations

### 4. Pattern Order
Patterns with higher benefit values are applied first when multiple patterns match.

### 5. Operand Matching
- Use the same variable name to match identical operands
- Use different variable names for potentially different operands

```mlir
// Matches x - x (same operand)
%x = pdl.operand
%op = pdl.operation "arith.subf" (%x, %x : !pdl.value, !pdl.value)

// Matches x - y (different operands)
%x = pdl.operand
%y = pdl.operand
%op = pdl.operation "arith.subf" (%x, %y : !pdl.value, !pdl.value)
```

## Common Pitfalls and Solutions

### 1. Type Mismatches
**Problem**: Using integer constants for floating-point operations
```mlir
// Wrong
%c0_attr = pdl.attribute = 0 : i32  // For arith.addf
```
**Solution**: Match types correctly
```mlir
// Correct
%c0_attr = pdl.attribute = 0.0 : f64  // For arith.addf
```

### 2. Missing Result Extraction
**Problem**: Trying to use operation directly instead of its result
```mlir
// Wrong
pdl.replace %op with (%constant_op : !pdl.value)
```
**Solution**: Extract result first
```mlir
// Correct
%result = pdl.result 0 of %constant_op
pdl.replace %op with (%result : !pdl.value)
```

### 3. Incorrect Attribute Syntax
**Problem**: Wrong attribute specification
```mlir
// Wrong
%op = pdl.operation "arith.constant" {%attr} -> (%t : !pdl.type)
```
**Solution**: Use proper attribute syntax
```mlir
// Correct
%op = pdl.operation "arith.constant" {"value" = %attr} -> (%t : !pdl.type)
```

## Advanced Patterns

### Multi-Operation Matching
Match chains of operations:
```mlir
pdl.pattern @complex : benefit(4) {
  %t = pdl.type
  %x = pdl.operand
  %y = pdl.operand
  
  // First operation
  %add = pdl.operation "arith.addf" (%x, %y : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  %add_res = pdl.result 0 of %add
  
  // Second operation using first's result
  %sub = pdl.operation "arith.subf" (%add_res, %y : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  
  pdl.rewrite %sub {
    pdl.replace %sub with (%x : !pdl.value)
  }
}
```

### Creating Multiple Operations in Rewrite
```mlir
pdl.rewrite %op {
  // Create first operation
  %c1_attr = pdl.attribute = 1.0 : f64
  %c1_op = pdl.operation "arith.constant" {"value" = %c1_attr} -> (%t : !pdl.type)
  %c1_res = pdl.result 0 of %c1_op
  
  // Create second operation
  %add = pdl.operation "arith.addf" (%x, %c1_res : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  %add_res = pdl.result 0 of %add
  
  // Replace with result
  pdl.replace %op with (%add_res : !pdl.value)
}
```

## References

- [xDSL PDL Tutorial](https://xdsl.readthedocs.io/latest/marimo/html/pdl/)
- [MLIR PDL Dialect](https://mlir.llvm.org/docs/Dialects/PDLOps/)
- [xDSL Documentation](https://xdsl.readthedocs.io/)