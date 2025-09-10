# PDL Interpreter: The Execution Engine Behind PDL Patterns

## Overview

PDL Interpreter (`pdl_interp`) is the **intermediate representation** that PDL patterns are compiled to before execution. Understanding PDL Interpreter is crucial for advanced pattern matching and debugging.

**Key Insight**: When you write PDL patterns, they're compiled to PDL Interpreter operations which perform the actual matching and rewriting.

## PDL to PDL Interpreter Compilation Flow

```
PDL Pattern → PDL Interpreter IR → Pattern Execution
```

### Example: How `x + 0 → x` Gets Compiled

**Original PDL Pattern:**
```mlir
pdl.pattern @add_zero : benefit(2) {
  %t = pdl.type
  %x = pdl.operand
  %c0_attr = pdl.attribute = 0 : i32
  %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%t : !pdl.type)
  %c0_res = pdl.result 0 of %c0_op
  %add_op = pdl.operation "arith.addi" (%x, %c0_res : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  
  pdl.rewrite %add_op {
    pdl.replace %add_op with (%x : !pdl.value)
  }
}
```

**Compiled PDL Interpreter IR:**
```mlir
pdl_interp.func @matcher(%root_op : !pdl.operation) {
  // Check if operation is arith.addi
  pdl_interp.check_operation_name of %root_op is "arith.addi" -> ^bb1, ^fail
  
^bb1:
  // Check operand count
  pdl_interp.check_operand_count of %root_op is 2 -> ^bb2, ^fail
  
^bb2:
  // Get second operand (the potential zero)
  %operand1 = pdl_interp.get_operand 1 of %root_op
  
  // Get defining operation of second operand
  %const_op = pdl_interp.get_defining_op of %operand1 : !pdl.value
  pdl_interp.is_not_null %const_op : !pdl.operation -> ^bb3, ^fail
  
^bb3:
  // Check if it's a constant
  pdl_interp.check_operation_name of %const_op is "arith.constant" -> ^bb4, ^fail
  
^bb4:
  // Check constant value is zero
  %value_attr = pdl_interp.get_attribute "value" of %const_op
  pdl_interp.check_attribute %value_attr is 0 : i32 -> ^bb5, ^fail
  
^bb5:
  // Get first operand (x)
  %operand0 = pdl_interp.get_operand 0 of %root_op
  
  // Record successful match and apply rewriter
  pdl_interp.record_match @rewriter(%operand0, %root_op : !pdl.value, !pdl.operation) 
    : benefit(2), root("arith.addi") -> ^fail
  
^fail:
  pdl_interp.finalize
}

pdl_interp.func @rewriter(%x : !pdl.value, %op : !pdl.operation) {
  pdl_interp.replace %op with (%x : !pdl.value)
  pdl_interp.finalize
}
```

## PDL Interpreter Operations Reference

### Matching Operations

#### `pdl_interp.check_operation_name`
Verifies operation type.
```mlir
pdl_interp.check_operation_name of %op is "arith.addi" -> ^success, ^failure
```

#### `pdl_interp.check_operand_count`
Verifies number of operands.
```mlir
pdl_interp.check_operand_count of %op is 2 -> ^success, ^failure
```

#### `pdl_interp.check_result_count`
Verifies number of results.
```mlir
pdl_interp.check_result_count of %op is 1 -> ^success, ^failure
```

#### `pdl_interp.get_operand`
Retrieves specific operand.
```mlir
%operand = pdl_interp.get_operand 0 of %op  // Get first operand
```

#### `pdl_interp.get_result`
Retrieves specific result.
```mlir
%result = pdl_interp.get_result 0 of %op  // Get first result
```

#### `pdl_interp.get_defining_op`
Gets the operation that defines a value.
```mlir
%def_op = pdl_interp.get_defining_op of %value : !pdl.value
```

#### `pdl_interp.get_attribute`
Retrieves named attribute.
```mlir
%attr = pdl_interp.get_attribute "value" of %op
```

#### `pdl_interp.check_attribute`
Verifies attribute value.
```mlir
pdl_interp.check_attribute %attr is 0 : i32 -> ^success, ^failure
```

#### `pdl_interp.is_not_null`
Checks if value/operation exists.
```mlir
pdl_interp.is_not_null %op : !pdl.operation -> ^exists, ^null
```

#### `pdl_interp.are_equal`
Compares two values/types.
```mlir
pdl_interp.are_equal %val1, %val2 : !pdl.value -> ^equal, ^not_equal
```

#### `pdl_interp.get_value_type`
Gets the type of a value.
```mlir
%type = pdl_interp.get_value_type of %value : !pdl.type
```

### Rewriting Operations

#### `pdl_interp.create_operation`
Creates new operation.
```mlir
%new_op = pdl_interp.create_operation "arith.addi"(%x, %y : !pdl.value, !pdl.value) 
  -> (%result_type : !pdl.type)
```

#### `pdl_interp.replace`
Replaces operation with new values.
```mlir
pdl_interp.replace %old_op with (%new_value : !pdl.value)
```

#### `pdl_interp.erase`
Removes operation.
```mlir
pdl_interp.erase %op
```

#### `pdl_interp.finalize`
Terminates pattern execution.
```mlir
pdl_interp.finalize
```

#### `pdl_interp.record_match`
Records successful match and triggers rewriter.
```mlir
pdl_interp.record_match @rewriter_func(%args...) 
  : benefit(2), root("op.name") -> ^next
```

## Advanced Example: Associativity Transformation

Pattern: `(a + b) - c → a + (b - c)`

**PDL Interpreter Implementation:**
```mlir
pdl_interp.func @matcher(%root: !pdl.operation) {
  // Match subtraction operation
  pdl_interp.check_operation_name of %root is "arith.subi" -> ^check_sub, ^fail
  
^check_sub:
  // Get first operand (should be addition result)
  %sub_lhs = pdl_interp.get_operand 0 of %root
  %add_op = pdl_interp.get_defining_op of %sub_lhs : !pdl.value
  pdl_interp.is_not_null %add_op : !pdl.operation -> ^check_add, ^fail
  
^check_add:
  // Verify it's an addition
  pdl_interp.check_operation_name of %add_op is "arith.addi" -> ^extract, ^fail
  
^extract:
  // Extract operands
  %a = pdl_interp.get_operand 0 of %add_op  // a
  %b = pdl_interp.get_operand 1 of %add_op  // b
  %c = pdl_interp.get_operand 1 of %root    // c
  
  // Get types for result
  %type = pdl_interp.get_value_type of %sub_lhs : !pdl.type
  
  // Record match and apply transformation
  pdl_interp.record_match @rewriter(%a, %b, %c, %type, %root : 
    !pdl.value, !pdl.value, !pdl.value, !pdl.type, !pdl.operation)
    : benefit(1), root("arith.subi") -> ^fail
    
^fail:
  pdl_interp.finalize
}

pdl_interp.func @rewriter(%a: !pdl.value, %b: !pdl.value, %c: !pdl.value, 
                          %type: !pdl.type, %root: !pdl.operation) {
  // Create b - c
  %sub_op = pdl_interp.create_operation "arith.subi"(%b, %c : !pdl.value, !pdl.value) 
    -> (%type : !pdl.type)
  %sub_result = pdl_interp.get_result 0 of %sub_op
  
  // Create a + (b - c)
  %add_op = pdl_interp.create_operation "arith.addi"(%a, %sub_result : !pdl.value, !pdl.value)
    -> (%type : !pdl.type)
  %add_result = pdl_interp.get_result 0 of %add_op
  
  // Replace original operation
  pdl_interp.replace %root with (%add_result : !pdl.value)
  pdl_interp.finalize
}
```

## Control Flow in PDL Interpreter

PDL Interpreter uses **basic blocks** and **branching** for control flow:

- **Success path**: Continue to next check
- **Failure path**: Jump to `^fail` block
- **Conditional branches**: Based on check results

### Pattern Matching Strategy

1. **Early exit**: Fail fast on first mismatch
2. **Ordered checks**: Most discriminating checks first
3. **Null checks**: Verify values exist before use
4. **Type safety**: Ensure type compatibility

## Using PDL Interpreter with xDSL

### Running PDL Interpreter Patterns

```bash
# Apply PDL interpreter patterns directly
xdsl-opt input.mlir -p apply-pdl-interp -o output.mlir
```

### Converting PDL to PDL Interpreter

PDL patterns are automatically compiled to PDL Interpreter when using:
```bash
xdsl-opt input.mlir -p apply-pdl
```

The compilation happens internally:
1. PDL patterns are parsed
2. Converted to PDL Interpreter IR
3. PDL Interpreter executes the matching

## Benefits of Understanding PDL Interpreter

1. **Debugging**: Understand why patterns match or don't match
2. **Performance**: Write more efficient patterns
3. **Advanced patterns**: Create complex multi-operation transformations
4. **Custom matchers**: Write PDL Interpreter directly for special cases

## Best Practices

### 1. Efficient Matching Order
```mlir
// Check operation name first (fast)
pdl_interp.check_operation_name of %op is "arith.addi" -> ^next, ^fail

// Then check counts (fast)
pdl_interp.check_operand_count of %op is 2 -> ^next, ^fail

// Then extract and check values (slower)
%val = pdl_interp.get_operand 0 of %op
```

### 2. Minimize Graph Traversal
```mlir
// Cache extracted values
%op1 = pdl_interp.get_defining_op of %val : !pdl.value
// Reuse %op1 for multiple checks instead of re-extracting
```

### 3. Type Checking
```mlir
// Always verify type compatibility
%t1 = pdl_interp.get_value_type of %val1 : !pdl.type
%t2 = pdl_interp.get_value_type of %val2 : !pdl.type
pdl_interp.are_equal %t1, %t2 : !pdl.type -> ^compatible, ^fail
```

## Common Patterns in PDL Interpreter

### Identity Elimination
```mlir
// x op identity → x
// Check for identity constant, replace with other operand
```

### Constant Folding
```mlir
// const1 op const2 → const3
// Extract both constants, compute result, create new constant
```

### Strength Reduction
```mlir
// expensive op → cheaper op
// Match expensive operation, replace with equivalent cheaper one
```

### Associativity/Commutativity
```mlir
// (a op b) op c → a op (b op c)
// Rearrange operations for better optimization opportunities
```

## Debugging PDL Interpreter

### Viewing Generated PDL Interpreter

To see the PDL Interpreter IR generated from your PDL patterns:

```bash
# Compile PDL to PDL Interpreter without applying
xdsl-opt patterns.pdl -p compile-pdl-to-interp -o interp.mlir
```

### Understanding Match Failures

When patterns don't match, trace through the PDL Interpreter:
1. Which check failed?
2. What values were compared?
3. Is the matching order optimal?

## Summary

PDL Interpreter is the **execution engine** of PDL patterns:
- Provides fine-grained control over matching
- Enables complex multi-operation patterns
- Offers direct manipulation of IR structure
- Essential for understanding pattern behavior

Understanding PDL Interpreter helps write:
- More efficient patterns
- Complex transformations
- Better debugging of pattern matching
- Custom optimization passes