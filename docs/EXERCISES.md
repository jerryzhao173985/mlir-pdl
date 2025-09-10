# PDL Pattern Writing Exercises

Practice writing PDL patterns by completing these exercises. Solutions are provided at the bottom.

## Exercise 1: Basic Subtraction
Write a pattern that transforms `x - 0` into `x`.

```mlir
// Your pattern here:
pdl.pattern @exercise1 : benefit(2) {
  // TODO: Complete this pattern
}
```

Test with:
```mlir
func.func @test(%x : f64) -> f64 {
  %c0 = arith.constant 0.0 : f64
  %result = arith.subf %x, %c0 : f64
  func.return %result : f64
}
```

## Exercise 2: Multiplication by Two
Write a pattern that transforms `x * 2` into `x + x`.

```mlir
pdl.pattern @exercise2 : benefit(2) {
  // TODO: Complete this pattern
}
```

## Exercise 3: Zero Division
Write a pattern that transforms `0 / x` into `0`.

```mlir
pdl.pattern @exercise3 : benefit(3) {
  // TODO: Complete this pattern
}
```

## Exercise 4: Negation by Multiplication
Write a pattern that transforms `x * -1` into `-x`.

```mlir
pdl.pattern @exercise4 : benefit(3) {
  // TODO: Complete this pattern
}
```

## Exercise 5: Complex Pattern
Write a pattern that transforms `(x + y) - y` into `x`.

```mlir
pdl.pattern @exercise5 : benefit(4) {
  // TODO: Complete this pattern
}
```

## Exercise 6: Integer XOR
Write a pattern that transforms `x ^ x` (XOR with itself) into `0`.

```mlir
pdl.pattern @exercise6 : benefit(3) {
  // TODO: Complete this pattern for i32 type
}
```

## Exercise 7: Double Pattern Application
Write two patterns that work together:
1. First pattern: `x + x` -> `2 * x`
2. Second pattern: `2 * x` -> `x << 1` (left shift by 1)

```mlir
pdl.pattern @exercise7a : benefit(2) {
  // TODO: x + x -> 2 * x
}

pdl.pattern @exercise7b : benefit(3) {
  // TODO: 2 * x -> x << 1
}
```

## Exercise 8: Conditional Pattern
Write a pattern that only applies to specific types. Transform `x & 0` to `0` only for i32 types.

```mlir
pdl.pattern @exercise8 : benefit(3) {
  // TODO: Complete for i32 only
}
```

---

## Solutions

### Exercise 1 Solution
```mlir
pdl.pattern @exercise1 : benefit(2) {
  %t = pdl.type
  %x = pdl.operand
  %c0_attr = pdl.attribute = 0.0 : f64
  %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%t : !pdl.type)
  %c0 = pdl.result 0 of %c0_op
  %sub = pdl.operation "arith.subf" (%x, %c0 : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  
  pdl.rewrite %sub {
    pdl.replace %sub with (%x : !pdl.value)
  }
}
```

### Exercise 2 Solution
```mlir
pdl.pattern @exercise2 : benefit(2) {
  %t = pdl.type
  %x = pdl.operand
  %c2_attr = pdl.attribute = 2.0 : f64
  %c2_op = pdl.operation "arith.constant" {"value" = %c2_attr} -> (%t : !pdl.type)
  %c2 = pdl.result 0 of %c2_op
  %mul = pdl.operation "arith.mulf" (%x, %c2 : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  
  pdl.rewrite %mul {
    %add = pdl.operation "arith.addf" (%x, %x : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    %add_res = pdl.result 0 of %add
    pdl.replace %mul with (%add_res : !pdl.value)
  }
}
```

### Exercise 3 Solution
```mlir
pdl.pattern @exercise3 : benefit(3) {
  %t = pdl.type
  %x = pdl.operand
  %c0_attr = pdl.attribute = 0.0 : f64
  %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%t : !pdl.type)
  %c0 = pdl.result 0 of %c0_op
  %div = pdl.operation "arith.divf" (%c0, %x : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  
  pdl.rewrite %div {
    pdl.replace %div with (%c0 : !pdl.value)
  }
}
```

### Exercise 4 Solution
```mlir
pdl.pattern @exercise4 : benefit(3) {
  %t = pdl.type
  %x = pdl.operand
  %neg1_attr = pdl.attribute = -1.0 : f64
  %neg1_op = pdl.operation "arith.constant" {"value" = %neg1_attr} -> (%t : !pdl.type)
  %neg1 = pdl.result 0 of %neg1_op
  %mul = pdl.operation "arith.mulf" (%x, %neg1 : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  
  pdl.rewrite %mul {
    %neg = pdl.operation "arith.negf" (%x : !pdl.value) -> (%t : !pdl.type)
    %neg_res = pdl.result 0 of %neg
    pdl.replace %mul with (%neg_res : !pdl.value)
  }
}
```

### Exercise 5 Solution
```mlir
pdl.pattern @exercise5 : benefit(4) {
  %t = pdl.type
  %x = pdl.operand
  %y = pdl.operand
  
  // Match x + y
  %add = pdl.operation "arith.addf" (%x, %y : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  %add_res = pdl.result 0 of %add
  
  // Match (x + y) - y
  %sub = pdl.operation "arith.subf" (%add_res, %y : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  
  pdl.rewrite %sub {
    pdl.replace %sub with (%x : !pdl.value)
  }
}
```

### Exercise 6 Solution
```mlir
pdl.pattern @exercise6 : benefit(3) {
  %t = pdl.type
  %x = pdl.operand
  %xor = pdl.operation "arith.xori" (%x, %x : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  
  pdl.rewrite %xor {
    %c0_attr = pdl.attribute = 0 : i32
    %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%t : !pdl.type)
    %c0 = pdl.result 0 of %c0_op
    pdl.replace %xor with (%c0 : !pdl.value)
  }
}
```

### Exercise 7 Solution
```mlir
pdl.pattern @exercise7a : benefit(2) {
  %t = pdl.type
  %x = pdl.operand
  %add = pdl.operation "arith.addi" (%x, %x : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  
  pdl.rewrite %add {
    %c2_attr = pdl.attribute = 2 : i32
    %c2_op = pdl.operation "arith.constant" {"value" = %c2_attr} -> (%t : !pdl.type)
    %c2 = pdl.result 0 of %c2_op
    %mul = pdl.operation "arith.muli" (%x, %c2 : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    %mul_res = pdl.result 0 of %mul
    pdl.replace %add with (%mul_res : !pdl.value)
  }
}

pdl.pattern @exercise7b : benefit(3) {
  %t = pdl.type
  %x = pdl.operand
  %c2_attr = pdl.attribute = 2 : i32
  %c2_op = pdl.operation "arith.constant" {"value" = %c2_attr} -> (%t : !pdl.type)
  %c2 = pdl.result 0 of %c2_op
  %mul = pdl.operation "arith.muli" (%x, %c2 : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  
  pdl.rewrite %mul {
    %c1_attr = pdl.attribute = 1 : i32
    %c1_op = pdl.operation "arith.constant" {"value" = %c1_attr} -> (%t : !pdl.type)
    %c1 = pdl.result 0 of %c1_op
    %shift = pdl.operation "arith.shli" (%x, %c1 : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    %shift_res = pdl.result 0 of %shift
    pdl.replace %mul with (%shift_res : !pdl.value)
  }
}
```

### Exercise 8 Solution
```mlir
pdl.pattern @exercise8 : benefit(3) {
  %t = pdl.type : i32  // Constraint to i32 type
  %x = pdl.operand
  %c0_attr = pdl.attribute = 0 : i32
  %c0_op = pdl.operation "arith.constant" {"value" = %c0_attr} -> (%t : !pdl.type)
  %c0 = pdl.result 0 of %c0_op
  %and = pdl.operation "arith.andi" (%x, %c0 : !pdl.value, !pdl.value) -> (%t : !pdl.type)
  
  pdl.rewrite %and {
    pdl.replace %and with (%c0 : !pdl.value)
  }
}
```