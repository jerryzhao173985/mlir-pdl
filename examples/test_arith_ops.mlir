// Test cases for arithmetic operations to test PDL patterns

// Test: x - 0 should become x
func.func @test_sub_zero(%x : f64) -> f64 {
  %c0 = arith.constant 0.0 : f64
  %result = arith.subf %x, %c0 : f64
  func.return %result : f64
}

// Test: 0 - x should become -x
func.func @test_zero_sub(%x : f64) -> f64 {
  %c0 = arith.constant 0.0 : f64
  %result = arith.subf %c0, %x : f64
  func.return %result : f64
}

// Test: x * 2 should become x + x
func.func @test_mul_two(%x : f64) -> f64 {
  %c2 = arith.constant 2.0 : f64
  %result = arith.mulf %x, %c2 : f64
  func.return %result : f64
}

// Test: x * -1 should become -x
func.func @test_mul_neg_one(%x : f64) -> f64 {
  %neg1 = arith.constant -1.0 : f64
  %result = arith.mulf %x, %neg1 : f64
  func.return %result : f64
}

// Test: 0 / x should become 0
func.func @test_zero_div(%x : f64) -> f64 {
  %c0 = arith.constant 0.0 : f64
  %result = arith.divf %c0, %x : f64
  func.return %result : f64
}

// Test: x / -1 should become -x
func.func @test_div_neg_one(%x : f64) -> f64 {
  %neg1 = arith.constant -1.0 : f64
  %result = arith.divf %x, %neg1 : f64
  func.return %result : f64
}

// Test: -(-x) should become x
func.func @test_double_neg(%x : f64) -> f64 {
  %neg1 = arith.negf %x : f64
  %neg2 = arith.negf %neg1 : f64
  func.return %neg2 : f64
}

// Test: (x + y) - y should become x
func.func @test_add_then_sub(%x : f64, %y : f64) -> f64 {
  %add = arith.addf %x, %y : f64
  %result = arith.subf %add, %y : f64
  func.return %result : f64
}

// Test: (x - y) + y should become x
func.func @test_sub_then_add(%x : f64, %y : f64) -> f64 {
  %sub = arith.subf %x, %y : f64
  %result = arith.addf %sub, %y : f64
  func.return %result : f64
}

// Test: Complex expression combining multiple patterns
func.func @test_complex_optimization(%a : f64, %b : f64) -> f64 {
  %c0 = arith.constant 0.0 : f64
  %c1 = arith.constant 1.0 : f64
  %c2 = arith.constant 2.0 : f64
  %neg1 = arith.constant -1.0 : f64
  
  // These should all simplify
  %t1 = arith.subf %a, %a : f64         // -> 0
  %t2 = arith.addf %b, %t1 : f64        // -> b + 0 = b
  %t3 = arith.mulf %t2, %c2 : f64       // -> b * 2 = b + b
  %t4 = arith.divf %t3, %c1 : f64       // -> (b + b) / 1 = b + b
  %t5 = arith.mulf %t4, %neg1 : f64     // -> (b + b) * -1 = -(b + b)
  %t6 = arith.negf %t5 : f64            // -> --(b + b) = b + b
  %t7 = arith.subf %t6, %c0 : f64       // -> (b + b) - 0 = b + b
  
  func.return %t7 : f64
}

// Test: Integer operations
func.func @test_integer_ops(%x : i32, %y : i32) -> (i32, i32, i32) {
  %c0 = arith.constant 0 : i32
  %neg1 = arith.constant -1 : i32
  
  %and_zero = arith.andi %x, %c0 : i32      // -> 0
  %or_neg1 = arith.ori %x, %neg1 : i32      // -> -1
  %xor_self = arith.xori %y, %y : i32       // -> 0
  
  func.return %and_zero, %or_neg1, %xor_self : i32, i32, i32
}

// Test: Multiple uses - patterns should still apply
func.func @test_multiple_uses(%x : f64, %y : f64) -> (f64, f64) {
  %sub = arith.subf %x, %x : f64   // -> 0
  %add1 = arith.addf %y, %sub : f64 // -> y + 0 = y
  %add2 = arith.addf %sub, %y : f64 // -> 0 + y = y
  func.return %add1, %add2 : f64, f64
}

// Test: Nested patterns
func.func @test_nested(%x : f64) -> f64 {
  %c0 = arith.constant 0.0 : f64
  %c1 = arith.constant 1.0 : f64
  %c2 = arith.constant 2.0 : f64
  
  // Create nested computation that should simplify
  %inner1 = arith.subf %x, %x : f64        // -> 0
  %inner2 = arith.addf %inner1, %c0 : f64  // -> 0 + 0 = 0
  %inner3 = arith.mulf %x, %c1 : f64       // -> x * 1 = x
  %inner4 = arith.addf %inner3, %inner2 : f64 // -> x + 0 = x
  %result = arith.divf %inner4, %c1 : f64  // -> x / 1 = x
  
  func.return %result : f64
}