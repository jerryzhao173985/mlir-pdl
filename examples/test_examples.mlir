// Test cases for various PDL patterns

// Test case 1: x + 0 optimization
func.func @test_add_zero(%x : f64) -> f64 {
  %c0 = arith.constant 0.0 : f64
  %result = arith.addf %x, %c0 : f64
  func.return %result : f64
}

// Test case 2: x - x optimization  
func.func @test_sub_self(%x : f64) -> f64 {
  %result = arith.subf %x, %x : f64
  func.return %result : f64
}

// Test case 3: x / x optimization
func.func @test_div_self(%x : f64) -> f64 {
  %result = arith.divf %x, %x : f64
  func.return %result : f64
}

// Test case 4: x * 0 optimization
func.func @test_mul_zero(%x : f64) -> f64 {
  %c0 = arith.constant 0.0 : f64
  %result = arith.mulf %x, %c0 : f64
  func.return %result : f64
}

// Test case 5: x * 1 optimization
func.func @test_mul_one(%x : f64) -> f64 {
  %c1 = arith.constant 1.0 : f64
  %result = arith.mulf %x, %c1 : f64
  func.return %result : f64
}

// Test case 6: Complex expression
func.func @test_complex(%a : f64, %b : f64) -> f64 {
  %c0 = arith.constant 0.0 : f64
  %c1 = arith.constant 1.0 : f64
  %t1 = arith.subf %a, %a : f64        // Should become 0
  %t2 = arith.addf %b, %t1 : f64       // Should become b + 0 = b
  %t3 = arith.mulf %t2, %c1 : f64      // Should become b * 1 = b
  %t4 = arith.divf %t3, %c1 : f64      // Should become b / 1 = b
  func.return %t4 : f64
}