// This is a standalone input file - contains NO patterns
// To optimize this file, use:
// xdsl-opt separate_input.mlir -p 'apply-pdl{pdl_file="../patterns/separate_patterns.mlir"}' -o optimized.mlir

func.func @test_integer_optimizations(%arg0: i32, %arg1: i32) -> (i32, i32, i32, i32) {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    
    // These operations will be optimized by patterns
    %r1 = arith.addi %arg0, %zero : i32   // Should become %arg0
    %r2 = arith.muli %arg1, %one : i32    // Should become %arg1
    %r3 = arith.subi %arg0, %arg0 : i32   // Should become 0
    %r4 = arith.muli %arg1, %zero : i32   // Should become 0
    
    func.return %r1, %r2, %r3, %r4 : i32, i32, i32, i32
}

func.func @test_float_optimizations(%x: f64, %y: f64) -> (f64, f64, f64, f64) {
    %zero = arith.constant 0.0 : f64
    %one = arith.constant 1.0 : f64
    
    // These operations will be optimized by patterns
    %r1 = arith.addf %x, %zero : f64      // Should become %x
    %r2 = arith.mulf %y, %one : f64       // Should become %y
    %r3 = arith.subf %x, %x : f64         // Should become 0.0
    %r4 = arith.divf %y, %y : f64         // Should become 1.0
    
    func.return %r1, %r2, %r3, %r4 : f64, f64, f64, f64
}

func.func @complex_expression(%a: i32, %b: i32, %c: i32) -> i32 {
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    
    // Complex expression that should simplify
    %t1 = arith.addi %a, %zero : i32      // → %a
    %t2 = arith.muli %t1, %one : i32      // → %a
    %t3 = arith.subi %b, %b : i32         // → 0
    %t4 = arith.addi %t2, %t3 : i32       // → %a + 0 → %a
    %t5 = arith.muli %c, %zero : i32      // → 0
    %t6 = arith.addi %t4, %t5 : i32       // → %a + 0 → %a
    
    func.return %t6 : i32  // Should ultimately return %a
}