// Standalone PDL Pattern File
// This file contains ONLY patterns - no input code
// Use with: xdsl-opt input.mlir -p 'apply-pdl{pdl_file="separate_patterns.mlir"}'

// ============================================
// Arithmetic Identity Patterns
// ============================================

// Pattern: x + 0 → x (integer)
pdl.pattern @addi_zero : benefit(2) {
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

// Pattern: x + 0 → x (floating-point)
pdl.pattern @addf_zero : benefit(2) {
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

// Pattern: x * 1 → x (integer)
pdl.pattern @muli_one : benefit(2) {
    %t = pdl.type
    %x = pdl.operand
    %one_attr = pdl.attribute = 1 : i32
    %one_op = pdl.operation "arith.constant" {"value" = %one_attr} -> (%t : !pdl.type)
    %one = pdl.result 0 of %one_op
    %mul = pdl.operation "arith.muli" (%x, %one : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    
    pdl.rewrite %mul {
        pdl.replace %mul with (%x : !pdl.value)
    }
}

// Pattern: x * 1 → x (floating-point)
pdl.pattern @mulf_one : benefit(2) {
    %t = pdl.type
    %x = pdl.operand
    %one_attr = pdl.attribute = 1.0 : f64
    %one_op = pdl.operation "arith.constant" {"value" = %one_attr} -> (%t : !pdl.type)
    %one = pdl.result 0 of %one_op
    %mul = pdl.operation "arith.mulf" (%x, %one : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    
    pdl.rewrite %mul {
        pdl.replace %mul with (%x : !pdl.value)
    }
}

// ============================================
// Self-Operation Patterns
// ============================================

// Pattern: x - x → 0 (integer)
pdl.pattern @subi_self : benefit(3) {
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

// Pattern: x - x → 0 (floating-point)
pdl.pattern @subf_self : benefit(3) {
    %t = pdl.type
    %x = pdl.operand
    %sub = pdl.operation "arith.subf" (%x, %x : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    
    pdl.rewrite %sub {
        %zero_attr = pdl.attribute = 0.0 : f64
        %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%t : !pdl.type)
        %zero = pdl.result 0 of %zero_op
        pdl.replace %sub with (%zero : !pdl.value)
    }
}

// Pattern: x / x → 1 (integer)
pdl.pattern @divi_self : benefit(3) {
    %t = pdl.type
    %x = pdl.operand
    %div = pdl.operation "arith.divsi" (%x, %x : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    
    pdl.rewrite %div {
        %one_attr = pdl.attribute = 1 : i32
        %one_op = pdl.operation "arith.constant" {"value" = %one_attr} -> (%t : !pdl.type)
        %one = pdl.result 0 of %one_op
        pdl.replace %div with (%one : !pdl.value)
    }
}

// Pattern: x / x → 1 (floating-point)
pdl.pattern @divf_self : benefit(3) {
    %t = pdl.type
    %x = pdl.operand
    %div = pdl.operation "arith.divf" (%x, %x : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    
    pdl.rewrite %div {
        %one_attr = pdl.attribute = 1.0 : f64
        %one_op = pdl.operation "arith.constant" {"value" = %one_attr} -> (%t : !pdl.type)
        %one = pdl.result 0 of %one_op
        pdl.replace %div with (%one : !pdl.value)
    }
}

// ============================================
// Zero Multiplication Patterns
// ============================================

// Pattern: x * 0 → 0 (integer)
pdl.pattern @muli_zero : benefit(3) {
    %t = pdl.type
    %x = pdl.operand
    %zero_attr = pdl.attribute = 0 : i32
    %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%t : !pdl.type)
    %zero = pdl.result 0 of %zero_op
    %mul = pdl.operation "arith.muli" (%x, %zero : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    
    pdl.rewrite %mul {
        pdl.replace %mul with (%zero : !pdl.value)
    }
}

// Pattern: 0 * x → 0 (integer)
pdl.pattern @zero_muli : benefit(3) {
    %t = pdl.type
    %x = pdl.operand
    %zero_attr = pdl.attribute = 0 : i32
    %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%t : !pdl.type)
    %zero = pdl.result 0 of %zero_op
    %mul = pdl.operation "arith.muli" (%zero, %x : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    
    pdl.rewrite %mul {
        pdl.replace %mul with (%zero : !pdl.value)
    }
}

// Pattern: x * 0 → 0 (floating-point)
pdl.pattern @mulf_zero : benefit(3) {
    %t = pdl.type
    %x = pdl.operand
    %zero_attr = pdl.attribute = 0.0 : f64
    %zero_op = pdl.operation "arith.constant" {"value" = %zero_attr} -> (%t : !pdl.type)
    %zero = pdl.result 0 of %zero_op
    %mul = pdl.operation "arith.mulf" (%x, %zero : !pdl.value, !pdl.value) -> (%t : !pdl.type)
    
    pdl.rewrite %mul {
        pdl.replace %mul with (%zero : !pdl.value)
    }
}