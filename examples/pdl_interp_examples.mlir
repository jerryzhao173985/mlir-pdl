// PDL Interpreter Examples from xDSL
// These examples show how PDL patterns are compiled to PDL Interpreter IR

// ============================================
// Example 1: x + 0 → x
// ============================================

// Input program to optimize
func.func @add_zero_example() -> i32 {
  %x = arith.constant 4 : i32
  %zero = arith.constant 0 : i32
  %result = arith.addi %x, %zero : i32
  func.return %result : i32
}

// PDL Interpreter matcher function
pdl_interp.func @add_zero_matcher(%root : !pdl.operation) {
  // Get second operand
  %operand1 = pdl_interp.get_operand 1 of %root
  
  // Get defining operation of second operand
  %const_op = pdl_interp.get_defining_op of %operand1 : !pdl.value
  pdl_interp.is_not_null %const_op : !pdl.operation -> ^check_const, ^fail
  
^check_const:
  // Check if root is addition
  pdl_interp.check_operation_name of %root is "arith.addi" -> ^check_counts, ^fail
  
^check_counts:
  pdl_interp.check_operand_count of %root is 2 -> ^check_result_count, ^fail
  
^check_result_count:
  pdl_interp.check_result_count of %root is 1 -> ^get_first_operand, ^fail
  
^get_first_operand:
  %operand0 = pdl_interp.get_operand 0 of %root
  pdl_interp.is_not_null %operand0 : !pdl.value -> ^check_const_op, ^fail
  
^check_const_op:
  // Verify second operand comes from constant
  pdl_interp.check_operation_name of %const_op is "arith.constant" -> ^check_const_counts, ^fail
  
^check_const_counts:
  pdl_interp.check_operand_count of %const_op is 0 -> ^check_const_results, ^fail
  
^check_const_results:
  pdl_interp.check_result_count of %const_op is 1 -> ^get_value, ^fail
  
^get_value:
  // Check if constant is zero
  %value_attr = pdl_interp.get_attribute "value" of %const_op
  pdl_interp.is_not_null %value_attr : !pdl.attribute -> ^check_zero, ^fail
  
^check_zero:
  pdl_interp.check_attribute %value_attr is 0 : i32 -> ^verify_types, ^fail
  
^verify_types:
  // Ensure type consistency
  %result = pdl_interp.get_result 0 of %root
  %const_result = pdl_interp.get_result 0 of %const_op
  %result_type = pdl_interp.get_value_type of %result : !pdl.type
  %const_type = pdl_interp.get_value_type of %const_result : !pdl.type
  pdl_interp.are_equal %result_type, %const_type : !pdl.type -> ^match, ^fail
  
^match:
  // Record successful match
  pdl_interp.record_match @rewriters::@add_zero_rewriter(%operand0, %root : !pdl.value, !pdl.operation) 
    : benefit(2), loc([%const_op, %root]), root("arith.addi") -> ^fail
    
^fail:
  pdl_interp.finalize
}

// Rewriter module
builtin.module @rewriters {
  pdl_interp.func @add_zero_rewriter(%x : !pdl.value, %op : !pdl.operation) {
    // Replace addition with first operand
    pdl_interp.replace %op with (%x : !pdl.value)
    pdl_interp.finalize
  }
}

// ============================================
// Example 2: (a + b) - c → a + (b - c)
// ============================================

// Input program to optimize
func.func @associativity_example() -> i32 {
  %a = arith.constant 3 : i32
  %b = arith.constant 5 : i32
  %c = arith.constant 7 : i32
  %sum = arith.addi %a, %b : i32
  %result = arith.subi %sum, %c : i32
  func.return %result : i32
}

// PDL Interpreter matcher for associativity
pdl_interp.func @associativity_matcher(%root: !pdl.operation) {
  // Get result of subtraction
  %sub_result = pdl_interp.get_result 0 of %root
  pdl_interp.is_not_null %sub_result : !pdl.value -> ^get_sub_lhs, ^fail
  
^get_sub_lhs:
  // Get first operand of subtraction (should be addition result)
  %sub_lhs = pdl_interp.get_operand 0 of %root
  %add_op = pdl_interp.get_defining_op of %sub_lhs : !pdl.value
  pdl_interp.is_not_null %add_op : !pdl.operation -> ^check_sub, ^fail
  
^check_sub:
  // Verify root is subtraction
  pdl_interp.check_operation_name of %root is "arith.subi" -> ^check_sub_counts, ^fail
  
^check_sub_counts:
  pdl_interp.check_operand_count of %root is 2 -> ^check_sub_results, ^fail
  
^check_sub_results:
  pdl_interp.check_result_count of %root is 1 -> ^check_add, ^fail
  
^check_add:
  // Verify first operand is addition
  pdl_interp.check_operation_name of %add_op is "arith.addi" -> ^check_add_counts, ^fail
  
^check_add_counts:
  pdl_interp.check_operand_count of %add_op is 2 -> ^check_add_results, ^fail
  
^check_add_results:
  pdl_interp.check_result_count of %add_op is 1 -> ^extract_operands, ^fail
  
^extract_operands:
  // Extract all operands
  %a = pdl_interp.get_operand 0 of %add_op
  pdl_interp.is_not_null %a : !pdl.value -> ^get_b, ^fail
  
^get_b:
  %b = pdl_interp.get_operand 1 of %add_op
  pdl_interp.is_not_null %b : !pdl.value -> ^get_c, ^fail
  
^get_c:
  %c = pdl_interp.get_operand 1 of %root
  pdl_interp.is_not_null %c : !pdl.value -> ^verify_connection, ^fail
  
^verify_connection:
  // Verify addition result feeds into subtraction
  %add_result = pdl_interp.get_result 0 of %add_op
  pdl_interp.are_equal %add_result, %sub_lhs : !pdl.value -> ^get_types, ^fail
  
^get_types:
  // Get type for new operations
  %type = pdl_interp.get_value_type of %add_result : !pdl.type
  %sub_type = pdl_interp.get_value_type of %sub_result : !pdl.type
  pdl_interp.are_equal %type, %sub_type : !pdl.type -> ^match, ^fail
  
^match:
  // Record match and apply transformation
  pdl_interp.record_match @rewriters::@associativity_rewriter(%b, %c, %type, %a, %root : 
    !pdl.value, !pdl.value, !pdl.type, !pdl.value, !pdl.operation) 
    : benefit(1), generatedOps(["arith.subi", "arith.addi"]), 
      loc([%add_op, %root]), root("arith.subi") -> ^fail
      
^fail:
  pdl_interp.finalize
}

module @rewriters {
  pdl_interp.func @associativity_rewriter(%b: !pdl.value, %c: !pdl.value, 
                                          %type: !pdl.type, %a: !pdl.value, 
                                          %root: !pdl.operation) {
    // Create b - c
    %sub_op = pdl_interp.create_operation "arith.subi"(%b, %c : !pdl.value, !pdl.value) 
      -> (%type : !pdl.type)
    %sub_result = pdl_interp.get_result 0 of %sub_op
    
    // Create a + (b - c)
    %add_op = pdl_interp.create_operation "arith.addi"(%a, %sub_result : !pdl.value, !pdl.value) 
      -> (%type : !pdl.type)
    
    // Replace original with new result
    %add_results = pdl_interp.get_results of %add_op : !pdl.range<value>
    pdl_interp.replace %root with (%add_results : !pdl.range<value>)
    pdl_interp.finalize
  }
}

// ============================================
// Example 3: x - x → 0
// ============================================

pdl_interp.func @sub_self_matcher(%root: !pdl.operation) {
  // Check operation is subtraction
  pdl_interp.check_operation_name of %root is "arith.subf" -> ^get_operands, ^fail
  
^get_operands:
  // Get both operands
  %lhs = pdl_interp.get_operand 0 of %root
  %rhs = pdl_interp.get_operand 1 of %root
  
  // Check if operands are the same
  pdl_interp.are_equal %lhs, %rhs : !pdl.value -> ^get_type, ^fail
  
^get_type:
  // Get result type for creating zero constant
  %result = pdl_interp.get_result 0 of %root
  %type = pdl_interp.get_value_type of %result : !pdl.type
  
  // Record match
  pdl_interp.record_match @rewriters::@sub_self_rewriter(%type, %root : !pdl.type, !pdl.operation)
    : benefit(3), loc([%root]), root("arith.subf") -> ^fail
    
^fail:
  pdl_interp.finalize
}

module @rewriters {
  pdl_interp.func @sub_self_rewriter(%type: !pdl.type, %root: !pdl.operation) {
    // Create zero constant with appropriate type
    %zero_attr = pdl_interp.create_attribute 0.0 : f64
    %zero_op = pdl_interp.create_operation "arith.constant" {"value" = %zero_attr} 
      -> (%type : !pdl.type)
    %zero_result = pdl_interp.get_result 0 of %zero_op
    
    // Replace subtraction with zero
    pdl_interp.replace %root with (%zero_result : !pdl.value)
    pdl_interp.finalize
  }
}

// ============================================
// Example 4: Constant Folding - const1 + const2 → const3
// ============================================

pdl_interp.func @const_fold_matcher(%root: !pdl.operation) {
  // Check for addition
  pdl_interp.check_operation_name of %root is "arith.addi" -> ^get_lhs, ^fail
  
^get_lhs:
  // Get left operand and its defining op
  %lhs = pdl_interp.get_operand 0 of %root
  %lhs_op = pdl_interp.get_defining_op of %lhs : !pdl.value
  pdl_interp.is_not_null %lhs_op : !pdl.operation -> ^check_lhs_const, ^fail
  
^check_lhs_const:
  // Check if left is constant
  pdl_interp.check_operation_name of %lhs_op is "arith.constant" -> ^get_rhs, ^fail
  
^get_rhs:
  // Get right operand and its defining op
  %rhs = pdl_interp.get_operand 1 of %root
  %rhs_op = pdl_interp.get_defining_op of %rhs : !pdl.value
  pdl_interp.is_not_null %rhs_op : !pdl.operation -> ^check_rhs_const, ^fail
  
^check_rhs_const:
  // Check if right is constant
  pdl_interp.check_operation_name of %rhs_op is "arith.constant" -> ^get_values, ^fail
  
^get_values:
  // Extract constant values
  %lhs_val = pdl_interp.get_attribute "value" of %lhs_op
  %rhs_val = pdl_interp.get_attribute "value" of %rhs_op
  
  // Get result type
  %result = pdl_interp.get_result 0 of %root
  %type = pdl_interp.get_value_type of %result : !pdl.type
  
  // Record match for constant folding
  pdl_interp.record_match @rewriters::@const_fold_rewriter(%lhs_val, %rhs_val, %type, %root : 
    !pdl.attribute, !pdl.attribute, !pdl.type, !pdl.operation)
    : benefit(4), loc([%lhs_op, %rhs_op, %root]), root("arith.addi") -> ^fail
    
^fail:
  pdl_interp.finalize
}

// Note: Actual constant folding computation would happen in C++ implementation
// This is a simplified representation
module @rewriters {
  pdl_interp.func @const_fold_rewriter(%lhs_val: !pdl.attribute, %rhs_val: !pdl.attribute,
                                       %type: !pdl.type, %root: !pdl.operation) {
    // In real implementation, compute lhs_val + rhs_val
    // For demonstration, we create a constant with computed value
    %result_attr = pdl_interp.create_attribute 7 : i32  // Placeholder for computed value
    %const_op = pdl_interp.create_operation "arith.constant" {"value" = %result_attr}
      -> (%type : !pdl.type)
    %const_result = pdl_interp.get_result 0 of %const_op
    
    pdl_interp.replace %root with (%const_result : !pdl.value)
    pdl_interp.finalize
  }
}