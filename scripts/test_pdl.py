#!/usr/bin/env python3
"""
Interactive PDL Testing Script for xDSL
Test and learn PDL pattern matching with various examples
"""

import os
import subprocess
import sys
from pathlib import Path

class PDLTester:
    def __init__(self):
        self.script_dir = Path(__file__).parent
        self.repo_dir = self.script_dir.parent
        self.patterns_dir = self.repo_dir / "patterns"
        self.examples_dir = self.repo_dir / "examples"
        
        self.patterns = {
            "basic": self.patterns_dir / "patterns.pdl",
            "advanced": self.patterns_dir / "advanced_patterns.pdl",
            "custom": self.patterns_dir / "custom_pattern.pdl"
        }
        self.test_files = {
            "original": self.examples_dir / "input.mlir",
            "arith": self.examples_dir / "test_arith_ops.mlir",
            "examples": self.examples_dir / "test_examples.mlir"
        }
        
    def run_xdsl_opt(self, input_file, pattern_file, verbose=False):
        """Run xdsl-opt with PDL patterns"""
        combined_file = self.repo_dir / "combined_temp.mlir"
        output_file = self.repo_dir / "output_temp.mlir"
        
        # Combine pattern and input files
        with open(combined_file, 'w') as f:
            with open(pattern_file if isinstance(pattern_file, Path) else self.patterns_dir / pattern_file, 'r') as pf:
                f.write(pf.read())
            f.write('\n')
            with open(input_file if isinstance(input_file, Path) else self.examples_dir / input_file, 'r') as inf:
                f.write(inf.read())
        
        # Run xdsl-opt
        cmd = ["xdsl-opt", str(combined_file), "-p", "apply-pdl", "-o", str(output_file)]
        if verbose:
            cmd.append("--print-between-passes")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Read output
        output = ""
        if output_file.exists():
            with open(output_file, 'r') as f:
                output = f.read()
        
        # Cleanup
        combined_file.unlink(missing_ok=True)
        output_file.unlink(missing_ok=True)
        
        return output, result.stderr
    
    def extract_function_body(self, mlir_text, func_name):
        """Extract just the function body from MLIR output"""
        lines = mlir_text.split('\n')
        in_func = False
        func_lines = []
        
        for line in lines:
            if f"func.func @{func_name}" in line:
                in_func = True
            if in_func:
                func_lines.append(line)
                if line.strip() == '}' and in_func:
                    break
        
        return '\n'.join(func_lines)
    
    def test_pattern(self, test_name, input_mlir, pattern_file):
        """Test a specific pattern and show results"""
        print(f"\n{'='*60}")
        print(f"Testing: {test_name}")
        print(f"{'='*60}")
        
        # Create temporary input file
        temp_input = self.repo_dir / "temp_test.mlir"
        with open(temp_input, 'w') as f:
            f.write(input_mlir)
        
        # Run optimization
        output, stderr = self.run_xdsl_opt("temp_test.mlir", pattern_file)
        
        # Show results
        print("\nInput:")
        print(input_mlir)
        
        if output:
            # Extract just the function from output
            if "func.func @test" in output:
                func_output = self.extract_function_body(output, "test")
                print("\nOutput (optimized):")
                print(func_output)
            else:
                print("\nOutput:")
                print(output)
        
        if stderr:
            print("\nErrors/Warnings:")
            print(stderr)
        
        # Cleanup
        temp_input.unlink(missing_ok=True)
        
        return output
    
    def run_interactive(self):
        """Run interactive testing session"""
        print("PDL Pattern Testing Tool for xDSL")
        print("="*60)
        
        while True:
            print("\nOptions:")
            print("1. Test basic arithmetic patterns")
            print("2. Test advanced patterns")
            print("3. Test specific pattern")
            print("4. Write custom pattern and test")
            print("5. Show all available patterns")
            print("6. Run full test suite")
            print("7. Exit")
            
            choice = input("\nSelect option (1-7): ").strip()
            
            if choice == '1':
                self.test_basic_patterns()
            elif choice == '2':
                self.test_advanced_patterns()
            elif choice == '3':
                self.test_specific_pattern()
            elif choice == '4':
                self.write_and_test_pattern()
            elif choice == '5':
                self.show_patterns()
            elif choice == '6':
                self.run_test_suite()
            elif choice == '7':
                print("Exiting...")
                break
            else:
                print("Invalid option!")
    
    def test_basic_patterns(self):
        """Test basic arithmetic patterns"""
        test_cases = [
            ("x + 0 -> x", """func.func @test(%x : f64) -> f64 {
  %c0 = arith.constant 0.0 : f64
  %result = arith.addf %x, %c0 : f64
  func.return %result : f64
}"""),
            ("x - x -> 0", """func.func @test(%x : f64) -> f64 {
  %result = arith.subf %x, %x : f64
  func.return %result : f64
}"""),
            ("x / x -> 1", """func.func @test(%x : f64) -> f64 {
  %result = arith.divf %x, %x : f64
  func.return %result : f64
}"""),
        ]
        
        for name, mlir in test_cases:
            self.test_pattern(name, mlir, "patterns.pdl")
    
    def test_advanced_patterns(self):
        """Test advanced patterns"""
        test_cases = [
            ("x - 0 -> x", """func.func @test(%x : f64) -> f64 {
  %c0 = arith.constant 0.0 : f64
  %result = arith.subf %x, %c0 : f64
  func.return %result : f64
}"""),
            ("x * 2 -> x + x", """func.func @test(%x : f64) -> f64 {
  %c2 = arith.constant 2.0 : f64
  %result = arith.mulf %x, %c2 : f64
  func.return %result : f64
}"""),
            ("-(-x) -> x", """func.func @test(%x : f64) -> f64 {
  %neg1 = arith.negf %x : f64
  %neg2 = arith.negf %neg1 : f64
  func.return %neg2 : f64
}"""),
        ]
        
        for name, mlir in test_cases:
            self.test_pattern(name, mlir, "advanced_patterns.pdl")
    
    def test_specific_pattern(self):
        """Test a specific pattern interactively"""
        print("\nEnter MLIR code (end with empty line):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        
        mlir_code = '\n'.join(lines)
        
        print("\nSelect pattern file:")
        for key, file in self.patterns.items():
            print(f"  {key}: {file}")
        
        pattern_choice = input("Choice: ").strip()
        pattern_file = self.patterns.get(pattern_choice, "patterns.pdl")
        
        self.test_pattern("Custom Test", mlir_code, pattern_file)
    
    def write_and_test_pattern(self):
        """Write a custom PDL pattern and test it"""
        print("\nEnter PDL pattern (end with empty line):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        
        pattern = '\n'.join(lines)
        
        # Save to temporary pattern file
        temp_pattern = self.repo_dir / "temp_pattern.pdl"
        with open(temp_pattern, 'w') as f:
            f.write(pattern)
        
        print("\nEnter test MLIR code (end with empty line):")
        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)
        
        mlir_code = '\n'.join(lines)
        
        self.test_pattern("Custom Pattern Test", mlir_code, "temp_pattern.pdl")
        
        # Cleanup
        temp_pattern.unlink(missing_ok=True)
    
    def show_patterns(self):
        """Display all available patterns"""
        for name, file in self.patterns.items():
            filepath = file if isinstance(file, Path) else self.patterns_dir / file
            if filepath.exists():
                print(f"\n{'='*60}")
                print(f"Patterns in {file}:")
                print(f"{'='*60}")
                with open(filepath, 'r') as f:
                    for line in f:
                        if "pdl.pattern @" in line:
                            pattern_name = line.split('@')[1].split(':')[0].strip()
                            benefit = line.split('benefit(')[1].split(')')[0] if 'benefit(' in line else '?'
                            print(f"  @{pattern_name} (benefit: {benefit})")
    
    def run_test_suite(self):
        """Run comprehensive test suite"""
        print("\nRunning full test suite...")
        
        # Test all combinations
        for test_name, test_file in self.test_files.items():
            for pattern_name, pattern_file in self.patterns.items():
                if not (test_file if isinstance(test_file, Path) else self.examples_dir / test_file).exists():
                    continue
                if not (pattern_file if isinstance(pattern_file, Path) else self.patterns_dir / pattern_file).exists():
                    continue
                    
                print(f"\n--- Testing {test_file} with {pattern_file} ---")
                output, stderr = self.run_xdsl_opt(test_file, pattern_file)
                
                if output:
                    # Count number of operations before and after
                    with open(test_file if isinstance(test_file, Path) else self.examples_dir / test_file, 'r') as f:
                        original = f.read()
                    
                    orig_ops = original.count('arith.')
                    opt_ops = output.count('arith.')
                    
                    print(f"Operations: {orig_ops} -> {opt_ops} (reduced by {orig_ops - opt_ops})")
                
                if stderr:
                    print(f"Warnings: {stderr[:100]}...")

if __name__ == "__main__":
    tester = PDLTester()
    
    if len(sys.argv) > 1:
        # Command line mode
        if len(sys.argv) == 3:
            input_file = sys.argv[1]
            pattern_file = sys.argv[2]
            output, stderr = tester.run_xdsl_opt(input_file, pattern_file, verbose=True)
            print(output)
            if stderr:
                print("Errors:", stderr, file=sys.stderr)
        else:
            print("Usage: python test_pdl.py [input.mlir pattern.pdl]")
            print("   or: python test_pdl.py (for interactive mode)")
    else:
        # Interactive mode
        tester.run_interactive()