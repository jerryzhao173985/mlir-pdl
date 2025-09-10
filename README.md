# MLIR PDL Pattern Rewriting with xDSL

A comprehensive learning framework for PDL (Pattern Description Language) pattern-based rewrites in MLIR using xDSL-opt. This repository provides tutorials, examples, and interactive tools to master PDL pattern matching and transformations.

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/jerryzhao173985/mlir-pdl.git
cd mlir-pdl

# Test a simple pattern
./scripts/run_pdl.sh examples/input.mlir patterns/patterns.pdl

# Run interactive testing tool
python scripts/test_pdl.py
```

## 📁 Repository Structure

```
mlir-pdl/
├── patterns/           # PDL pattern definitions
│   ├── patterns.pdl          # Basic arithmetic patterns
│   ├── advanced_patterns.pdl # Advanced optimization patterns
│   └── custom_pattern.pdl    # Custom example patterns
├── examples/           # MLIR test inputs
│   ├── input.mlir            # Original test case
│   ├── test_arith_ops.mlir  # Arithmetic operation tests
│   └── test_examples.mlir   # Various test scenarios
├── scripts/            # Tools and utilities
│   ├── run_pdl.sh           # Shell script for pattern application
│   └── test_pdl.py          # Interactive Python testing tool
└── docs/              # Documentation
    ├── PDL_TUTORIAL.md      # Complete PDL guide
    ├── EXERCISES.md         # Practice exercises with solutions
    └── README.md            # Original tutorial notes
```

## 🎯 Features

- **20+ Pre-built Patterns**: Ready-to-use optimization patterns for arithmetic operations
- **Interactive Testing**: Python tool for experimenting with patterns
- **Comprehensive Tutorial**: Step-by-step guide to PDL concepts
- **Practice Exercises**: 8 exercises with solutions to master PDL
- **Test Suite**: Multiple test cases demonstrating pattern applications

## 🛠️ Requirements

- [xDSL](https://github.com/xdslproject/xdsl) installed and `xdsl-opt` in PATH
- Python 3.6+ (for interactive tools)
- Bash shell (for scripts)

### Installing xDSL

```bash
pip install xdsl
```

## 📖 Usage Examples

### Basic Pattern Application

Apply patterns to optimize MLIR code:

```bash
# Apply basic patterns
./scripts/run_pdl.sh examples/input.mlir patterns/patterns.pdl

# Apply advanced patterns
./scripts/run_pdl.sh examples/test_arith_ops.mlir patterns/advanced_patterns.pdl
```

### Direct xdsl-opt Usage

```bash
# Combine pattern and input files
cat patterns/patterns.pdl examples/input.mlir > combined.mlir

# Apply PDL patterns
xdsl-opt combined.mlir -p apply-pdl -o output.mlir

# View transformations step-by-step
xdsl-opt combined.mlir -p apply-pdl --print-between-passes
```

### Interactive Testing

```bash
python scripts/test_pdl.py
```

Options:
1. Test basic arithmetic patterns
2. Test advanced patterns
3. Test specific pattern
4. Write custom pattern and test
5. Show all available patterns
6. Run full test suite

## 📚 Available Patterns

### Basic Patterns (`patterns.pdl`)
- `x + 0 → x` - Addition identity
- `x - x → 0` - Self subtraction
- `x / x → 1` - Self division
- `x * 0 → 0` - Multiplication by zero
- `x * 1 → x` - Multiplication identity
- `x / 1 → x` - Division identity

### Advanced Patterns (`advanced_patterns.pdl`)
- `x - 0 → x` - Subtraction identity
- `0 - x → -x` - Zero minus value
- `x * 2 → x + x` - Multiplication to addition
- `x * -1 → -x` - Negation by multiplication
- `0 / x → 0` - Zero division
- `-(-x) → x` - Double negation
- `(x + y) - y → x` - Addition cancellation
- `x ^ x → 0` - XOR with self
- And many more...

## 🧪 Example Transformation

Input MLIR:
```mlir
func.func @main(%a : f64, %b : f64, %c : f64) -> f64 {
  %0 = arith.divf %b, %b : f64    // b/b
  %1 = arith.subf %a, %a : f64    // a-a
  %2 = arith.addf %c, %1 : f64    // c+(a-a)
  %3 = arith.divf %2, %0 : f64    // (c+(a-a))/(b/b)
  func.return %3 : f64
}
```

After applying patterns:
```mlir
func.func @main(%a : f64, %b : f64, %c : f64) -> f64 {
  func.return %c : f64  // Simplified to just return c
}
```

## 🎓 Learning Path

1. **Start with the Tutorial**: Read `docs/PDL_TUTORIAL.md` for comprehensive PDL concepts
2. **Run Examples**: Use `./scripts/run_pdl.sh` to see patterns in action
3. **Practice Exercises**: Complete exercises in `docs/EXERCISES.md`
4. **Write Custom Patterns**: Create your own patterns and test them
5. **Interactive Exploration**: Use `test_pdl.py` for experimentation

## 📝 Writing Your Own Patterns

Basic pattern structure:

```mlir
pdl.pattern @pattern_name : benefit(priority) {
  // Match section
  %t = pdl.type
  %x = pdl.operand
  %operation = pdl.operation "arith.op" (%x : !pdl.value) -> (%t : !pdl.type)
  
  // Rewrite section
  pdl.rewrite %operation {
    pdl.replace %operation with (%x : !pdl.value)
  }
}
```

Key concepts:
- **benefit**: Higher values = higher priority (typically 1-10)
- **pdl.type**: Matches any type
- **pdl.operand**: Matches operation inputs
- **pdl.operation**: Matches specific operations
- **pdl.replace**: Replaces matched operation

## 🔍 Debugging Tips

1. **Verify Pattern Syntax**: Ensure proper PDL operation formatting
2. **Check Type Matching**: Use correct type literals (e.g., `0.0 : f64` not `0 : f64`)
3. **Pattern Priority**: Higher benefit patterns apply first
4. **Use Verbose Mode**: Add `--print-between-passes` to see transformations

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Add new patterns
- Improve documentation
- Create more test cases
- Fix bugs or enhance tools

## 📄 License

This project is open source and available under the MIT License.

## 🔗 Resources

- [xDSL Project](https://github.com/xdslproject/xdsl)
- [MLIR Documentation](https://mlir.llvm.org/)
- [PDL Dialect Reference](https://mlir.llvm.org/docs/Dialects/PDLOps/)

## ✨ Acknowledgments

Built with xDSL - A Python-native SSA compiler toolkit

---

**Created by**: Jerry Zhao  
**Repository**: [github.com/jerryzhao173985/mlir-pdl](https://github.com/jerryzhao173985/mlir-pdl)