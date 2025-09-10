func.func @main(%a : f64, %b : f64, %c : f64) -> f64 {
  %0 = arith.divf %b, %b : f64
  %1 = arith.subf %a, %a : f64
  %2 = arith.addf %c, %1 : f64
  %3 = arith.divf %2, %0 : f64
  func.return %3 : f64
}