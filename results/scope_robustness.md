Scope robustness at 99% sparsity, 10 seeds. Paired difference = criterion minus magnitude, same scope.

| scope | criterion | test acc (mean +/- sd) | diff vs magnitude (pp) | 95% CI (pp) | p | W/T/L |
|---|---|---|---|---|---|---|
| layer | magnitude | 0.8464 +/- 0.0837 | - | - | - | - |
| layer | taylor | 0.9533 +/- 0.0106 | +10.69 | [+6.33, +16.40] | 0.0020 | 10/0/0 |
| layer | fisher | 0.9502 +/- 0.0154 | +10.38 | [+6.00, +16.04] | 0.0020 | 10/0/0 |
| global | magnitude | 0.9596 +/- 0.0105 | - | - | - | - |
| global | taylor | 0.9653 +/- 0.0073 | +0.58 | [-0.11, +1.36] | 0.2070 | 5/1/4 |
| global | fisher | 0.9662 +/- 0.0110 | +0.67 | [+0.02, +1.31] | 0.1055 | 7/1/2 |
