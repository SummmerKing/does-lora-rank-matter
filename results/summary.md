# Results

Model: `Qwen/Qwen2.5-1.5B-Instruct`. Greedy decoding, pass@1. 95% CIs bootstrap over problems; p-values are paired sign-flip tests, Holm-corrected within each benchmark.

## humaneval (164 problems)

| config | seeds | pass@1 | 95% CI | seed SD |
|---|---|---|---|---|
| base | 1 | 52.4 | [45.1, 59.8] | 0.0 |
| lora_r4 | 3 | 45.3 | [38.0, 52.6] | 1.9 |
| lora_r16 | 3 | 45.7 | [38.6, 53.0] | 1.1 |
| lora_r64 | 3 | 44.5 | [37.4, 51.6] | 2.7 |
| loraplus_r16 | 3 | 43.5 | [36.4, 50.6] | 0.7 |

| comparison | Δ pass@1 (pts) | 95% CI | p | p (Holm) |
|---|---|---|---|---|
| lora_r4 vs base | -7.1 | [-14.6, -0.0] | 0.067 | 0.337 |
| lora_r16 vs base | -6.7 | [-14.2, +0.8] | 0.093 | 0.370 |
| lora_r64 vs base | -7.9 | [-15.4, -0.4] | 0.047 | 0.284 |
| loraplus_r16 vs base | -8.9 | [-16.3, -1.6] | 0.021 | 0.146 |
| lora_r4 vs lora_r16 | -0.4 | [-2.6, +1.8] | 0.858 | 0.917 |
| lora_r64 vs lora_r16 | -1.2 | [-3.9, +1.4] | 0.459 | 0.917 |
| loraplus_r16 vs lora_r16 | -2.2 | [-4.7, +0.2] | 0.097 | 0.370 |

## mbpp (257 problems)

| config | seeds | pass@1 | 95% CI | seed SD |
|---|---|---|---|---|
| base | 1 | 49.0 | [43.2, 55.3] | 0.0 |
| lora_r4 | 3 | 55.6 | [49.9, 61.3] | 0.4 |
| lora_r16 | 3 | 55.3 | [49.4, 61.1] | 1.0 |
| lora_r64 | 3 | 54.7 | [49.0, 60.6] | 1.0 |
| loraplus_r16 | 3 | 54.9 | [49.2, 60.6] | 0.0 |

| comparison | Δ pass@1 (pts) | 95% CI | p | p (Holm) |
|---|---|---|---|---|
| lora_r4 vs base | +6.6 | [+1.2, +12.2] | 0.020 | 0.139 |
| lora_r16 vs base | +6.2 | [+0.8, +11.7] | 0.030 | 0.178 |
| lora_r64 vs base | +5.7 | [+0.1, +11.2] | 0.051 | 0.204 |
| loraplus_r16 vs base | +5.8 | [+0.4, +11.3] | 0.038 | 0.191 |
| lora_r4 vs lora_r16 | +0.4 | [-1.3, +2.1] | 0.769 | 1.000 |
| lora_r64 vs lora_r16 | -0.5 | [-2.1, +1.0] | 0.629 | 1.000 |
| loraplus_r16 vs lora_r16 | -0.4 | [-2.3, +1.6] | 0.789 | 1.000 |

## Validation loss vs pass@1

- humaneval: Spearman ρ = -0.53 over 12 runs (negative means lower loss goes with higher pass@1).
- mbpp: Spearman ρ = +0.31 over 12 runs (negative means lower loss goes with higher pass@1).
