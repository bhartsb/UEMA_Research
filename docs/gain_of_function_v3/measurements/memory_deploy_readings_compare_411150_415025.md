Memory deploy readings beyond vision (memory_deploy_readings_v2). Body 2 is numbered from 0 (the dashboard's body 3); the verdict is read without it. A change counts only beyond the largest change measured between two no-change checkpoints.
Moved beyond their lines, without body 2: 0 improved, 0 regressed. The operator's rule: several improving together marks a central pathology.

attention spread: the entropy of each attention row as a share of the even spread (0 = every row reads one input; 1 = every row reads all its inputs evenly)

| part | before (without body 2 / all 8) | after (without body 2 / all 8) | change without body 2 | no-change line | verdict |
|---|---|---|---|---|---|
| body head's pool (which of the forward model's tokens it reads) - the input it reads most: vision (27% of its weight) before, vision (24% of its weight) after | 0.970 / 0.971 | 0.961 / 0.961 | -0.0086 | 0.0641 | within noise |
| gaze head's pool (which of the forward model's tokens it reads) - the input it reads most: proprioception (35% of its weight) before, proprioception (33% of its weight) after | 0.838 / 0.837 | 0.823 / 0.822 | -0.0151 | 0.0488 | within noise |
| forward model, block 1, stage 1 - senses read each other - the input it reads most: vestibular (25% of its weight) before, vestibular (24% of its weight) after | 0.002 / 0.002 | 0.001 / 0.001 | -0.0006 | 0.0010 | within noise |
| forward model, block 1, stage 2 - each sense reads its past | 0.115 / 0.111 | 0.113 / 0.110 | -0.0016 | 0.0483 | within noise |
| forward model, block 1, stage 3 - everything reads everything | 0.106 / 0.103 | 0.096 / 0.093 | -0.0097 | 0.0335 | within noise |
| forward model, block 2, stage 1 - senses read each other - the input it reads most: vision (28% of its weight) before, vision (28% of its weight) after | 0.052 / 0.054 | 0.059 / 0.061 | +0.0075 | 0.0128 | within noise |
| forward model, block 2, stage 2 - each sense reads its past | 0.041 / 0.040 | 0.049 / 0.049 | +0.0086 | 0.0104 | within noise |
| forward model, block 2, stage 3 - everything reads everything | 0.074 / 0.075 | 0.093 / 0.093 | +0.0188 | 0.0402 | within noise |
| forward model, block 3, stage 1 - senses read each other - the input it reads most: proprioception (45% of its weight) before, proprioception (43% of its weight) after | 0.026 / 0.028 | 0.025 / 0.027 | -0.0015 | 0.0201 | within noise |
| forward model, block 3, stage 2 - each sense reads its past | 0.607 / 0.612 | 0.605 / 0.609 | -0.0017 | 0.0414 | within noise |
| forward model, block 3, stage 3 - everything reads everything | 0.245 / 0.241 | 0.247 / 0.246 | +0.0017 | 0.0674 | within noise |
| forward model, block 4, stage 1 - senses read each other - the input it reads most: proprioception (36% of its weight) before, proprioception (38% of its weight) after | 0.029 / 0.030 | 0.020 / 0.022 | -0.0092 | 0.0188 | within noise |
| forward model, block 4, stage 2 - each sense reads its past | 0.150 / 0.154 | 0.143 / 0.148 | -0.0073 | 0.0563 | within noise |
| forward model, block 4, stage 3 - everything reads everything | 0.118 / 0.118 | 0.127 / 0.129 | +0.0089 | 0.0383 | within noise |
| audio prediction head's attention (which of the forward model's newest tokens it reads) - the input it reads most: vestibular (69% of its weight) before, vestibular (64% of its weight) after | 0.291 / 0.276 | 0.303 / 0.293 | +0.0116 | 0.0903 | within noise |
| proprioception prediction head's attention (which of the forward model's newest tokens it reads) - the input it reads most: vestibular (53% of its weight) before, vestibular (50% of its weight) after | 0.383 / 0.374 | 0.384 / 0.375 | +0.0015 | 0.0991 | within noise |
| tactile prediction head's attention (which of the forward model's newest tokens it reads) - the input it reads most: tactile (50% of its weight) before, tactile (50% of its weight) after | 0.000 / 0.000 | 0.000 / 0.000 | +0.0000 | 0.0000 | within noise |
| vestibular prediction head's attention (which of the forward model's newest tokens it reads) - the input it reads most: proprioception (50% of its weight) before, proprioception (50% of its weight) after | 0.000 / 0.000 | 0.000 / 0.000 | +0.0000 | 0.0000 | within noise |
| vision prediction head's attention (which of the forward model's newest tokens it reads) - the input it reads most: tactile (50% of its weight) before, tactile (50% of its weight) after | 0.000 / 0.000 | 0.000 / 0.000 | -0.0000 | 0.0000 | within noise |

unit liveness: the share of the part's units whose output varies over the body's moments beyond its bf16 rounding floor (0 = no unit responds to the moment; 1 = every unit does)

| part | before (without body 2 / all 8) | after (without body 2 / all 8) | change without body 2 | no-change line | verdict |
|---|---|---|---|---|---|
| body head | 0.551 / 0.557 | 0.513 / 0.504 | -0.0374 | 0.0929 | within noise |
| forward model, block 1: feed-forward network | 0.974 / 0.972 | 0.963 / 0.961 | -0.0112 | 0.0314 | within noise |
| forward model, block 2: feed-forward network | 0.946 / 0.942 | 0.941 / 0.943 | -0.0047 | 0.0554 | within noise |
| forward model, block 3: feed-forward network | 0.996 / 0.995 | 0.986 / 0.986 | -0.0104 | 0.0123 | within noise |
| forward model, block 4: feed-forward network | 0.999 / 0.999 | 0.983 / 0.982 | -0.0156 | 0.0156 | within noise |
| the motor controller | 0.877 / 0.883 | 0.851 / 0.859 | -0.0263 | 0.0924 | within noise |
| the memory's needs forecast | 0.833 / 0.844 | 0.851 / 0.865 | +0.0179 | 0.0640 | within noise |
| audio prediction head | 0.998 / 0.996 | 1.000 / 1.000 | +0.0022 | 0.0022 | within noise |
| proprioception prediction head | 0.857 / 0.870 | 0.995 / 0.994 | +0.1376 | 0.3854 | within noise |
| tactile prediction head | 0.037 / 0.036 | 0.004 / 0.006 | -0.0335 | 0.1496 | within noise |
| vestibular prediction head | 0.161 / 0.154 | 0.371 / 0.384 | +0.2091 | 0.2552 | within noise |
| vision prediction head | 0.111 / 0.127 | 0.013 / 0.031 | -0.0975 | 0.4754 | within noise |
| the tenure pose head | 0.268 / 0.268 | 0.265 / 0.263 | -0.0028 | 0.1685 | within noise |

the memory's worth: ln(next-moment error with the memory's varying part taken out / error with it) (0 = the memory's variation helps no prediction; above 0 = it lowers the error)

| part | before (without body 2 / all 8) | after (without body 2 / all 8) | change without body 2 | no-change line | verdict |
|---|---|---|---|---|---|
| audio | -0.0000 / -0.0000 | -0.0000 / -0.0000 | +0.0000 | 0.0000 | within noise |
| proprioception | +0.0000 / +0.0000 | -0.0000 / -0.0000 | -0.0000 | 0.0000 | within noise |
| tactile | +0.0000 / +0.0000 | +0.0000 / +0.0000 | -0.0000 | 0.0000 | within noise |
| vestibular | +0.0000 / +0.0000 | +0.0000 / +0.0000 | -0.0000 | 0.0000 | within noise |
| vision | +0.0000 / +0.0000 | -0.0000 / -0.0000 | -0.0000 | 0.0000 | within noise |

the memory bias's size and what it does to the heads, on the checkpoint's own last 128 recorded moments (the scale survey's construction) (memory bias: root mean square per number of each sense slot's bias (tokens are about 5 per number; lower = better); top slot: the share of a head's attention on the slot it reads most (1 / 6 = even, 1 = one slot; lower = better); between bodies: how far a head's output differs between bodies, as a share of how far that sense's own tokens differ (0 = the same output for every body, a constant head; higher = better))

| part | before (without body 2 / all 8) | after (without body 2 / all 8) | change without body 2 | no-change line | verdict |
|---|---|---|---|---|---|
| audio prediction head: how far its output differs between bodies, as a share of how far the bodies' own audio tokens differ | 0.187 / 0.195 | 0.107 / 0.176 | -0.0802 | 0.0851 | within noise |
| proprioception prediction head: how far its output differs between bodies, as a share of how far the bodies' own proprioception tokens differ | 0.004 / 0.004 | 0.014 / 0.012 | +0.0097 | 0.0130 | within noise |
| tactile prediction head: how far its output differs between bodies, as a share of how far the bodies' own tactile tokens differ | 0.000 / 0.000 | 0.000 / 0.000 | -0.0002 | 0.0003 | within noise |
| vestibular prediction head: how far its output differs between bodies, as a share of how far the bodies' own vestibular tokens differ | unmeasured / unmeasured | unmeasured / unmeasured | unmeasured | unmeasured | unmeasured: a reading is missing |
| vision prediction head: how far its output differs between bodies, as a share of how far the bodies' own vision tokens differ | 0.005 / 0.005 | 0.003 / 0.004 | -0.0020 | 0.0079 | within noise |
| the memory bias on the audio slot | 6.360 / 6.360 | 5.585 / 5.585 | -0.7746 | 1.1712 | within noise |
| the memory bias on the motor slot | 103.895 / 103.895 | 127.788 / 127.788 | +23.8930 | 82.5053 | within noise |
| the memory bias on the proprioception slot | 205.677 / 205.677 | 207.284 / 207.284 | +1.6075 | 3.6880 | within noise |
| the memory bias on the tactile slot | 258.454 / 258.454 | 267.229 / 267.229 | +8.7757 | 25.2236 | within noise |
| the memory bias on the vestibular slot | 6.909 / 6.909 | 6.622 / 6.622 | -0.2874 | 1.1346 | within noise |
| the memory bias on the vision slot | 98.629 / 98.629 | 105.437 / 105.437 | +6.8080 | 26.6820 | within noise |
| audio prediction head: its attention's share on the slot it reads most - the input it reads most: vestibular (71% of its weight) before, vestibular (64% of its weight) after | 0.705 / 0.728 | 0.641 / 0.663 | -0.0646 | 0.2188 | within noise |
| proprioception prediction head: its attention's share on the slot it reads most - the input it reads most: vestibular (54% of its weight) before, vestibular (50% of its weight) after | 0.542 / 0.565 | 0.528 / 0.553 | -0.0144 | 0.2843 | within noise |
| tactile prediction head: its attention's share on the slot it reads most - the input it reads most: tactile (50% of its weight) before, tactile (50% of its weight) after | 0.500 / 0.500 | 0.500 / 0.500 | +0.0000 | 0.0000 | within noise |
| vestibular prediction head: its attention's share on the slot it reads most - the input it reads most: proprioception (50% of its weight) before, proprioception (50% of its weight) after | 0.500 / 0.500 | 0.500 / 0.500 | +0.0000 | 0.0000 | within noise |
| vision prediction head: its attention's share on the slot it reads most - the input it reads most: tactile (50% of its weight) before, tactile (50% of its weight) after | 0.500 / 0.500 | 0.500 / 0.500 | +0.0000 | 0.0000 | within noise |

written: D:\UEMA_Research\external\D\UEMA_scratch\memory_readings\compare_411150_415025\regression_pack.json
