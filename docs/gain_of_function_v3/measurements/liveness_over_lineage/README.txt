Liveness of the H84 lineage's learned parts at seven checkpoints (2026-09-22, CPU, the code of commit
fc0863a62 = before the gain-of-function build, 24 lived moments per checkpoint from its own episode).
Checkpoints at steps 7,196 to 220,675 load only with a FRESH touch encoder (its input was widened after
they were saved), so for them every part downstream of the touch token is approximate; the motor pool
does not read touch and its numbers are exact. Step 266,375 failed to build its probe world (a scene
file race in the scratch folder) and has no row.

step     pose-head  wellness-copy  reflex-blk0  recon-touch  memory-out  motor-pool  touch-pool  core-blk0-cross
7196     0.258      0.504          1.000        0.977        0.719       0.011       (fresh)     0.167
35165    0.055      0.234          0.406        0.789        0.510       0.005       (fresh)     0.016
67077    0.055      0.129          0.250        0.643        0.323       0.002       (fresh)     0.021
131952   0.027      0.102          0.109        0.438        0.354       0.001       (fresh)     0.005
220675   0.992*     0.059          0.914*       0.896*       0.740*      0.001       (fresh)     0.172*
328375   0.000      0.090          0.125        0.151        0.177       0.000       0.007       0.003
344400   0.000      0.051          0.047        0.049        0.365       0.001       0.091       0.004
(live share = units whose input exceeded -3 on some moment; attention = entropy as a share of the
even-spread maximum, 0 = locked on one input. * = out of line with its neighbours; with a fresh touch
encoder at that step the inner state differs, so these are not trusted.)
