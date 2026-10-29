# MIMo

`mimo/` is MIMo, the simulated infant body this code runs on, included so a clone needs nothing else:
https://github.com/trieschlab/MIMo at commit `f2be26b` ("Renamed to Mimo3") of its `v3` branch, with
`MIMo_bridge_work.patch` applied. MIT License: `mimo/LICENSE` (the same text as `LICENSE_MIMo.txt`). Install it
from the repository root with `pip install -e third_party/MIMo/mimo --no-deps`.

`mimo/` is file for file the MIMo the saved state was trained with (checked on Oct 22, 2026: 0 differing files).
Apart from the patch it is MIMo's own text, which names the Gym library after its former owner in three places
(`mimo_infant/touch/touch.py`, MIMo's `README.md`).

`MIMo_bridge_work.patch` is `git diff f2be26b c7383b1` of the fork this code was developed against, kept so the
changes can be read on their own. It changes five files and carries the MIT License of MIMo:

- `mimo_infant/vision/vision.py`: SimpleVision renders each eye's own camera under
  gymnasium 1.0 and later (both eyes rendered the free camera).
- `mimo_infant/simulation/envs/mimo_env.py`: the sensory-delay and motor-delay histories are
  created before the first observation (a delay above 0 crashed at construction), a delay is
  exactly k steps, and `custom_measurements` is passed by keyword (it was landing in
  `mimo_version`).
- `mimo_infant/simulation/benchmark.py`: uses `env.unwrapped.dt`.
- `mimo_infant/simulation/assets/mimo/MIMo_metav2.xml`: PositionalModel locks joints with the
  XML equality constraints (control on torque motors was clipped and zeroed every substep);
  inter-finger coupling tendons are limit bands, not springs.

To rebuild `mimo/` from MIMo's own repository instead:

```
git clone https://github.com/trieschlab/MIMo
git -C MIMo checkout f2be26b
git -C MIMo apply <this folder>/MIMo_bridge_work.patch
```
