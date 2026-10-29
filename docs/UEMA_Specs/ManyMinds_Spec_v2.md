# Many Minds Architecture — Specification v0.2 (DRAFT)

**Status:** Working draft — under active discussion
**Amends:** UEMA_Spec_v3_4 Sections 13–18
**Supersedes:** ManyMinds_Spec_v1.md
**Date:** March 2026

---

## 1. Design Principles

The Many Minds architecture must follow the same philosophy as the rest of UEMA: capabilities emerge from prediction and wellness gradients, not from programmed protocols. Specifically:

- No hardcoded message types, alert enums, or communication protocols
- No explicit override/release commands
- No ownership flags on motor channels (should maybe various levels of imagined wellness change help drive this? For example a sudden drastic negative change imagined. not sure.)
- Inter-mind communication is learned, not prescribed
- Motor control transitions emerge from attention and confidence, not from arbitration rules

Where v3.4 specified a fixed protocol (alert → urgency threshold → override → release), this revision replaces it with emergent dynamics grounded in prediction error and learned internal state sharing.

---

## 1.1 Single-Mind Architecture Reference

Every mind in the hierarchy is a complete instance of the UEMA core loop. This section documents the full single-mind architecture so that all Many Minds reasoning accounts for every component.

### The Core Loop (per frame, per mind)

```
Sensors (P_t)                              ← Grounded Register: raw physical state
  ↓
Encoders (shared, run once)                ← 6 modality encoders: V, A, T, P, M, InterMind
  ↓                                           Each produces [B, 1, d_model] token
z_t = {z_vis, z_aud, z_tac, z_pro, z_mot, z_inter}
  ↓
Forward Model                              ← 8 three-stage transformer blocks
  │  Input: z_t + context_buffer (past frames)
  │  Stage 1: Cross-modal attention (within timestep, across modalities)
  │  Stage 2: Causal temporal attention (within modality, across time)
  │  Stage 3: Cross-modal temporal bridge (across both)
  │  Output: latents [B, seq_len, d_model]
  │          last_step [B, n_mod, d_model]    ← THIS IS THE MIND'S STATE (Section 2.9)
  │          context_update [B, n_mod, d_model]
  ↓
Prediction Heads (5)                       ← One per sensory modality + intermind (not motor)
  │  Each: learned query + MHA + MLP → predicted next-frame embedding [d_model]
  ↓
Wellness Evaluators
  │  Grounded (5 evaluators on raw P_t):
  │    prediction_error: 1 - clamp(PE, 0, 1)
  │    tactile_comfort:  1 - clamp(mean_pressure, 0, 1)
  │    capability:       mean(body_set_capacities)
  │    coherence:        body_schema coherence or cosine_sim fallback
  │    motor_state:      ROM boundary penalty or center preference
  │  Approximate (1 MLP on sensory tokens):
  │    Pools sensory tokens (excluding intermind) → [B, 5]
  │    Trained to match grounded outputs. Used in imagination.
  ↓
Wellness Gradient                          ← THE core motor drive signal
  │  torch.autograd.grad(total_wellness, motor_commands, create_graph=True)
  │  Produces [B, n_handles, 9] — direction in motor space that improves wellness
  ↓
Motor Controller                           ← Adaptive computation depth
  │  Input: pooled_latents + handle_state + wellness_grad
  │  Confidence head: pooled_latents → GELU → sigmoid → [0,1]
  │  If confident: depth=1 (fast path, ~1ms)
  │  If uncertain: depth=2..max (imagination rollouts)
  │  Output: [B, n_handles, 9] per handle: pos(3) + rot(4) + radius(1) + strength(1)
  ↓
Action Buffer (SHM)                        ← 1176 bytes Python→UE5
  ↓
Physics Handles (UE5)                      ← Execute motor commands on ragdoll
```

### Imagination Pipeline (within motor controller deep path)

```
Motor controller confidence < threshold
  ↓
forward_model.predict_from_action()        ← Single-step: what happens if I do this?
  │  Replaces motor token, runs prediction heads
  │  Detaches non-motor tokens (one-step only)
  ↓
score_predicted_wellness()                 ← Approximate evaluator scores imagined state
  ↓
forward_model.imagine_step()               ← Multi-step chain: full gradient flow
  │  Does NOT detach — gradient flows through entire chain
  │  Shifts context buffer forward
  │  Used for imagination-aware training (spec §6.3)
  ↓
Cross-simulation attention (future)        ← Multiple rollouts with noise → consensus
  │  Attend across rollouts to find invariants
  │  Divergent details wash out, intention crystallizes (what does wash out mean?)
```

### Intention (future, requires accurate forward model)

Where does "desired state" come from? Two mechanisms, operating at different levels:

**Level 1 — Pain/pleasure → wellness gradient direction (Patent US 11,537,850 B2; Spec §8)**
The patent specifies that pain attenuates sensor resolution (withdrawal) while pleasure amplifies it (approach). In UEMA, this translates directly: the wellness evaluator maps current sensory state to a wellness vector, and `torch.autograd.grad(wellness, motor_commands)` produces a gradient that points TOWARD states the system has learned are pleasant and AWAY from states it has learned are painful. The forward model learns these associations ("this visual pattern led to wellness increase → approach when detected"). A predicted-pleasant state IS a goal state — the forward model can generate it as a target for IntentionGenerator to invert.

**Level 2 — Curiosity → moderate-PE seeking → select_goal() (Spec §21)**
`CuriositySystem.compute_drive()` returns positive when PE is in the goldilocks zone (0.1–0.4), negative when too low (boredom) or too high (overwhelm). This drive modulates the wellness gradient direction. In Stream 4 (Spec §20.5), `curiosity.select_goal(fwd_model, z_t)` uses the forward model to identify reachable states where PE would be moderate — that IS the desired state generator.

**The chain:**
```
Patent layer:   pain/pleasure → sensor attenuation/amplification
Spec §8:        pain/pleasure → wellness gradient direction (approach/avoid)
Spec §21:       curiosity drive → moderate-PE seeking → select_goal()
Spec §10:       desired_state → IntentionGenerator → model inversion → motor plan
Spec §10.3:     plan_multistep + cross-simulation attention → robust execution
```

**The mechanism:**
```
Desired state (from wellness gradient + curiosity drive)
  ↓
IntentionGenerator (Spec §10.2)            ← Model inversion via gradient descent
  │  Optimizes motor commands to minimize:
  │    MSE(forward_model.predict(current, action), desired) + strength penalty
  │  50 optimization steps, lr=0.1
  ↓
plan_multistep() (Spec §10.3)             ← Chain intentions with imagination rollouts
  │  Plan a few steps → execute → re-observe in Grounded Register → replan
  │  Naturally short (3-5 steps) because imagination drifts
```

**Plain-language walkthrough of the three stages:**

*Stage 1 — "Where do I want to be?" (Desired State).* Two drives combine to produce a target state vector. The wellness gradient computes `grad(wellness, motor_commands)`, which points toward sensory states that have historically felt good and away from states that felt bad — a compass needle toward "pleasant." Separately, the curiosity system looks for reachable states where prediction error would be moderate (not boring, not overwhelming — the goldilocks zone of 0.1–0.4), using the forward model to ask "what nearby states would be interestingly surprising?" These two signals merge into a single concrete target: a latent-space vector representing "where I'd like to end up."

*Stage 2 — "What actions get me there?" (IntentionGenerator / model inversion).* The forward model predicts: "given current state + this action → next state." IntentionGenerator runs that model backwards via gradient descent: start with random motor commands, ask the forward model what state they'd produce, measure the gap to the desired state (MSE) plus an action-strength penalty, adjust the motor commands to shrink the gap, and repeat 50 times. The result is a set of motor commands the forward model believes will move toward the desired state. Same idea as training a neural net, except instead of updating weights you update the input actions.

*Stage 3 — "Plan a little, act, recheck" (plan_multistep).* Rather than planning a long sequence (which accumulates forward-model error), the system plans only 3–5 steps ahead, executes them in the real environment, re-observes actual sensory state via the Grounded Register, and replans from the new real state. Plans are intentionally short because each imagined step drifts further from reality. Frequent re-grounding in actual observation lets the system self-correct rather than commit to a long, increasingly wrong plan.

*Intuition:* navigating a foggy path. You know which direction feels good (wellness + curiosity). You mentally simulate a few steps to pick foot placement (IntentionGenerator). You take those steps, look around again, and replan — never charting the whole route at once because your mental simulation gets fuzzier the further out you go.

**UEMA Spec dependency gaps for this mechanism:**
- `curiosity.select_goal()` is called in Stream 4 pseudocode (§20.5) but the function body is never defined. The curiosity *drive* (scalar) is specified, but the projection from "I want moderate PE" to a specific target state vector in `[d_model]` latent space is not. **Note:** `select_goal()` is a Stream 4 feature (post-1M steps), not a Many Minds prerequisite. Many Minds needs mastery detection, splitting, and motor blending — none of which require goal selection.
- IntentionGenerator (§10) EXISTS and is FUNCTIONAL (`intention_generator.py:78-189`, 11 tests). Disabled by default (`enabled: false` in config), which is correct — it requires an accurate forward model. Includes `plan_multistep` and `CrossSimulationAttention`.
- CuriositySystem (§21) EXISTS and is FUNCTIONAL (`curiosity.py:12-48`). `compute_drive()` is implemented and called in the training step. Only `select_goal()` is unimplemented (Stream 4, see above).

### Body Schema (future, spec §11)

```
DifferentiableBodySchema                   ← EMA motor-sensor correlation tracker
  │  update(motor_delta, sensor_delta)     ← Every frame
  │  self_membership(): sigmoid(temp * (corr - 0.5))
  │  coherence_score(): mean(|membership - 0.5| * 2)
  │  Feeds coherence wellness evaluator
  │  ~6.6K elements (EMA buffers, ~26 KB, no trainable weights)
```

### Training Loss Components

```
total_loss = (
    1.0  × prediction_loss          # Per-modality MSE (forward model)
  + 0.1  × contrastive_loss         # VICReg or InfoNCE (encoder collapse prevention)
  + 0.5  × imagination_loss         # Multi-step chain against real trajectory
  + 0.01 × latent_reg_loss          # Predictions within 2σ of real distribution
  + 0.3  × approx_wellness_loss     # Approximate matches grounded evaluator
  + 0.2  × anticipation_loss        # NOT IMPLEMENTED — future observable motor preparation
  + 0.1  × gaze_loss                # Active gaze direction
  + var  × motor_loss               # Mode-dependent (BASELINE/DIRECT/FORWARD)
)
Single AdamW optimizer, lr=3e-4, weight_decay=0.01, grad_clip=1.0, bf16
```

### Key Dimensions (spec target)

| Component | Value |
|-----------|-------|
| d_model | 768 |
| n_heads | 12 (head_dim=64) |
| n_layers (forward model) | 8 three-stage blocks |
| n_modalities | 6 (V, A, T, P, M, InterMind) |
| n_handles | 23 (full body) |
| context_length | 8 current, target 32 (~0.5s at 60 FPS) |
| n_bodysets | 10 (fatigue/energy tracking) |
| vision_rate | 24 Hz |
| other_rates | 60 Hz |
| total_params | ~271M |

---

## 2. Established Ideas

### 2.1 Minds Are Prediction Engines Over Domains

Each mind has its own forward model, motor controller, and wellness evaluator operating over a subset of handles and sensors. A mind is always running inference — predicting, computing PE, generating motor commands.

"Frozen" means gradients are naturally near-zero because the domain is mastered, not that inference stops. A mastered mind still perceives, predicts, detects anomalies, and generates motor commands every frame. It just has nothing left to learn in its domain — until conditions change.

### 2.2 Shared Sensory Stream = Ground Truth

All minds receive identical sensory tokens (vision, audio, tactile, proprio, motor). Encoders run once, tokens are broadcast read-only. This is physical reality — same body, same world. Every mind sees the exit sign. Every mind feels the slip. The shared stream is how minds observe the consequences of each other's actions without needing explicit messages.

### 2.3 Inter-Mind Channel = Readable Internal State

z_intermind is NOT a message channel. It carries each mind's **readable internal state** — forward model hidden activations, motor intent, prediction error — directly to other minds, bypassing the physics delay of the shared sensory stream.

This is analogous to efference copies in neuroscience: when the motor cortex sends a command, a copy goes to sensory cortex so it can predict the consequence before it arrives through actual sensation. The inter-mind channel serves the same purpose across minds.

**Why this must exist (why shared sensory stream alone is insufficient):**

A human reaching for a cup pre-shapes the fingers during the reach, before contact, before tactile feedback. The hand region doesn't wait for sensory consequences of the arm's movement. It has the arm's motor intent directly through fast internal neural pathways.

This is what makes biological motor learning fast compared to flat RL. An RL robot learns one monolithic policy over all joints — no decomposition, no anticipatory coordination, millions of trials. Biological systems have hierarchical decomposition AND direct internal state sharing between regions. The hand reads the arm's intent and acts in parallel.

Without z_intermind, each mind would have to wait for the physical consequences of other minds' actions to appear in the sensory stream. Too slow for coordinated movement.

### 2.3.1 Inter-Mind Influence Is Immediate in Computation, Not Delayed Through Physics

The purpose of z_intermind is not merely to provide abstract context. It allows one mind's internal state to alter another mind's computation in the same frame before motor output is produced. This is how higher-level concern can rapidly reshape present-world behavior without waiting for the consequences of action to arrive through the shared sensory stream.

Concretely, z_intermind is encoded alongside the other modality tokens and enters the receiving mind's forward model on the current frame. Because the motor controller reads the resulting latent state, a change in another mind's readable internal state can immediately change:
- the receiving mind's predictions,
- its approximate wellness estimate,
- its confidence,
- its adaptive computation depth,
- and its motor output.

This is strongest at the top-mind ↔ attending-mind interface. The top mind does not send prescribed messages or override commands. It changes the attending mind's computation by shifting the attending mind's latent state through z_intermind.

### 2.4 Per-Mind Filtering of Inter-Mind State

z_intermind must be per-mind, not a single shared vector. Each mind uses a learned identity embedding to attend over available internal states from other minds and extract what's relevant. The hand mind attends strongly to the arm mind's state. The posture mind may not care about the arm at all.

```python
# TO BE BUILT — this class does not yet exist in the codebase.
# The current implementation is a SelfAttentionPoolEncoder stub
# (`lem/model/encoders/self_attention.py`) receiving synthetic Gaussian
# noise with loss_weight=0. This pseudocode defines the target replacement.
class PerMindIntermindEncoder(nn.Module):
    def __init__(self, d_model, max_minds):
        self.mind_embeddings = nn.Embedding(max_minds, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads=4)

    def forward(self, mind_id, all_mind_states):
        # mind_id: int index for this mind
        # all_mind_states: [n_minds, d_model] internal states from all other minds
        query = self.mind_embeddings(mind_id).unsqueeze(0)
        z_intermind, weights = self.cross_attn(query, all_mind_states, all_mind_states)
        return z_intermind.squeeze(0)  # [d_model]
```

The encoder weights are shared across all minds. Each mind's identity determines what it extracts. The attention weights are interpretable and learned, not prescribed.

### 2.5 Motor Control Is Confidence-Weighted Competition

No ownership flags. No override/release protocol. Every mind generates motor commands for its channels with an implicit confidence. Control transitions emerge from attention dynamics:

- **"Override" is not a command.** The higher mind's PE spikes in the lower mind's domain. The higher mind focuses on those channels — its motor outputs become stronger and more confident. The lower mind's outputs are naturally weaker (its own PE is high, its confidence drops). The higher mind's outputs dominate through competition, not through a flag.

- **"Release" is not a command.** The higher mind finishes dealing with the situation. Its attention drifts elsewhere. Its motor outputs for those channels fade. The lower mind's steady-state outputs dominate again by default because nothing is competing.

- **Ceding control is natural.** The lower mind doesn't receive a "you're overridden" signal. Its outputs are simply outcompeted when the higher mind is actively engaged. When the higher mind loses interest, the lower mind was always there, always generating — it just becomes the loudest voice again.

This mirrors human experience: you don't consciously "release" control of driving back to autopilot. You just start thinking about dinner and driving becomes automatic again.

### 2.5.1 Attending-Mind Engagement: Learned Escalation Without Override

The attending mind is the present-world motor coordinator. The top mind has no PH handles and does not emit motor commands. Therefore, top-mind influence on immediate embodied behavior is expressed by changing the attending mind's operating mode, not by entering per-handle arbitration directly.

The attending mind computes a learned engagement signal each frame:

```
g_engage = sigmoid(MLP([latents_attend, z_inter_from_top, pe_attend, approx_wellness_attend]))
```

This is a forward-pass quantity only. It is not a runtime wellness gradient, not an override flag, and not a programmed protocol. It determines how strongly the attending mind operates in executive mode for the current frame.

**Low g_engage:**
- routine present-world coordination,
- specialist child minds dominate their mastered domains,
- top-mind influence is mostly advisory.

**High g_engage:**
- attending-mind computation depth increases,
- broad present-world coordination gain increases,
- domination by routine lower-level outputs on contested handles is reduced,
- the attending mind acts as a stronger executive coordinator for imminent danger, route-critical events, or unstable novel tasks.

This gives the architecture a continuous transition between guidance and strong re-engagement without requiring top-mind PH commands or hardcoded override logic.

### 2.6 Splitting = Crystallizing Stable Weights

When a region of competence has near-zero PE and near-zero gradients over a sustained period, those weights are copied into a new mind instance. The parent's capacity is freed for new learning. The child's weights are already near-optimal — "frozen" describes their natural state, not an imposed constraint.

### 2.7 Splitting Follows Kinematic Chains

Any mind can split, not just the top. Arm mind can split off hand mind. Posture mind can split off balance mind. This produces anatomically meaningful hierarchies rather than flat skill clusters.

Example growth trajectory:
```
Phase 1: [Top] → all 23 handles
Phase 2: [Top] splits off [Posture] (legs/feet/spine)
Phase 3: [Top] splits off [Arm_L] (UpperArm_L, LowerArm_L, Wrist_L + fingers)
Phase 4: [Arm_L] splits off [Hand_L] (Thumb_L..Pinky_L)
Phase 5: [Posture] splits off [Balance] (feet/ankles)

Result:
[Top] — high-level planning, novel situations, gaze
  ├── [Posture] — core, legs
  │     └── [Balance] — ankle adjustments
  ├── [Arm_L] — reaching (UpperArm, LowerArm, Wrist)
  │     └── [Hand_L] — grasping (fingers)
  ├── [Arm_R] — reaching
  │     └── [Hand_R] — grasping
  └── [Gaze] — eye tracking
```

### 2.8 Thawing Should Be Natural

If freezing is natural (gradients → 0 when PE → 0), then thawing should also be natural (gradients increase when PE increases due to domain change). A mastered domain encountering novel conditions produces real PE and real gradients. The mind adapts because there is something to learn again, not because a programmed trigger re-enables learning.

### 2.9 Inter-Mind State Content

Each mind's readable internal state is a learned projection of its forward model's last hidden layer — the output of the final transformer block before the prediction heads. This is the richest representation the mind has: it encodes the mind's current understanding of its domain, its predictions about what's coming next, and implicitly its motor intent (since motor commands are generated from this same representation).

Why the last hidden layer and not other candidates:

- **PE (scalar):** Too compressed. Tells you *how wrong* a mind is, not *what it's thinking about* or *what it intends to do*.
- **Prediction head outputs (per-modality vectors):** Already projected down to individual modalities. Loses cross-modal context — you'd know what the arm mind predicts for proprioception but not how that relates to what it sees.
- **Motor controller output (handle commands):** Only captures *what* the mind will do, not *why* or *what it expects to happen*. The hand mind needs the arm's intent and context, not just its raw joint targets.
- **Last hidden layer:** Contains all of the above in compressed form. The forward model's final representation integrates cross-modal and temporal context through 8 attention blocks. It's what the prediction heads and motor controller both read from. It is the mind's state.

Each mind projects its last hidden layer through a learned linear layer to produce a fixed-size vector `[d_model]`. The set of all minds' projected states becomes the key/value input to the per-mind attention encoder (Section 2.4), where each mind extracts what's relevant to it.

### 2.10 Imagination Scope Is the Higher Mind's Authority

The adaptive motor controller uses imagination rollouts when confidence is low: predict consequences of candidate actions via the forward model, score them with the approximate wellness evaluator, pick the best. This is where the hierarchy matters most.

Each mind can imagine within its domain:
- **Hand mind** imagines: "if I close these fingers, what tactile feedback will I get?"
- **Arm mind** imagines: "if I reach here, what proprioception results?"
- **Top mind** imagines: "if I reach, grasp, and lift, what happens to overall wellness over 30 frames?"

The higher mind's advantage is NOT better motor execution — the lower mind is better at that (it mastered the domain, that's why it was split off). The higher mind's advantage is **broader imagination scope combined with a broader wellness evaluator**:

- The driving mind can imagine "take exit → road curves → maintain lane." Domain-local wellness says fine.
- The higher mind can imagine "take exit → wrong destination → 30 minutes lost." Broader wellness says bad.

This connects directly to the five wellness evaluators. When a mind splits, the child gets domain-scoped evaluators:
- **prediction_error**: from its own (domain-scoped) forward model
- **tactile_comfort**: only patches in its domain
- **capability**: only its body sets
- **motor_state**: only its handles' ROM positions
- **coherence**: still needs body schema (scope unclear — Gap)

The parent retains evaluators with broader scope. The **approximate wellness evaluator** (used during imagination) must also be per-mind — trained to match that mind's grounded evaluators. A subordinate mind's approximate evaluator becomes specialized: good at scoring imagined grasps, poor at scoring imagined navigation decisions.

The higher mind doesn't need to generate better motor commands to "override." It needs to generate a wellness gradient that points in a different direction than what the lower mind is doing. The wellness gradient IS the motor drive signal in UEMA. Broader wellness scope → broader gradient → authority over the lower mind's narrower gradient.

For the top mind specifically, this authority is expressed by conditioning the attending mind through z_intermind and attending-side engagement modulation, not by direct top-mind PH commands.

### 2.11 What Gets Copied at Split Time

When a mind splits, the child receives a complete instance of the core loop (see Section 1.1):

| Component | Copied? | Scope Change |
|-----------|---------|-------------|
| Forward model (8 blocks) | Cloned | Full deepcopy. All 6 modality inputs retained. Domain specialization emerges through training rather than architectural pruning. |
| Prediction heads (5) | Cloned | Needed for imagination |
| Motor controller | Cloned | Domain-scoped to child's handles |
| Confidence head | Cloned (part of motor controller) | |
| Grounded wellness evaluators (5) | Cloned | Domain-scoped where applicable |
| Approximate wellness evaluator | Cloned | Per-mind, for imagination scoring |
| Context buffer | Fresh (empty) | Begins accumulating from split |
| PE history | Fresh | Begins tracking from split |
| Inter-mind state projection | New (learned linear) | Projects last hidden layer → d_model |

The parent retains its full-scope versions. After split:
- Parent's motor controller no longer generates commands for child's handles (naturally — those handles' wellness gradients are near zero in the parent's broader evaluator because the child is handling them well)
- Parent's forward model still predicts full sensory stream (it can imagine what the child's domain will do)
- Child's forward model receives a full deepcopy of the parent's forward model and retains all 6 modality inputs, including global vision and audio. "for_domain" refers to how the child's function is specialized through domain-scoped wellness and training, not to architectural pruning of the model itself.

### 2.13 Handle-to-Sensor Domain Mapping

When a mind splits, its motor domain is a set of PH handle indices. But wellness evaluators and the forward model also need to know which *sensors* belong to that domain. This mapping already exists in the codebase through `EBodySet` and `FHandleDefinition`.

#### The Mapping Chain

```
handle_index → FHandleDefinition.BodySet → EBodySet → sensor domains
```

**Handle → BodySet** (from `MotorCortexComponent::BuildHandleDefinitions()`):

| Index | Handle | BodySet |
|-------|--------|---------|
| 0 | Wrist_L | Arm_L |
| 1 | Thumb_L | Hand_L |
| 2 | Index_L | Hand_L |
| 3 | Middle_L | Hand_L |
| 4 | Ring_L | Hand_L |
| 5 | Pinky_L | Hand_L |
| 6 | Wrist_R | Arm_R |
| 7 | Thumb_R | Hand_R |
| 8 | Index_R | Hand_R |
| 9 | Middle_R | Hand_R |
| 10 | Ring_R | Hand_R |
| 11 | Pinky_R | Hand_R |
| 12 | Head | Head |
| 13 | UpperArm_L | Arm_L |
| 14 | LowerArm_L | Arm_L |
| 15 | UpperArm_R | Arm_R |
| 16 | LowerArm_R | Arm_R |
| 17 | Thigh_L | Leg_L |
| 18 | Calf_L | Leg_L |
| 19 | Thigh_R | Leg_R |
| 20 | Calf_R | Leg_R |
| 21 | Foot_L | Foot_L |
| 22 | Foot_R | Foot_R |

**BodySet → Sensor Domains:**

| Sensor Type | Scoping Mechanism |
|-------------|-------------------|
| **Tactile (TMC patches)** | Each TMC patch has a `bone_hash` identifying which skeleton bone generated it. Bones map to body sets via `FHandleDefinition`. A mind's tactile domain = patches whose `bone_hash` resolves to a bone in the mind's body sets. |
| **Body set state** | Direct. `FMultiModalBodySetState` provides `Capacity` and `AccumulatedEnergy` per body set (10 entries in SHM). A mind reads only the entries for its body sets. |
| **Proprioception** | Each of the 12 proprio bone entries has a `bone_index` field. Bones map to body sets. A mind's proprio domain = entries whose bone belongs to the mind's body sets. |
| **Vision** | Global. All minds receive the full vision token. No scoping. |
| **Audio** | Global. All minds receive the full audio token. No scoping. |

#### The `handle_set_to_sensor_scope` Function

```python
# TO BE BUILT — this function does not yet exist in the codebase.
# The body_sets.py module (Python-side handle-to-body-set mapping) is the
# foundation being built for this. This pseudocode defines the target API.
def handle_set_to_sensor_scope(handle_indices: Set[int], handle_defs: List[FHandleDefinition]) -> SensorScope:
    """Given a mind's handle set, compute which sensor dimensions belong to it."""
    body_sets = {handle_defs[i].body_set for i in handle_indices}
    bones = {handle_defs[i].bone_name for i in handle_indices}
    return SensorScope(
        body_sets=body_sets,
        bone_names=bones,
        # Tactile: filter patches by bone_hash ∈ bones
        # Proprio: filter entries by bone_index ∈ bones
        # Body set state: filter by body_set ∈ body_sets
        # Vision/Audio: always included (global)
    )
```

This runs once at split time and determines how the child's wellness evaluators are scoped:
- `prediction_error`: child's forward model PE (already domain-scoped by the forward model itself)
- `tactile_comfort`: `mean_pressure` over patches in `SensorScope.bone_names` only
- `capability`: `body_set_capacities` for `SensorScope.body_sets` only
- `motor_state`: ROM positions for handles in `handle_indices` only
- `coherence`: body schema correlations for handles in `handle_indices` only

### 2.12 Radius and Hierarchy Are Complementary

The radius parameter (motor synergy scope, 0-1) and the mind hierarchy address different dimensions:

- **Radius** = within-mind motor synergy. How broadly a single command affects nearby joints. Fine (radius→0, single fingertip) vs. gross (radius→1, whole-arm sweep) control within one mind's handle set.
- **Mind hierarchy** = across-mind temporal coordination. Arm reaches, then hand grasps. Sequential and parallel delegation across minds.

They compose: the Arm_L mind uses high radius for gross reaching. The Hand_L mind uses low radius for fine finger positioning. Neither needs to learn the other's policy. The inter-mind channel enables the hand to pre-shape during the reach by reading the arm's motor intent.

---

## 3. Open Gaps

### Gap A — RESOLVED (see Section 2.9)

What exactly is in the inter-mind state vector? Section 2.9 answers: a learned projection of the forward model's last hidden layer — the output of the final transformer block before prediction heads. This is the mind's compressed understanding of "what's happening right now and what's about to happen next." It encodes cross-modal and temporal context through 8 attention blocks. Each mind projects this to a fixed-size `[d_model]` vector via a learned linear layer.

**UEMA Spec coverage:** §16.1 `AlertSignal.anomaly_context` uses `forward_model.get_last_hidden()` — consistent. §16.3 defines `InterMindToken.content: Tensor[d_model]` — consistent. However, the UEMA spec wraps this in a prescribed protocol (alert types, urgency thresholds, override flags) that v2 replaces with learned attention over raw internal states.

### Gap B — RESOLVED: Wellness Gradient Combination

Unified wellness scalar eliminates the combination problem. All minds' wellness evaluations are summed into W_sys, then one gradient is taken via autograd. By linearity of differentiation, this equals summing individual gradients — but no separate gradient vectors ever exist that need a combination rule.

**Training (Stage A):** Unified system wellness gradient.
```
W_sys = sum(mind.wellness_evaluator(P_t, pe_i).sum() for mind in hierarchy.all_minds())
system_grad = torch.autograd.grad(W_sys, motor_commands, create_graph=True)[0]
```
Each motor controller receives the unified gradient and learns when to defer (parent's broader wellness threatened) and when to act (only its domain has signal). Scope IS the weighting: the parent's broader evaluator fires on more dimensions when it detects a problem, contributing more to the gradient. No explicit priority weighting needed.

**Inference (Stage B):** Confidence-weighted blend (forward pass only, no backward pass).
```
weights_h = softmax([conf_i for mind_i if h in mind_i.handles] / temperature)
final_cmd[h] = sum(weights_h[i] * cmds_i[h] for i in competing_minds)
```
Confidence is a learned output of each motor controller's confidence head (§9.2). The softmax temperature should be per-handle and learned. Whether broader-scope minds actually produce appropriately dominant confidence for contested handles is an empirical consequence of the training mechanism, not independently guaranteed at runtime.

Oscillation is naturally damped by three existing mechanisms: continuous physical inputs change smoothly, the context buffer provides temporal context, and fluctuating wellness produces low confidence which triggers deeper imagination (temporal smoothing).

**Contradictions resolved:** UEMA §17 (binary arbitration with override flags) is fully incompatible with this approach and is replaced entirely.

### Gap C — RESOLVED (programmed trigger, see below)

What triggers the split operation? **The UEMA Spec §15.1 answers this concretely** with `MasteryDetector`, and `MasteryTracker` in `curriculum.py:111-178` already implements the core tracking logic with matching thresholds:

```
PE_THRESHOLD = 0.05      # mean PE below this
VARIANCE_THRESHOLD = 0.01 # PE variance below this
WINDOW = 50,000 steps    # evaluation window
CONSISTENCY = 0.95       # fraction of window meeting criteria
```

The existing `MasteryTracker` uses a rolling window and checks these exact criteria for curriculum phase advancement. The trigger mechanism itself is already built. The remaining gap is **per-domain PE input** — `MasteryTracker` currently receives a single scalar `mean_pe`, not per-handle or per-domain breakdowns. To support Many Minds splitting, it needs per-handle PE from the loss manager so it can detect mastery within specific body regions, not just globally.

### Gap D — RESOLVED: Domain Boundaries at Split Time

Domain boundaries emerge from the model's own learned coordination, not from predefined groups. The `_get_domain_groups()` dictionary from the Implementation Plan is removed — it was a prescribed protocol that violates the foundational constraint.

**Mechanism:** Correlation-based domain discovery from existing signals. MasteryDetector's `find_mastered_domains()` is revised to: (1) compute per-handle PE over the evaluation window, (2) identify handles whose PE is consistently below threshold, (3) among those low-PE handles, cluster by motor-to-proprioceptive correlation from the body schema's EMA correlation buffers (`correlation_matrix: [n_handles, n_handles × n_proprio_per_handle]` in `body_schema.py`), (4) return each correlated cluster as a candidate domain for splitting. Correlation threshold is a hyperparameter (initial: 0.6).

**Sub-questions resolved:**
1. **Clustering algorithm:** Simple thresholding on the body schema correlation matrix. Not spectral clustering or k-means. The body's physics makes the right handles correlate.
2. **Kinematic chain constraints:** Emergent, not prescribed. Handles on the same kinematic chain show high motor→proprio correlation because moving one affects the others through the skeleton. Thumb_L will never split away from Hand_L because their activations are always correlated during grasping.
3. **Multiple children at once:** Yes. If multiple distinct clusters are identified simultaneously, all can split. Rare in practice because developmental constraints stagger mastery.
4. **Minimum mind size:** One handle. A handle with low PE that does not correlate strongly with any other handle forms its own domain (e.g., Gaze/Head at index 12).

**`clone_for_domain()` resolved:** Full deepcopy of the parent's weights (see Gap H). "for_domain" refers to domain-scoped wellness training, not architectural pruning. Sensor scoping via `handle_set_to_sensor_scope()` (§2.13).

**Contradiction resolved:** Implementation Plan §13.10 note 3 ("Domain groups are predefined, not discovered") is reversed. The Implementation Plan itself acknowledged this was a stopgap.

### Gap E: Disagreement between confident minds

Two minds may both be confident within their respective scopes while still pulling behavior in different directions. A narrow habitual controller may be confident locally while a broader mind carries concern about a larger consequence.

Training-side resolution is already given by Gap B: all minds' wellness evaluations are summed into one system wellness scalar and one autograd call produces the unified training gradient. This teaches the hierarchy when broader-scope concern should reshape lower-level behavior.

Inference-side resolution does NOT use runtime gradients. Runtime uses forward-pass quantities only. The broader mind's concern propagates through z_intermind, changing the receiving mind's latents, predictions, approximate wellness estimate, confidence, and motor output. At the top-mind ↔ attending-mind interface, the concrete mechanism is the attending engagement signal (§2.5.1). Strong broader-scope concern raises g_engage, pulling the attending mind into executive coordination. Weak broader-scope concern leaves routine lower-level control largely undisturbed.

So disagreement is resolved by:
1. unified wellness during training,
2. readable inter-mind state during inference,
3. learned attending-side engagement modulation for the apex relation,

not by override flags, ownership rules, or runtime gradient combination.

### Gap F: Left brain / right brain implications for initial split topology

The architecturally important split is not hemispheric. It is attentional: a present-world attending mind versus a broader top mind.

The top mind:
- broadest perception space,
- longest temporal horizon,
- deepest self-model,
- no PH handles,
- no direct motor output,
- influences behavior by shifting the attending mind's computation through z_intermind.

The attending mind:
- full-body present-world perception space,
- own grounded self-model,
- present-world motor coordinator,
- retains broad online responsibility for navigation, situational judgment, route decisions, gaze allocation, and overall embodied coordination,
- reads the top mind continuously through z_intermind.

Below the attending mind sits the motor hierarchy whose topology emerges from correlation-based domain discovery (Gap D). Specialist child minds handle mastered local domains, but the attending mind remains the executive coordinator for the current embodied situation.

This makes the apex split compatible with dual-task behavior. The top mind can devote capacity beyond the immediate physical present only when the attending mind's PE is low enough that attending-side engagement remains low. When the physical situation becomes urgent or unstable, the attending mind's state pulls the top mind back in, and the top mind immediately reshapes present-world behavior by raising attending engagement through z_intermind.

So the meaningful question is not "left vs. right hemisphere," but "when does the system benefit from separating broad present-world coordination from broader long-horizon cognition?"

### Gaps G and J — RESOLVED: Inter-Mind Bootstrapping

Gaps G (child can't use inter-mind signals post-split) and J (system transition from zero inter-mind to active inter-mind) are the same cold-start problem at different scales. Three mechanisms using existing architecture eliminate it:

**1. Pre-split self-reading.** During single-mind training, the one mind reads its own last-hidden-layer through the inter-mind channel instead of zeros. The inter-mind encoder processes real neural activations from step 0. Its `loss_weight` is set to a nonzero value (initial: 0.1) from the beginning. The target `PerMindIntermindEncoder` (§2.4) uses cross-attention with a learned identity embedding as query over projected internal states — not a trivial identity mapping. The current `SelfAttentionPoolEncoder` stub must be replaced or extended to accept the forward model's last hidden layer as input.

**2. Deepcopy continuity.** At split time, the child is a deepcopy. Its internal state is nearly identical to what the parent was producing for that domain. The inter-mind token the parent reads from the child looks almost the same as what it was self-reading. Minimal distribution shift at the moment of split.

**3. Gradual divergence.** After split, the child specializes and the parent continues learning. Their internal states diverge gradually, giving both inter-mind encoders time to adapt through normal training. No abrupt distribution shift. This applies to every split including the top-mind split.

### Gap H — RESOLVED: Subordinate Forward Model Scope

The child receives a full deepcopy of the parent's forward model. No architectural pruning is performed at split time.

The child retains all 6 modality inputs, including global vision and audio. This is required because domain prediction still depends on non-domain context: for example, a hand mind needs vision to predict contact, and a local controller may still need broader body context to predict the sensory consequences of its actions.

Domain specialization therefore does not come from cutting away inputs or shrinking the forward model. It emerges through training. The child's wellness evaluators are domain-scoped, so learning pressure is concentrated on the child's own handles, body sets, and sensor consequences. The child uses global modalities as context, but its functional role becomes increasingly specialized to its domain.

The parent retains full-stream prediction and can still imagine what the child's domain will do. The child retains full-context input but becomes increasingly domain-specialized through training. This resolves `clone_for_domain()` as follows: clone means full deepcopy of the model; "for_domain" refers to functional specialization through domain-scoped learning, not pruning of architecture.

Top-mind specialization is a separate issue from child scope. After the apex split, the top mind diverges through longer imagination chains. The attending mind keeps the standard short imagination depth (3–5 steps). The top mind uses a larger per-mind imagination depth (initial: 16+, subject to tuning), and imagination loss at those longer horizons is expected to bias it toward longer-timescale predictive structure. Whether this produces the desired long-horizon specialization remains an empirical question to validate, but the architectural definition of child forward-model scope is resolved.

### Gap I — RESOLVED: Optimizer Mechanics of Natural Freezing/Thawing

Natural gradient dynamics with optimizer state eviction for memory efficiency. No `is_training` flag. No `learning_rate = 0.0`. A mastered mind has near-zero PE, produces near-zero gradients, and its weights naturally stop changing.

**Memory optimization:** If a mind's gradient norm has been below threshold for the MasteryDetector window (same 50K steps), its Adam optimizer state is evicted from GPU memory. The mind still runs inference, computes wellness, generates motor commands, and monitors its domain. It just doesn't update weights because gradients are naturally near-zero.

**Natural thaw:** If PE rises (novel situation), gradients become nonzero, optimizer state is reinitialized with fresh Adam momentum, and training resumes. Reinitializing Adam after eviction loses accumulated variance estimates, which could cause briefly larger weight updates as estimates restabilize. This is bounded by gradient clipping (`grad_clip=1.0`) and is appropriate for a mind encountering genuinely novel conditions where old estimates were stale.

**Contradiction resolved:** UEMA §15.2's explicit freeze (`is_training = False`, `learning_rate = 0.0`) is replaced. Natural gradient dynamics are preserved; the memory optimization introduces a small training discontinuity at thaw time that pure uninterrupted dynamics would not have.

### Gap K — RESOLVED (see Section 2.13)

### Gap J — RESOLVED (merged with Gap G, see above)

---

## 4. Superseded Ideas from v1

The following ideas from ManyMinds_Spec_v1.md are superseded by this revision:

| v1 Concept | Status | Reason |
|-----------|--------|--------|
| Hardcoded message types (alert, override, release, status, goal, complete) | **Removed** | Contradicts UEMA philosophy. Communication must be learned, not prescribed as enum types. |
| Override/release as explicit operations with flags | **Removed** | Override is emergent from higher mind focusing (stronger outputs). Release is emergent from higher mind losing interest (outputs fade). |
| Motor channel ownership sets | **Removed** | Replaced by confidence-weighted competition. No mind "owns" handles. |
| GoalSignal / CompletionSignal classes | **Removed** | Prescribed protocol. If goal-like and completion-like behaviors are needed, they must emerge from learned inter-mind state patterns. |
| ThawDetector with programmed thresholds | **Removed** | Thawing should be natural (gradients increase when PE increases). Programmed thaw triggers contradict the design principle. |
| `override_active` flag on Mind | **Removed** | No flags. Competition determines control. |
| MotorArbitrator with ownership rules | **Removed** | Replaced by confidence-weighted blending. |

---

## 5. What Remains from v3.4

The following v3.4 concepts are retained and compatible with this revision:

- **SensoryBus** with shared sensory token broadcast (5 modalities)
- **Per-mind forward model** running over shared tokens
- **Mastery detection** concept (low PE + low variance over window) — trigger mechanism resolved (Gap C, programmed `MasteryDetector` from §15.1)
- **InterMind encoder** concept — reinterpreted as internal state reader, not message encoder
- **Growth trajectory** from single mind to hierarchy — extended to allow non-top splitting
- **Mind data structure** core fields (mind_id, parent, children, depth, forward_model, motor_controller, wellness_evaluator, pe_history)
