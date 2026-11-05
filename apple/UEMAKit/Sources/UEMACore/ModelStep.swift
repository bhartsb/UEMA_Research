// ModelStep.swift: the count of model steps and the fixed timing of one model step.
// Work item WI-20, target B of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). Plain Swift: no import beyond the standard library.
// The "First build on a Mac" checklist of this work item is in Protocols.swift's header.
//
// SOURCES READ BEFORE WRITING (November 4, 2026), each with the fact it gave:
// - README.md line 13: "(one model step is 50 ms of simulated time: ...)".
// - docs/UEMA_APPLE_PORT_SPECIFICATION_2026-11-04.html, "The setting": "Every model step
//   (one 50 ms model update: 5 env steps of 10 ms, each 2 physics steps of 5 ms
//   [I2 env.py:1157])".
// - Source/Python/mimo_bridge/env.py line 1157: "self.env = EnvCls(model_path=base_path,
//   initial_qpos=initial_qpos, frame_skip=2," (2 physics steps per env step).
// - third_party/MIMo/mimo/mimo_infant/simulation/envs/mimo_env.py lines 573-575:
//   "def _single_mujoco_step(self):" / "mujoco.mj_step(self.model, self.data)" /
//   "mujoco.mj_rnePostConstraint(self.model, self.data)"; lines 616-619: "for _ in
//   range(n_frames):" / "self.actuation_model.substep_update()" /
//   "self._single_mujoco_step()" / "self._substep_callback()".
// - apple/UEMAKit/Specs/command_decoder.md section 1: "the trainer calls 5 times per model
//   step (`action_repeat` 5, weights/b1_continue_B1K_skip13_492800/run/args.json:154;
//   train_mimo.py:2346, :3030)".
// - weights/b1_continue_B1K_skip13_492800/run/trainer.log line 5: "ctx=128".
// - Source/Python/mimo_bridge/lem_model.py line 4283: "if len(self.goal_ring) >= 20 and
//   step % self.intention_every == 0:", with intention_every 10 (planner_loss.md section 1:
//   "`intention_every` 10, args.json:62"); line 998: "self.goal_ring: deque =
//   deque(maxlen=200)".
// - docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html row WI-25's pass line: "steps
//   20 model steps (1.0 s)".

/// A count of model steps since the session started, with the fixed timing of one model
/// step: 50 ms of simulated time (README.md:13), run as 5 env steps of 10 ms, each 2 physics
/// steps of 5 ms (specification "The setting"; env.py:1157). The count is the Python
/// trainer's `step`: the planner's cadence test `step % 10 == 0` (lem_model.py:4283) reads it.
public struct ModelStep: Equatable, Hashable, Comparable, Codable, Sendable {
    /// Env steps per model step: 5 (`action_repeat` 5, args.json:154).
    public static let envStepsPerModelStep = 5

    /// Physics steps per env step: 2 (`frame_skip=2`, env.py:1157), each one mj_step followed
    /// by mj_rnePostConstraint (mimo_env.py:573-575).
    public static let physicsStepsPerEnvStep = 2

    /// Physics steps per model step: 5 x 2 = 10.
    public static let physicsStepsPerModelStep = envStepsPerModelStep * physicsStepsPerEnvStep

    /// The physics timestep in seconds: 0.005 (specification "The setting": "2 physics steps
    /// of 5 ms"; plan section 2, stage 1: "a 0.005 s timestep").
    public static let physicsTimestepSeconds = 0.005

    /// Seconds per env step: 0.005 x 2 = 0.010.
    public static let envStepSeconds = physicsTimestepSeconds * Double(physicsStepsPerEnvStep)

    /// Seconds per model step: 0.010 x 5 = 0.050.
    public static let seconds = envStepSeconds * Double(envStepsPerModelStep)

    /// Milliseconds per model step: 50 (README.md:13).
    public static let milliseconds = 50

    /// The forward model's context in model steps: 128 (trainer.log line 5, "ctx=128");
    /// 6 tokens x 128 = 768 positions (specification REQ-B-8).
    public static let contextLength = 128

    /// Model steps between planning events: 10 (`intention_every` 10, args.json:62;
    /// lem_model.py:4283).
    public static let planningInterval = 10

    /// Entries the goal ring keeps: the last 200 model steps (lem_model.py:998).
    public static let goalRingLength = 200

    /// Entries the goal ring must hold before a planning event runs: 20 (lem_model.py:4283).
    public static let goalRingMinimumForPlanning = 20

    /// The count of model steps completed since the session started; 0 before the first.
    public var count: Int

    /// A model step count; `count` must be 0 or more. A negative count is a programming error
    /// and stops the process (`precondition`); a negative count read from a file throws instead
    /// (`init(from:)`).
    public init(count: Int) {
        precondition(count >= 0, "a model step count is 0 or more; got \(count)")
        self.count = count
    }

    /// The count before the first model step: 0.
    public static let start = ModelStep(count: 0)

    /// Simulated seconds these model steps span: count x 0.050 (20 model steps = 1.0 s, the
    /// WI-25 pass line).
    public var simulatedSeconds: Double { Double(count) * ModelStep.seconds }

    /// Simulated milliseconds these model steps span: count x 50.
    public var simulatedMilliseconds: Int { count * ModelStep.milliseconds }

    /// The count `steps` model steps later (1 by default); the result must be 0 or more, as for
    /// `init(count:)`.
    public func advanced(by steps: Int = 1) -> ModelStep {
        ModelStep(count: count + steps)
    }

    /// True when `count % 10 == 0`, the cadence half of the Python planning condition
    /// (lem_model.py:4283). The other half, the goal ring holding at least 20 entries, is
    /// `Planning.canPlan(at:)`.
    public var isPlanningCandidate: Bool { count % ModelStep.planningInterval == 0 }

    /// True when `lhs` counts fewer model steps than `rhs`.
    public static func < (lhs: ModelStep, rhs: ModelStep) -> Bool {
        lhs.count < rhs.count
    }

    private enum CodingKeys: String, CodingKey {
        case count
    }

    /// Decodes `{"count": n}`; a negative `n` throws `DecodingError.dataCorrupted` rather than
    /// stopping the process the way `init(count:)` does.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let count = try container.decode(Int.self, forKey: .count)
        guard count >= 0 else {
            throw DecodingError.dataCorruptedError(
                forKey: .count, in: container,
                debugDescription: "a model step count is 0 or more; got \(count)")
        }
        self.count = count
    }
}
