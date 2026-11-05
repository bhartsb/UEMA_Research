// WellnessTests.swift: the tests of Sources/UEMAModel/Wellness.swift (work item WI-36). Nothing in
// this file was compiled on the machine that wrote it (no Swift toolchain on Windows).
//
// The expected numbers of the motor_state and tactile_comfort tests were printed on November 4,
// 2026 by the released Python code itself, run with torch in the training environment from
// Source/Python: WellnessEvaluators._formula_motor_state_boundary (wellness.py:334-395) with
// rom_limit 1.0 and torch.autograd.grad of its sum with respect to the centred pose (the
// derivative lem_model.py:3101-3106 takes), and contact_response (contact_wellness.py:36-94) with
// profile 'mechanical_v4'. The inputs are written out in each test.

import XCTest
import Foundation
import MLX
import UEMACore
@testable import UEMAModel

final class WellnessTests: XCTestCase {

    // MARK: motor_state and the wellness gradient

    /// The handle state of the torch run: every number 0 except handle 0 x 0.30, handle 1 x
    /// -0.10, handle 2 y 0.195, handle 3 z -0.25, handle 4 y 0.40, handle 5 z 0.20.
    private func torchPoseHandleState() -> [Float] {
        var state = [Float](repeating: 0, count: 260)
        state[0 * 10 + 0] = 0.30
        state[1 * 10 + 0] = -0.10
        state[2 * 10 + 1] = 0.195
        state[3 * 10 + 2] = -0.25
        state[4 * 10 + 1] = 0.40
        state[5 * 10 + 2] = 0.20
        return state
    }

    /// Centring subtracts each axis's mean over the 26 handles: x mean 0.2 / 26.
    func testCentredPoseSubtractsTheHandleMean() throws {
        let pose = try MotorStateBoundary.centredPose(handleState: torchPoseHandleState())
        XCTAssertEqual(pose.count, 234)
        XCTAssertEqual(pose[0], 0.292307705, accuracy: 1e-6)
        XCTAssertEqual(pose[3 * 9 + 2], -0.248076916, accuracy: 1e-6)
        XCTAssertEqual(pose[4 * 9 + 1], 0.377115399, accuracy: 1e-6)
        XCTAssertEqual(pose[5 * 9 + 2], 0.201923072, accuracy: 1e-6)
        // columns 3 to 8 are copied: handle 0's column 3 (quaternion x) is 0 here
        XCTAssertEqual(pose[3], 0)
    }

    /// torch printed: score 0.956683338; gradient -1.22100127 at (handle 0, x), 1.22100127 at
    /// (3, z), -1.22100127 at (4, y) (three numbers past the reach: slope 2), -0.75138545 at
    /// (5, z) (q 0.615 inside the band: slope 2q); every other number 0.
    func testScoreAndGradientMatchTorch() throws {
        let boundary = MotorStateBoundary()
        let pose = try MotorStateBoundary.centredPose(handleState: torchPoseHandleState())
        XCTAssertEqual(try boundary.score(centredPose: pose), 0.956683338, accuracy: 1e-6)
        let gradient = try boundary.gradient(centredPose: pose)
        let expected: [Int: Float] = [0 * 9 + 0: -1.22100127, 3 * 9 + 2: 1.22100127,
                                      4 * 9 + 1: -1.22100127, 5 * 9 + 2: -0.75138545]
        for index in 0 ..< 234 {
            XCTAssertEqual(gradient[index], expected[index] ?? 0, accuracy: 1e-5, "number \(index)")
        }
    }

    /// Every handle inside 90 % of the reach (0.189 position_range units = 37.8 cm): score 1,
    /// gradient 0.
    func testInteriorPoseScoresOneWithZeroGradient() throws {
        var state = [Float](repeating: 0, count: 260)
        state[0] = 0.15
        state[10 + 1] = -0.15
        let boundary = MotorStateBoundary()
        let pose = try MotorStateBoundary.centredPose(handleState: state)
        XCTAssertEqual(try boundary.score(centredPose: pose), 1)
        XCTAssertTrue(try boundary.gradient(centredPose: pose).allSatisfy { $0 == 0 })
    }

    // MARK: tactile_comfort

    /// torch printed, for four sites (pressure, area, vibration 4-7; history):
    ///   site 0: 0.05, 0.2, (0.5, 0.1, -0.2, 0); history 0
    ///   site 1: 2.5, 0.1, 0; history (300000 Pa, 2.0e6 Pa s, 500 J/m2)
    ///   site 2: all 0; history 0
    ///   site 3: 0.01, area 0, 0; history (0, 0, 100 J/m2)
    /// comfort 0.346667558, pain 0.42765221, pleasant 0.301982671.
    func testTactileComfortMatchesTorch() throws {
        func fields(_ pressure: Float, _ area: Float, _ vibration: [Float]) -> [Float] {
            [pressure, 0, 0, area] + vibration + [0, 0]
        }
        let sites = [
            try SkinSite(fields: fields(0.05, 0.2, [0.5, 0.1, -0.2, 0])),
            try SkinSite(fields: fields(2.5, 0.1, [0, 0, 0, 0]), history: [300_000, 2.0e6, 500]),
            try SkinSite(fields: fields(0, 0, [0, 0, 0, 0])),
            try SkinSite(fields: fields(0.01, 0, [0, 0, 0, 0]), history: [0, 0, 100]),
        ]
        let result = try TactileComfortChannel().evaluate(sites: sites)
        XCTAssertEqual(result.comfort, 0.346667558, accuracy: 1e-6)
        XCTAssertEqual(result.pain, 0.42765221, accuracy: 1e-6)
        XCTAssertEqual(result.pleasant, 0.301982671, accuracy: 1e-6)
    }

    /// No tactile reading: comfort 0.5 (wellness.py:405-406); zero sites: refused.
    func testTactileWithoutReading() throws {
        XCTAssertEqual(try TactileComfortChannel().evaluate(sites: nil).comfort, 0.5)
        XCTAssertThrowsError(try TactileComfortChannel().evaluate(sites: [])) { error in
            XCTAssertEqual(error as? WellnessError, .noSkinSites)
        }
    }

    /// Twice the pressure reference (438,800 Pa) with no history: acute load 1, pain 1 / 2.
    func testMechanicalPainIsHyperbolic() {
        let pain = MechanicalLoadPain().localPain(pressurePascal: 438_800, history: [0, 0, 0], painGain: 1)
        XCTAssertEqual(pain, 0.5, accuracy: 1e-12)
        XCTAssertEqual(MechanicalLoadPain().localPain(pressurePascal: 100_000, history: [0, 0, 0], painGain: 1), 0)
    }

    // MARK: prediction_error, coherence, capability

    func testPredictionErrorExcludesVestibularAndUngradedSenses() throws {
        let zero = Token.zero
        let one = Token(repeating: 1)
        let half = Token(repeating: 0.5)
        let predicted: TokenSet = [.vision: zero, .tactile: zero, .vestibular: zero]
        let actual: TokenSet = [.vision: half, .tactile: one, .vestibular: one]
        let channel = PredictionErrorChannel()
        // vision (0.25) and tactile (1.0) real: 1 - 0.625
        XCTAssertEqual(channel.score(predicted: predicted, actual: actual, realSenses: [.vision, .tactile, .vestibular]),
                       0.375, accuracy: 1e-6)
        // tactile not graded real: vision alone, 1 - 0.25
        XCTAssertEqual(channel.score(predicted: predicted, actual: actual, realSenses: [.vision]), 0.75, accuracy: 1e-6)
        // nothing matched: 1
        XCTAssertEqual(channel.score(predicted: [:], actual: actual, realSenses: Set(SenseName.active)), 1)
    }

    func testCoherenceUsesTheBodyMapElseLikeness() {
        let channel = CoherenceChannel()
        let result = channel.score(extent: 1.4, predicted: [:], actual: [:])
        XCTAssertEqual(result.value, 1)
        XCTAssertEqual(result.source, .bodyMap)
        let a = Token(repeating: 1)
        let fallback = channel.score(extent: nil, predicted: [.vision: a, .vestibular: a], actual: [.vision: a, .vestibular: Token(repeating: -1)])
        XCTAssertEqual(fallback.value, 1, accuracy: 1e-6)
        XCTAssertEqual(fallback.source, .predictionLikeness)
    }

    func testCapabilityAveragesTheFirstTen() throws {
        var features = [Float](repeating: 0, count: 21)
        for index in 0 ..< 10 {
            features[index] = 0.8
        }
        features[15] = 5
        XCTAssertEqual(try CapabilityChannel().score(features: features), 0.8, accuracy: 1e-6)
        XCTAssertEqual(try CapabilityChannel().score(features: nil), 0)
        XCTAssertThrowsError(try CapabilityChannel().score(features: [1]))
    }

    // MARK: Shapes against the manifest

    /// The 26 wellness tensors the sum does not read equal the manifest's wellness component,
    /// 295,914 elements (receipt component_params.wellness).
    func testWellnessTensorTableEqualsTheManifest() throws {
        let manifest = try WeightManifest.bundled()
        var manifestShapes: [String: [Int]] = [:]
        for record in manifest.tensors(inComponent: "wellness") {
            manifestShapes[record.key] = record.shape
        }
        var portShapes: [String: [Int]] = [:]
        for entry in WellnessTensorTable.tensorShapes {
            portShapes[entry.key] = entry.shape
        }
        XCTAssertEqual(portShapes, manifestShapes)
        XCTAssertEqual(WellnessTensorTable.tensorShapes.count, 26)
        XCTAssertEqual(WellnessTensorTable.tensorShapes.reduce(0) { $0 + $1.shape.reduce(1, *) }, 295_914)
    }

    // MARK: The fixtures

    /// On each model-pass fixture: prediction_error from predicted/<sense> against token/<sense>
    /// equals wellness/prediction_error to 1e-6; motor_state from command/handle_state equals
    /// wellness/motor_state and the gradient equals command/wellness_grad; with the recorded
    /// tactile_comfort, capability, coherence and self_touch fed in, the sum equals wellness/total
    /// to 1e-5. In all three fixtures the 78 position numbers of command/handle_state are 0
    /// (measured with numpy on November 4, 2026: largest |centred position| 0.0), so motor_state is
    /// 1 and the gradient is 234 zeros there; testScoreAndGradientMatchTorch is the test of the
    /// formula's nonzero branch.
    func testFixtureWellnessSum() throws {
        for name in ControllerFixture.names {
            let fixture = try ControllerFixture(name: name)
            var predicted: TokenSet = [:]
            var actual: TokenSet = [:]
            for sense in SenseName.active {
                actual[sense] = try fixture.token("token/" + sense.rawValue)
                if fixture.has("predicted/" + sense.rawValue) {
                    predicted[sense] = try fixture.token("predicted/" + sense.rawValue)
                }
            }
            XCTAssertEqual(predicted.count, 5, "the motor sense has no prediction head")
            let recorded: (WellnessChannel) throws -> Float = { channel in
                try fixture.floats("wellness/" + channel.rawValue).first ?? .nan
            }

            let predictionValue = PredictionErrorChannel().score(predicted: predicted, actual: actual,
                                                                 realSenses: Set(SenseName.active))
            XCTAssertEqual(predictionValue, try recorded(.predictionError), accuracy: 1e-6, name)

            let handleState = try fixture.floats("command/handle_state")
            let boundary = MotorStateBoundary()
            let pose = try MotorStateBoundary.centredPose(handleState: handleState)
            XCTAssertEqual(try boundary.score(centredPose: pose), try recorded(.motorState), accuracy: 1e-6, name)
            let gradient = try boundary.gradient(centredPose: pose)
            let fixtureGradient = try fixture.floats("command/wellness_grad")
            XCTAssertEqual(fixtureGradient.count, 234)
            for index in 0 ..< min(234, fixtureGradient.count) {
                XCTAssertEqual(gradient[index], fixtureGradient[index], accuracy: 1e-6, "\(name) gradient \(index)")
            }

            let recordedCoherence = try recorded(.coherence)
            let coherence = CoherenceChannel().score(extent: recordedCoherence, predicted: predicted, actual: actual)
            let tactileValue = try recorded(.tactileComfort)
            let capabilityValue = try recorded(.capability)
            let motorValue = try boundary.score(centredPose: pose)
            let selfTouchValue = try recorded(.selfTouch)
            let channels: [WellnessChannel: Float] = [
                .predictionError: predictionValue,
                .tactileComfort: tactileValue,
                .capability: capabilityValue,
                .coherence: coherence.value,
                .motorState: motorValue,
                .selfTouch: selfTouchValue,
            ]
            let total = GroundedWellnessScorer.sum(channels)
            let fixtureTotal = try fixture.floats("wellness/total").first ?? .nan
            XCTAssertEqual(total, fixtureTotal, accuracy: 1e-5, name)
            print("wellness \(name): sum \(total), fixture \(fixtureTotal), |difference| \(abs(total - fixtureTotal))")
        }
    }

    /// The scorer end to end on hand-set inputs: every channel's number lands under its name and
    /// the total is their sum.
    func testScorerAddsTheSixChannels() throws {
        var capability = [Float](repeating: 1, count: 21)
        capability[0] = 0.5
        let inputs = WellnessInputs(
            predicted: [.vision: Token.zero], actual: [.vision: Token(repeating: 0.5)],
            realSenses: [.vision], skin: nil, capability: capability, coherenceExtent: 0.25,
            handleState: [Float](repeating: 0, count: 260))
        let score = try GroundedWellnessScorer().score(inputs)
        XCTAssertEqual(score.channels[.predictionError] ?? -1, 0.75, accuracy: 1e-6)
        XCTAssertEqual(score.channels[.tactileComfort], 0.5)
        XCTAssertEqual(score.channels[.capability] ?? -1, 0.95, accuracy: 1e-6)
        XCTAssertEqual(score.channels[.coherence], 0.25)
        XCTAssertEqual(score.channels[.motorState], 1)
        XCTAssertEqual(score.channels[.selfTouch], 0)
        XCTAssertEqual(score.total, 0.75 + 0.5 + 0.95 + 0.25 + 1, accuracy: 1e-5)
        XCTAssertEqual(score.gradient.count, 234)
        XCTAssertEqual(score.coherenceSource, .bodyMap)
    }
}
