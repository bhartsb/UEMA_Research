// ReleasedSceneCommandTests.swift: CommandDecoder.released over MuJoCoPhysics on the released crib
// scene, steps 3 and 4 of the "First build on a Mac" checklist in
// Sources/UEMACommand/CommandDecoder.swift (work item WI-37).
//
// INPUT. Sources/UEMAScene/Resources/crib_age24_trunk_v1/scene.xml, found from this file's own
// path the way MuJoCoPhysicsTests finds it (#filePath three folders up is apple/UEMAKit).
//
// CHECKS.
//  - Step 3: `CommandDecoder.released(physics:)` builds on the scene (nu 90 = MotorControls.count),
//    and the position reach of each position handle equals model_receipt.json
//    motor_synergy.pos_reach_cm (wrists 42.85, thumbs 5.54, index 5.48, middle 6.09, ring 5.48,
//    pinky 6.80, head 4.81, lower_trunk 5.58 cm, quoted in that checklist) within 0.01 cm.
//  - Step 4, the pattern of Source/Python/mimo_bridge/tests/test_command_moves_body.py: from the
//    reset state, one command held for 25 env steps (a fresh decode each env step) with a
//    40-degree turn about x as the orientation target and strength 1 on one of upperarm_R,
//    upperarm_L, head, thigh_R moves some qpos entry by more than 1e-3; the same command at
//    strength 0 moves less than at strength 1; every control lies within -1 to 1.
//
// First run: the Linux harness of apple/FIRST_BUILD_CHECKLIST.md (Swift 6.2, MuJoCo 3.3.0 for
// Linux), November 5, 2026.

import Foundation
import UEMACommand
import UEMACore
import UEMAPhysics
import XCTest

final class ReleasedSceneCommandTests: XCTestCase {
    /// apple/UEMAKit/Sources/UEMAScene/Resources/crib_age24_trunk_v1/scene.xml.
    static var sceneURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // UEMACommandTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // UEMAKit
            .appendingPathComponent("Sources/UEMAScene/Resources/crib_age24_trunk_v1/scene.xml")
    }

    /// The released scene loaded and its released decoder.
    func loaded() throws -> (MuJoCoPhysics, CommandDecoder) {
        let physics = MuJoCoPhysics()
        try physics.load(sceneXML: Self.sceneURL)
        return (physics, try CommandDecoder.released(physics: physics))
    }

    /// Step 3: the decoder builds on nu 90 and its position reach matches the receipt.
    func testReleasedDecoderBuildsWithTheReceiptReach() throws {
        let (physics, decoder) = try loaded()
        XCTAssertEqual(physics.sizes.nu, MotorControls.count)
        let engine = try XCTUnwrap(decoder.engine as? HandleCommandDecoder)
        let expectedCm: [String: Double] = [
            "wrist_L": 42.85, "wrist_R": 42.85, "thumb_L": 5.54, "thumb_R": 5.54,
            "index_L": 5.48, "index_R": 5.48, "middle_L": 6.09, "middle_R": 6.09,
            "ring_L": 5.48, "ring_R": 5.48, "pinky_L": 6.80, "pinky_R": 6.80,
            "head": 4.81, "lower_trunk": 5.58,
        ]
        XCTAssertEqual(engine.layout.positionHandles, Set(expectedCm.keys))
        for (row, name) in engine.layout.names.enumerated() {
            guard let cm = expectedCm[name] else { continue }
            XCTAssertEqual(engine.handles[row].reachMetres * 100, cm, accuracy: 0.01, name)
        }
    }

    /// The qpos travelled, per entry, by `command` held for 25 env steps from the reset state.
    func hold(_ command: Command, physics: MuJoCoPhysics, decoder: CommandDecoder) throws -> [Double] {
        try physics.reset()
        let before = physics.qpos
        for _ in 0 ..< 25 {
            let controls = try decoder.decode(command, mask: .allTrue, gaze: nil, fingerGoals: nil)
            XCTAssertTrue(controls.isWithinRange)
            try physics.stepEnvStep(controls: controls, afterEachPhysicsStep: { _ in })
        }
        return zip(physics.qpos, before).map { abs($0 - $1) }
    }

    /// test_command_moves_body.py `_null_motor` with one handle turned 40 degrees about x.
    func turnCommand(_ handle: BodyHandle, strength: Float) -> Command {
        var command = Command.zero(scale: .centimetres)
        for row in BodyHandle.allCases { command[row, .quaternionW] = 1 }
        let half = Float(40.0 * Double.pi / 180.0 / 2.0)
        command[handle, .quaternionX] = sin(half)
        command[handle, .quaternionW] = cos(half)
        command[handle, .radius] = 1
        command[handle, .strength] = strength
        return command
    }

    /// Step 4: a full-strength held command moves the body part it names.
    func testAHeldCommandMovesTheBodyPartItNames() throws {
        let (physics, decoder) = try loaded()
        for handle in [BodyHandle.upperarmR, .upperarmL, .head, .thighR] {
            let moved = try hold(turnCommand(handle, strength: 1), physics: physics, decoder: decoder)
            let largest = moved.max() ?? 0
            print("WI-37 released scene: \(handle) full strength moved qpos by \(largest)")
            XCTAssertGreaterThan(largest, 1e-3, "\(handle)")
        }
    }

    /// Step 4: strength decides the motion.
    func testZeroStrengthMovesLessThanFullStrength() throws {
        let (physics, decoder) = try loaded()
        let strong = try hold(turnCommand(.upperarmR, strength: 1), physics: physics, decoder: decoder).max() ?? 0
        let weak = try hold(turnCommand(.upperarmR, strength: 0), physics: physics, decoder: decoder).max() ?? 0
        print("WI-37 released scene: upperarm_R strength 1 moved \(strong), strength 0 moved \(weak)")
        XCTAssertGreaterThan(strong, weak)
    }
}
