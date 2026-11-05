// Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
// Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
//
// VisionReaderTests.swift: VisionReader (Sources/UEMASenses/Vision/VisionReader.swift) over the
// released crib scene in MuJoCoPhysics, with a renderer that fills each image with a number
// counting its renders, so no graphics library is needed. The scene is found from this file's
// path (#filePath four folders up is apple/UEMAKit), as MuJoCoPhysicsTests finds it.
//
// Checks: the eye joints' qpos addresses on the released scene (76, 77, 79, 80, Python mujoco
// 3.3.0); the reading's layout ([890,880] and [2, 2], the four patched cameras, no ring); the
// gaze sign rule of env.eye_yaw_pitch; the capture after the last env step of a burst being the
// next read; a stale repeat on a model step the cadence skips; reset.

import Foundation
import UEMACore
import UEMAPhysics
import UEMASenses
import XCTest

final class VisionReaderTests: XCTestCase {
    static var sceneURL: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // Vision
            .deletingLastPathComponent()  // UEMASensesTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // UEMAKit
            .appendingPathComponent("Sources/UEMAScene/Resources/crib_age24_trunk_v1/scene.xml")
    }

    func loaded() throws -> MuJoCoPhysics {
        let physics = MuJoCoPhysics()
        try physics.load(sceneXML: Self.sceneURL)
        return physics
    }

    /// Runs the 5 env steps of one model step at zero control, observing each.
    func runBurst(_ physics: MuJoCoPhysics, _ reader: VisionReader) throws {
        for _ in 0 ..< ModelStep.envStepsPerModelStep {
            try physics.stepEnvStep(controls: .zero, afterEachPhysicsStep: { _ in })
            try reader.observeEnvStep(physics)
        }
    }

    func testEyeJointAddressesOnTheReleasedScene() throws {
        let addresses = EyeJointAddresses(tables: try loaded().modelTables())
        XCTAssertEqual(addresses, EyeJointAddresses(leftHorizontal: 76, leftVertical: 77,
                                                    rightHorizontal: 79, rightVertical: 80))
        let gaze = try addresses.gaze(qpos: (0 ..< 146).map(Double.init))
        XCTAssertEqual(gaze.left, EyeGaze(yaw: 76, pitch: 77))
        XCTAssertEqual(gaze.right, EyeGaze(yaw: -79, pitch: 80))
        XCTAssertThrowsError(try addresses.gaze(qpos: [0]))
        let none = EyeJointAddresses(leftHorizontal: nil, leftVertical: nil, rightHorizontal: nil, rightVertical: nil)
        XCTAssertEqual(try none.gaze(qpos: []).left, .level)
    }

    func testFirstReadRendersTheResetStateInTheTokenizerLayout() throws {
        let physics = try loaded()
        let renderer = CountingRenderer()
        let reader = VisionReader(renderer: renderer, scene: physics,
                                  eyeJoints: EyeJointAddresses(tables: try physics.modelTables()))
        let reading = try reader.read(physics, at: .start)
        XCTAssertEqual(reading.sense, .vision)
        XCTAssertTrue(reading.isFresh)
        XCTAssertEqual(reading.primary.shape, [890_880])
        XCTAssertEqual(reading.auxiliary?.shape, [2, 2])
        XCTAssertEqual(renderer.rendered, [.eyeLeft, .eyeRight, .foveaLeft, .foveaRight])
        let unpacked = try VisionReadingLayout.released.images(of: reading)
        XCTAssertEqual(unpacked.images[.foveaRight]?.rgb.first, 4)
        XCTAssertNil(unpacked.images[.ringLeft])
    }

    func testCaptureAfterTheLastEnvStepIsTheNextReadWithItsGaze() throws {
        let physics = try loaded()
        let renderer = CountingRenderer()
        let reader = VisionReader(renderer: renderer, scene: physics,
                                  eyeJoints: EyeJointAddresses(tables: try physics.modelTables()))
        _ = try reader.read(physics, at: .start)
        try runBurst(physics, reader)
        XCTAssertEqual(renderer.rendered.count, 8, "one capture per burst, after env step 4")
        let expected = try EyeJointAddresses(tables: try physics.modelTables()).gaze(qpos: physics.qpos)
        // A later state change does not reach the captured reading.
        var qpos = physics.qpos
        qpos[76] += 0.25
        try physics.setState(qpos: qpos, qvel: physics.qvel)
        let reading = try reader.read(physics, at: ModelStep(count: 1))
        XCTAssertTrue(reading.isFresh)
        let unpacked = try VisionReadingLayout.released.images(of: reading)
        XCTAssertEqual(unpacked.images[.eyeLeft]?.rgb.first, 5)
        XCTAssertEqual(unpacked.left.yaw, Double(Float(expected.left.yaw)))
        XCTAssertEqual(unpacked.right.yaw, Double(Float(expected.right.yaw)))
        XCTAssertEqual(renderer.rendered.count, 8)
    }

    func testSkippedModelStepRepeatsTheLastReadingAsStale() throws {
        let physics = try loaded()
        let renderer = CountingRenderer()
        let reader = VisionReader(renderer: renderer, scene: physics,
                                  eyeJoints: EyeJointAddresses(tables: try physics.modelTables()),
                                  cadence: .everySecondModelStep)
        _ = try reader.read(physics, at: .start)  // renders: nothing captured yet
        try runBurst(physics, reader)                // model step 0 renders (0 % 2 == 0)
        let one = try reader.read(physics, at: ModelStep(count: 1))
        XCTAssertTrue(one.isFresh)
        try runBurst(physics, reader)                // model step 1 does not render
        let two = try reader.read(physics, at: ModelStep(count: 2))
        XCTAssertFalse(two.isFresh)
        XCTAssertEqual(two.primary, one.primary)
        XCTAssertEqual(renderer.rendered.count, 8)

        reader.reset()
        let afterReset = try reader.read(physics, at: .start)
        XCTAssertTrue(afterReset.isFresh)
        XCTAssertEqual(renderer.rendered.count, 12)
    }
}

/// A renderer whose k-th image (counting from 1) has every byte equal to k mod 256.
private final class CountingRenderer: EyeRendering {
    var rendered: [EyeCamera] = []

    func render(camera: EyeCamera, pose: CameraPose, geoms: [SceneGeom]) throws -> EyeImage {
        rendered.append(camera)
        let n = camera.pixelSize
        let value = UInt8(rendered.count % 256)
        return try EyeImage(camera: camera, width: n, height: n, rgb: [UInt8](repeating: value, count: n * n * 3))
    }
}
