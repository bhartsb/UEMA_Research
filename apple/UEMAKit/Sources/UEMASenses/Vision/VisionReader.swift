// Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
// Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
//
// VisionReader.swift: the vision sense's SenseReader. It renders the four patched eye cameras
// through an EyeRendering (SceneKitEyeRenderer on Apple platforms, a fake in tests), samples the
// eye joints' yaw and pitch, and packs both into UEMACore's VisionReadingLayout, the layout
// UEMATokenizer's VisionPatcher reads. Written November 4, 2026 for the cross-module consistency
// pass (item 1 of docs/apple_port/HANDOFF_2026-11-04.md): before it, UEMASenses rendered images
// (EyeRenderPass) but no SenseReader produced a vision reading.
//
// INPUT. A renderer, the scene it reads geoms and camera poses from (MuJoCoPhysics conforms to
// SceneGeomReading), the qpos addresses of the four eye joints, a render cadence and a layout.
//
// TIMING. The session loop reads model step N before it runs N's 5 env steps (RunViewModel
// runModelStep: "read, tokenize, model ... then 5 x (decode, write, 2 physics steps)"). The
// Python trainer renders on the last env step of a burst (train_mimo.py:3243-3245
// `render=(rep == reps - 1)`; env.py:3148-3149), and the next model step reads that image. So
// this reader captures images and gaze in observeEnvStep, after env step `cadence.envStepIndex`
// of a burst whose model step the cadence renders, and read(at:) returns the capture fresh. With
// no new capture, read returns the last one with isFresh false (the trainer's "stale"); before
// any capture (the reset state) read renders the current state itself.
//
// GAZE (env.eye_yaw_pitch, Source/Python/mimo_bridge/env.py:2212-2219): left (qpos of
// robot:left_eye_horizontal, qpos of robot:left_eye_vertical), right (minus qpos of
// robot:right_eye_horizontal, qpos of robot:right_eye_vertical); a joint the scene lacks reads 0.
// On the released scene these joints are ids 70, 71, 73, 74 at qpos addresses 76, 77, 79, 80
// (Python mujoco 3.3.0, November 4, 2026).

import Foundation
import UEMACore
import UEMAPhysics

/// The qpos addresses of the four eye joints; nil for a joint the scene lacks.
public struct EyeJointAddresses: Equatable, Hashable, Sendable {
    /// The joint names, in the sequence left horizontal, left vertical, right horizontal,
    /// right vertical (env.py:1315-1316).
    public static let jointNames = [
        "robot:left_eye_horizontal", "robot:left_eye_vertical",
        "robot:right_eye_horizontal", "robot:right_eye_vertical",
    ]

    /// qpos address of robot:left_eye_horizontal.
    public let leftHorizontal: Int?
    /// qpos address of robot:left_eye_vertical.
    public let leftVertical: Int?
    /// qpos address of robot:right_eye_horizontal.
    public let rightHorizontal: Int?
    /// qpos address of robot:right_eye_vertical.
    public let rightVertical: Int?

    /// Addresses given directly.
    public init(leftHorizontal: Int?, leftVertical: Int?, rightHorizontal: Int?, rightVertical: Int?) {
        self.leftHorizontal = leftHorizontal
        self.leftVertical = leftVertical
        self.rightHorizontal = rightHorizontal
        self.rightVertical = rightVertical
    }

    /// The addresses looked up by name in the model tables (jnt_qposadr of each joint).
    public init(tables: PhysicsModelTables) {
        func address(_ name: String) -> Int? {
            tables.jointNames.firstIndex(of: name).map { tables.jointQposAddress[$0] }
        }
        self.init(leftHorizontal: address(Self.jointNames[0]), leftVertical: address(Self.jointNames[1]),
                  rightHorizontal: address(Self.jointNames[2]), rightVertical: address(Self.jointNames[3]))
    }

    /// The two gazes of `qpos` (env.eye_yaw_pitch); an address outside qpos throws
    /// `UEMACoreError.indexOutOfRange`.
    public func gaze(qpos: [Double]) throws -> (left: EyeGaze, right: EyeGaze) {
        func value(_ address: Int?) throws -> Double {
            guard let address else { return 0 }
            guard address >= 0, address < qpos.count else {
                throw UEMACoreError.indexOutOfRange(type: "eye joint qpos address", index: address, count: qpos.count)
            }
            return qpos[address]
        }
        return (EyeGaze(yaw: try value(leftHorizontal), pitch: try value(leftVertical)),
                EyeGaze(yaw: -(try value(rightHorizontal)), pitch: try value(rightVertical)))
    }
}

/// The vision SenseReader of this file's header.
public final class VisionReader: SenseReader {
    /// The sense produced: vision.
    public let sense = SenseName.vision

    /// The render pass over the layout's cameras.
    public let pass: EyeRenderPass
    /// The scene the pass reads.
    public let scene: any SceneGeomReading
    /// The eye joints the gaze is read from.
    public let eyeJoints: EyeJointAddresses
    /// Which model steps render, and after which env step.
    public let cadence: EyeRenderCadence
    /// The reading layout (`VisionReadingLayout.released` by default).
    public let layout: VisionReadingLayout

    /// The model step whose env steps are running (the one last read), and the env steps of it
    /// observed so far.
    private var burstStep: ModelStep?
    private var envStepsObserved = 0
    /// A capture not read yet, and the last one read.
    private var pending: SenseReading?
    private var last: SenseReading?

    /// A reader over `renderer` and `scene`.
    public init(renderer: any EyeRendering, scene: any SceneGeomReading, eyeJoints: EyeJointAddresses,
                cadence: EyeRenderCadence = .released, layout: VisionReadingLayout = .released) {
        self.pass = EyeRenderPass(renderer: renderer, cameras: layout.cameras)
        self.scene = scene
        self.eyeJoints = eyeJoints
        self.cadence = cadence
        self.layout = layout
    }

    /// The current images and gaze, packed.
    private func capture(_ physics: any PhysicsReading) throws -> SenseReading {
        let images = try pass.render(from: scene)
        let gaze = try eyeJoints.gaze(qpos: physics.qpos)
        return try layout.reading(images: images, left: gaze.left, right: gaze.right)
    }

    /// Captures after env step `cadence.envStepIndex` of a burst the cadence renders.
    public func observeEnvStep(_ physics: any PhysicsReading) throws {
        let index = envStepsObserved
        envStepsObserved += 1
        guard let step = burstStep, cadence.renders(atModelStep: step, afterEnvStep: index) else { return }
        pending = try capture(physics)
    }

    /// The newest capture, fresh once; otherwise the last one with isFresh false; before any
    /// capture, a render of the current state.
    public func read(_ physics: any PhysicsReading, at step: ModelStep) throws -> SenseReading {
        let reading: SenseReading
        if let new = pending {
            reading = new
            pending = nil
        } else if let old = last {
            reading = SenseReading(sense: .vision, primary: old.primary, auxiliary: old.auxiliary, isFresh: false)
        } else {
            reading = try capture(physics)
        }
        last = reading
        burstStep = step
        envStepsObserved = 0
        return reading
    }

    /// Forgets every capture and the burst position.
    public func reset() {
        burstStep = nil
        envStepsObserved = 0
        pending = nil
        last = nil
    }
}
