// Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
// Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
//
// AudioReaderTests.swift: CribAudio and AudioReader (Sources/UEMASenses/AudioReader.swift).
//
// The fixture Fixtures/audio/audio_render.json is the Python CribAudio
// (Source/Python/mimo_bridge/audio_adapter.py) over a scripted 14-env-step scene: impacts (one
// below the 0.05 gate, two refused by the 3-in-0.2 s cap), a chime, a squeak, the bee for 6 env
// steps with numpy default_rng(7)'s recorded draws, the dragonfly with the synthesis shape, and a
// listener that moves and turns. export_audio_fixture.py beside it writes it. Lines: spectrum
// within 1e-5 on the 0..1 scale, metadata within 1e-9 (relative to max(1, |value|)), bin centres
// within 1e-9 Hz.

import Foundation
import UEMACore
import UEMAPhysics
import UEMASenses
import XCTest

final class AudioReaderTests: XCTestCase {
    static let packageRoot = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()  // UEMASensesTests
        .deletingLastPathComponent()  // Tests
        .deletingLastPathComponent()  // UEMAKit

    static var fixtureURL: URL {
        packageRoot.appendingPathComponent("Tests/UEMASensesTests/Fixtures/audio/audio_render.json")
    }

    static var sceneURL: URL {
        packageRoot.appendingPathComponent("Sources/UEMAScene/Resources/crib_age24_trunk_v1/scene.xml")
    }

    func testRenderMatchesThePythonCribAudio() throws {
        let record = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(contentsOf: Self.fixtureURL)) as? [String: Any])
        let draws = try XCTUnwrap(record["gaussian_draws"] as? [Double])
        let audio = CribAudio(gaussian: RecordedGaussian(draws))
        let frequencies = try XCTUnwrap(record["frequencies_hz"] as? [Double])
        XCTAssertEqual(frequencies.count, CribAudio.binCount)
        for (found, expected) in zip(audio.frequencies, frequencies) {
            XCTAssertEqual(found, expected, accuracy: 1e-9)
        }
        let steps = try XCTUnwrap(record["steps"] as? [[String: Any]])
        XCTAssertEqual(steps.count, 14)
        var largestSpectrum = 0.0
        var largestMetadata = 0.0
        for (k, step) in steps.enumerated() {
            var frame = AudioSourceFrame()
            for event in try XCTUnwrap(step["events"] as? [[String: Any]]) {
                let position = try XCTUnwrap(event["position"] as? [Double])
                switch event["kind"] as? String {
                case "impact": frame.events.append(.impact(position: position, impulse: try XCTUnwrap(event["impulse"] as? Double)))
                case "chime": frame.events.append(.chime(position: position))
                default: frame.events.append(.squeak(position: position))
                }
            }
            audio.apply(frame)
            let head = try XCTUnwrap(step["head"] as? [Double])
            let m = try XCTUnwrap(step["xmat_row_major"] as? [Double])
            let (_, spectrum, metadata) = audio.render(
                head: head, right: [m[0], m[3], m[6]], forward: [-m[2], -m[5], -m[8]],
                buzz: step["buzz"] as? [Double], flutter: step["flutter"] as? [Double], dt: 0.01)
            let expectedSpectrum = try XCTUnwrap(step["spectrum"] as? [Double])
            XCTAssertEqual(spectrum.count, expectedSpectrum.count)
            for (found, expected) in zip(spectrum, expectedSpectrum) {
                largestSpectrum = max(largestSpectrum, abs(Double(found) - expected))
            }
            let expectedMetadata = try XCTUnwrap(step["metadata"] as? [Double])
            for (i, (found, expected)) in zip(metadata, expectedMetadata).enumerated() {
                let difference = abs(found - expected) / max(1, abs(expected))
                largestMetadata = max(largestMetadata, difference)
                XCTAssertEqual(found, expected, accuracy: 1e-9 * max(1, abs(expected)), "env step \(k) metadata \(i)")
            }
        }
        print("audio fixture: largest spectrum difference \(largestSpectrum), largest metadata difference \(largestMetadata)")
        XCTAssertLessThanOrEqual(largestSpectrum, 1e-5)
    }

    func testSilentSceneReadsTheFloorAndZeroMetadata() throws {
        let physics = MuJoCoPhysics()
        try physics.load(sceneXML: Self.sceneURL)
        let reader = AudioReader(scene: physics)
        let first = try reader.read(physics, at: .start)
        XCTAssertEqual(first.sense, .audio)
        XCTAssertEqual(first.primary.shape, [256, 2])
        XCTAssertEqual(first.auxiliary?.shape, [AudioReader.metadataCount])
        XCTAssertTrue(first.primary.values.allSatisfy { $0 == 0 })
        XCTAssertEqual(first.auxiliary?.values, [0, 0, 0, 0, 0, 0])
    }

    func testReaderRendersEachEnvStepAndReturnsTheLast() throws {
        let physics = MuJoCoPhysics()
        try physics.load(sceneXML: Self.sceneURL)
        let sources = ScriptedSources()
        let reader = AudioReader(scene: physics, sources: sources)
        _ = try reader.read(physics, at: .start)
        XCTAssertEqual(sources.calls, 1)
        for _ in 0 ..< ModelStep.envStepsPerModelStep {
            try physics.stepEnvStep(controls: .zero, afterEachPhysicsStep: { _ in })
            try reader.observeEnvStep(physics)
        }
        XCTAssertEqual(sources.calls, 6)
        XCTAssertEqual(reader.audio.time, 0.06, accuracy: 1e-12)
        let reading = try reader.read(physics, at: ModelStep(count: 1))
        XCTAssertEqual(sources.calls, 6, "read returns the last env step's render")
        XCTAssertGreaterThan(reading.primary.values.max() ?? 0, 0, "the bee is audible")
        reader.reset()
        XCTAssertEqual(reader.audio.time, 0)
    }
}

/// The bee 30 cm in front of the origin on every env step.
private final class ScriptedSources: AudioSourceProviding {
    var calls = 0

    func sources(after physics: any PhysicsReading) throws -> AudioSourceFrame {
        calls += 1
        return AudioSourceFrame(buzzPosition: [0.3, 0, 0.3])
    }
}
