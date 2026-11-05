// Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
// Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
//
// AudioReaderTokenizerTests.swift: the audio reading layout across the two modules. UEMASenses'
// CribAudio renders the scripted scene of Tests/UEMASensesTests/Fixtures/audio/audio_render.json
// (the Python CribAudio's output); its linear magnitudes and 6 metadata numbers, packed as
// AudioReader packs them, go through AudioTokenizer; the encoder input must equal the Python
// render's 0..1 spectrum within 1e-6 and the spatial pair [ild_db / 20, itd_ms] within 1e-6.

import Foundation
import UEMACore
import UEMASenses
import UEMATokenizer
import XCTest

final class AudioReaderTokenizerTests: XCTestCase {
    func testReaderMagnitudesTokenizeToThePythonSpectrum() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .appendingPathComponent("UEMASensesTests/Fixtures/audio/audio_render.json")
        let record = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let audio = CribAudio(gaussian: RecordedGaussian(try XCTUnwrap(record["gaussian_draws"] as? [Double])))
        let tokenizer = AudioTokenizer()
        var largest = 0.0
        for step in try XCTUnwrap(record["steps"] as? [[String: Any]]) {
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
            let rendered = audio.render(head: head, right: [m[0], m[3], m[6]], forward: [-m[2], -m[5], -m[8]],
                                        buzz: step["buzz"] as? [Double], flutter: step["flutter"] as? [Double], dt: 0.01)
            let reading = SenseReading(
                sense: .audio,
                primary: try ShapedArray(shape: [256, 2], values: rendered.magnitudes.map { Float($0) }),
                auxiliary: try ShapedArray(shape: [AudioReader.metadataCount], values: rendered.metadata.map { Float($0) }))
            let input = try tokenizer.tokenize(reading)
            let expected = try XCTUnwrap(step["spectrum"] as? [Double])
            for (found, wanted) in zip(input.primary.values, expected) {
                largest = max(largest, abs(Double(found) - wanted))
            }
            let meta = try XCTUnwrap(step["metadata"] as? [Double])
            let pair = try XCTUnwrap(input.auxiliary?.values)
            XCTAssertEqual(Double(pair[0]), meta[2] / 20, accuracy: 1e-6)
            XCTAssertEqual(Double(pair[1]), meta[3], accuracy: 1e-6)
        }
        print("audio reader to tokenizer: largest spectrum difference \(largest)")
        XCTAssertLessThanOrEqual(largest, 1e-6)
    }
}
