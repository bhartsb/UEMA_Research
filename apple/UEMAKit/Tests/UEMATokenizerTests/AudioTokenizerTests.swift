// AudioTokenizerTests.swift: hand-computed values of AudioTokenizer (work item WI-30), computed
// with numpy on November 4, 2026: clip((10 log10(m + 1e-9) + 80) / 80, 0, 1) as float32
// (audio_adapter.py:276-277); spatial = [ild_db / 20, itd_ms / 1] (audio.py:72-73).

import XCTest
import UEMACore
import UEMATokenizer

final class AudioTokenizerTests: XCTestCase {
    func testSpectrumAndSpatialPair() throws {
        var spectrum = [Float](repeating: 0, count: 256 * 2)
        spectrum[0] = 1e-4      // bin 0 left: -40.0 dB
        spectrum[1] = 0         // bin 0 right: below the -80 dB floor
        spectrum[2] = 1         // bin 1 left: 0 dB
        spectrum[3] = 0.003     // bin 1 right: -25.2 dB
        let reading = SenseReading(
            sense: .audio,
            primary: try ShapedArray(shape: [256, 2], values: spectrum),
            auxiliary: try ShapedArray(shape: [6], values: [0.1, 0.1, 6.5, -0.17, 0.5, 0.9]))
        let input = try AudioTokenizer().tokenize(reading)
        XCTAssertEqual(input.primary.shape, [256, 2])
        XCTAssertEqual(input.primary.values[0], 0.50000054, accuracy: 1e-7)
        XCTAssertEqual(input.primary.values[1], 0)
        XCTAssertEqual(input.primary.values[2], 1)
        XCTAssertEqual(input.primary.values[3], 0.68464017, accuracy: 1e-7)
        XCTAssertEqual(input.primary.values[4], 0)
        let pair = try XCTUnwrap(input.auxiliary)
        XCTAssertEqual(pair.shape, [2])
        XCTAssertEqual(pair.values[0], 0.325, accuracy: 1e-7)
        XCTAssertEqual(pair.values[1], -0.17, accuracy: 1e-7)
    }

    func testNoMetadataGivesZeroSpatialPair() throws {
        let reading = SenseReading(sense: .audio, primary: ShapedArray(zerosOfShape: [256, 2]))
        let input = try AudioTokenizer().tokenize(reading)
        XCTAssertEqual(input.auxiliary?.values, [0, 0])
        XCTAssertThrowsError(try AudioTokenizer().tokenize(
            SenseReading(sense: .audio, primary: ShapedArray(zerosOfShape: [128, 2]))))
    }

    func testFullMetadataDividesAzimuthDegreesBy90() {
        let meta = AudioTokenizer.fullMetadata(
            leftRMS: 0.2, interauralLevelDecibels: -4, interauralTimeMilliseconds: 0.1,
            azimuthRadians: Float(Double.pi / 4), frontBack: 0.7)
        XCTAssertEqual(meta.count, 5)
        XCTAssertEqual(meta[0], -0.2, accuracy: 1e-7)
        XCTAssertEqual(meta[2], 0.5, accuracy: 1e-6)
        XCTAssertEqual(meta[3], 0.7)
        XCTAssertEqual(meta[4], 0.2)
    }

    /// One auxiliary layout: the reader's 6 raw metadata numbers. The [5] and [2] layouts this
    /// tokenizer took before November 4, 2026 are refused, as is any other width.
    func testOnlyTheSixNumberMetadataIsAccepted() throws {
        for width in [2, 3, 5, 7] {
            let reading = SenseReading(
                sense: .audio,
                primary: ShapedArray(zerosOfShape: [256, 2]),
                auxiliary: ShapedArray(zerosOfShape: [width]))
            XCTAssertThrowsError(try AudioTokenizer().tokenize(reading), "width \(width)")
        }
    }
}
