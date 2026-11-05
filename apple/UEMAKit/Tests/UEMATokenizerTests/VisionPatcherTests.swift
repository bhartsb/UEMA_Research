// VisionPatcherTests.swift: tests of VisionPatcher (work item WI-30). The synthetic expectations
// were computed with Source/Python/mimo_bridge/vision_adapter.py (patch_grid_rays, compose_gaze)
// under numpy 2.2.6 on November 4, 2026. The fixture test reads the WI-14 states of
// Tests/UEMASensesTests/Fixtures/eyes/ by path; the Python patcher reproduced their
// patches_sha256 and angles_sha256 from the same PNGs and gaze on November 4, 2026.

import CryptoKit
import Foundation
import ImageIO
import XCTest
import UEMACore
import UEMATokenizer

final class VisionPatcherTests: XCTestCase {
    /// 2 x (256 + 324) = 1,160 rows on the released layout (patches.json "expected_n_rule").
    func testReleasedLayoutHas1160Rows() {
        let patcher = VisionPatcher.released()
        XCTAssertEqual(patcher.patchesPerEye, 580)
        XCTAssertEqual(patcher.patchCount, 1160)
        XCTAssertEqual(patcher.readingValueCount, 2 * 256 * 256 * 3 + 2 * 288 * 288 * 3)
        XCTAssertEqual(patcher.spans[0], 0.06544985, accuracy: 1e-8)
        XCTAssertEqual(patcher.spans[1], 0.023271056, accuracy: 1e-8)
    }

    private func syntheticPatcher() throws -> VisionPatcher {
        let source = VisionSource(name: "test", leftCamera: .eyeLeft, rightCamera: .eyeRight,
                                  sizePixels: 32, fieldOfViewDegrees: 60)
        return try VisionPatcher(sources: [source])
    }

    /// Left byte at (y, x, channel) = ((y x 32 + x) x 3 + channel) mod 256; right = 255 minus it.
    private func syntheticImages() throws -> [EyeCamera: EyeImage] {
        var left = [UInt8](repeating: 0, count: 32 * 32 * 3)
        for index in 0 ..< left.count {
            left[index] = UInt8(index % 256)
        }
        let right = left.map { 255 - $0 }
        let leftImage = try EyeImage(camera: .eyeLeft, width: 32, height: 32, rgb: left)
        let rightImage = try EyeImage(camera: .eyeRight, width: 32, height: 32, rgb: right)
        return [.eyeLeft: leftImage, .eyeRight: rightImage]
    }

    func testSyntheticPatchBytesAreInterleavedAndRowMajor() throws {
        let patcher = try syntheticPatcher()
        let result = try patcher.patch(images: try syntheticImages(), left: .level, right: .level)
        XCTAssertEqual(result.patches.shape, [8, 768])
        XCTAssertEqual(result.angles.shape, [8, 4])
        let p = result.patches.values
        // Row 0 = left patch 0: pixel (0, 0, 0) = 0; inside the patch, (py 1, px 0, channel 2) is
        // number 1 x 48 + 2 = 50, the byte (32 x 3 + 2) = 98.
        XCTAssertEqual(p[0], 0)
        XCTAssertEqual(p[50], Float(98) / 255)
        // Row 2 = left patch 1 (grid row 0, column 1): its first byte is pixel (0, 16, 0) = 48.
        XCTAssertEqual(p[2 * 768], Float(48) / 255)
        // Row 3 = right patch 1: 255 - 48 = 207.
        XCTAssertEqual(p[3 * 768], Float(207) / 255)
        // Row 4 = left patch 2 (grid row 1, column 0): pixel (16, 0, 0) = (16 x 32 x 3) mod 256 = 0.
        XCTAssertEqual(p[4 * 768], 0)
    }

    func testSyntheticAnglesAtLevelGaze() throws {
        let patcher = try syntheticPatcher()
        let a = try patcher.patch(images: try syntheticImages(), left: .level, right: .level).angles.values
        // Row 0: left patch 0; row 1: right patch 0; row 2: left patch 1 (azimuth negative = right).
        XCTAssertEqual(a[0], 0.2810348868370056, accuracy: 1e-6)
        XCTAssertEqual(a[1], 0.2705497741699219, accuracy: 1e-6)
        XCTAssertEqual(a[2], -1)
        XCTAssertEqual(a[3], 0.5235988, accuracy: 1e-6)
        XCTAssertEqual(a[4], 0.2810348868370056, accuracy: 1e-6)
        XCTAssertEqual(a[6], 1)
        XCTAssertEqual(a[8], -0.2810348868370056, accuracy: 1e-6)
        XCTAssertEqual(a[9], 0.2705497741699219, accuracy: 1e-6)
    }

    func testSyntheticAnglesComposeTheGaze() throws {
        let patcher = try syntheticPatcher()
        let gaze = EyeGaze(yaw: 0.3, pitch: -0.2)
        let a = try patcher.patch(images: try syntheticImages(), left: gaze, right: .level).angles.values
        XCTAssertEqual(a[0], 0.5713974833488464, accuracy: 1e-6)
        XCTAssertEqual(a[1], 0.07808106392621994, accuracy: 1e-6)
        XCTAssertEqual(a[8], 0.028602542355656624, accuracy: 1e-6)
        XCTAssertEqual(a[9], 0.07808106392621994, accuracy: 1e-6)
        XCTAssertEqual(a[16], 0.6032071709632874, accuracy: 1e-6)
        XCTAssertEqual(a[17], -0.46214136481285095, accuracy: 1e-6)
        // The right eye kept the level gaze.
        XCTAssertEqual(a[4], 0.2810348868370056, accuracy: 1e-6)
    }

    func testReadingLayoutRoundTripsToThePatches() throws {
        let patcher = try syntheticPatcher()
        let images = try syntheticImages()
        let gaze = EyeGaze(yaw: 0.3, pitch: -0.2)
        let direct = try patcher.patch(images: images, left: gaze, right: .level)
        let reading = try patcher.reading(images: images, left: gaze, right: .level, isFresh: false)
        XCTAssertEqual(reading.primary.shape, [2 * 32 * 32 * 3])
        XCTAssertEqual(reading.auxiliary?.shape, [2, 2])
        let input = try patcher.tokenize(reading)
        XCTAssertEqual(input.sense, .vision)
        XCTAssertFalse(input.isFresh)
        XCTAssertEqual(input.primary, direct.patches)
        XCTAssertEqual(input.auxiliary, direct.angles)
    }

    func testMissingCameraAndWrongSizeThrow() throws {
        let patcher = try syntheticPatcher()
        var images = try syntheticImages()
        images[.eyeRight] = nil
        XCTAssertThrowsError(try patcher.patch(images: images, left: .level, right: .level)) { error in
            XCTAssertEqual(error as? TokenizerError, .missingImage(camera: "eye_right"))
        }
        images[.eyeRight] = try EyeImage(camera: .eyeRight, width: 48, height: 48,
                                         rgb: [UInt8](repeating: 0, count: 48 * 48 * 3))
        XCTAssertThrowsError(try patcher.patch(images: images, left: .level, right: .level))
        let wrong = SenseReading(sense: .audio, primary: ShapedArray(zerosOfShape: [256, 2]))
        XCTAssertThrowsError(try patcher.tokenize(wrong)) { error in
            XCTAssertEqual(error as? TokenizerError, .wrongSense(expected: .vision, found: .audio))
        }
    }

    // MARK: - The WI-14 fixture

    /// The eyes fixture folder of the UEMASensesTests target, found from this file's path.
    private static let eyesFolder = URL(fileURLWithPath: #filePath)
        .deletingLastPathComponent()
        .deletingLastPathComponent()
        .appendingPathComponent("UEMASensesTests/Fixtures/eyes", isDirectory: true)

    func testFixtureStatesGiveTheRecordedCountAnglesAndPatchDigest() throws {
        let states = ["step_000492800", "step_000493674", "step_000493980"]
        let patcher = VisionPatcher.released()
        for state in states {
            let folder = VisionPatcherTests.eyesFolder.appendingPathComponent(state, isDirectory: true)
            let jsonURL = folder.appendingPathComponent("patches.json")
            guard FileManager.default.fileExists(atPath: jsonURL.path) else {
                throw XCTSkip("fixture \(jsonURL.path) is not on disk (WI-14 writes it)")
            }
            let record = try XCTUnwrap(
                try JSONSerialization.jsonObject(with: Data(contentsOf: jsonURL)) as? [String: Any])
            let gaze = try XCTUnwrap(record["gaze"] as? [String: Any])
            let leftGaze = try XCTUnwrap(gaze["left_yaw_pitch_rad"] as? [Double])
            let rightGaze = try XCTUnwrap(gaze["right_yaw_pitch_rad"] as? [Double])
            var images: [EyeCamera: EyeImage] = [:]
            for camera in [EyeCamera.eyeLeft, .eyeRight, .foveaLeft, .foveaRight] {
                images[camera] = try VisionPatcherTests.loadPNG(
                    folder.appendingPathComponent("\(camera.mujocoName).png"), camera: camera)
            }
            let result = try patcher.patch(
                images: images,
                left: EyeGaze(yaw: leftGaze[0], pitch: leftGaze[1]),
                right: EyeGaze(yaw: rightGaze[0], pitch: rightGaze[1]))

            let expectedCount = try XCTUnwrap(record["n_total"] as? Int)
            XCTAssertEqual(result.patches.shape, [expectedCount, 768], state)
            XCTAssertEqual(result.angles.shape, [expectedCount, 4], state)

            let firstAngles = try XCTUnwrap(record["first_angle_rows"] as? [[Double]])
            var largestAngleDifference = 0.0
            for (row, expected) in firstAngles.prefix(4).enumerated() {
                for column in 0 ..< 4 {
                    let found = Double(result.angles.values[row * 4 + column])
                    largestAngleDifference = max(largestAngleDifference, abs(found - expected[column]))
                    XCTAssertEqual(found, expected[column], accuracy: 1e-5, "\(state) angle row \(row) column \(column)")
                }
            }

            let head = try XCTUnwrap(record["first_patch_row_head"] as? [Double])
            for (index, expected) in head.enumerated() {
                XCTAssertEqual(Double(result.patches.values[index]), expected, accuracy: 1e-7, "\(state) patch number \(index)")
            }

            let patchDigest = VisionPatcherTests.sha256Hex(result.patches.values)
            XCTAssertEqual(patchDigest, record["patches_sha256"] as? String, "\(state) patches digest")
            let angleDigest = VisionPatcherTests.sha256Hex(result.angles.values)
            let angleDigestMatches = angleDigest == (record["angles_sha256"] as? String)
            print("WI-30 \(state): N = \(expectedCount); first 4 angle rows largest difference "
                  + "\(largestAngleDifference) rad (line 1e-5); patches digest match "
                  + "\(patchDigest == (record["patches_sha256"] as? String)); angles digest match \(angleDigestMatches)")
        }
    }

    /// SHA-256 over the float32 bytes, row-major, little-endian (patches.json "digest_rule").
    private static func sha256Hex(_ values: [Float]) -> String {
        let data = values.withUnsafeBufferPointer { Data(buffer: $0) }
        return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    /// The PNG's decoded bytes as RGB 8-bit, read from the image's data provider without a
    /// colour-converting draw.
    private static func loadPNG(_ url: URL, camera: EyeCamera) throws -> EyeImage {
        let source = try XCTUnwrap(CGImageSourceCreateWithURL(url as CFURL, nil), "no image source for \(url.path)")
        let image = try XCTUnwrap(CGImageSourceCreateImageAtIndex(source, 0, nil), "no image in \(url.path)")
        guard image.bitsPerComponent == 8 else {
            throw FixtureDecodeError(message: "\(url.path): \(image.bitsPerComponent) bits per colour, the loader reads 8")
        }
        let data = try XCTUnwrap(image.dataProvider?.data, "no pixel data in \(url.path)")
        let length = CFDataGetLength(data)
        let bytes = try XCTUnwrap(CFDataGetBytePtr(data))
        let width = image.width
        let height = image.height
        let bytesPerPixel = image.bitsPerPixel / 8
        let bytesPerRow = image.bytesPerRow
        guard length >= bytesPerRow * height else {
            throw FixtureDecodeError(message: "\(url.path): \(length) bytes of pixel data, fewer than \(bytesPerRow) x \(height)")
        }
        var offsets = [0, 1, 2]
        if bytesPerPixel == 4 {
            let little = image.bitmapInfo.contains(.byteOrder32Little)
            switch image.alphaInfo {
            case .premultipliedFirst, .first, .noneSkipFirst:
                offsets = little ? [2, 1, 0] : [1, 2, 3]
            default:
                offsets = little ? [3, 2, 1] : [0, 1, 2]
            }
        } else if bytesPerPixel != 3 {
            // A palette or grey PNG has 1 or 2 bytes per pixel: reading 3 offsets would run past
            // the pixel. The WI-14 PNGs are "RGB 8-bit" (cameras.json image_format).
            throw FixtureDecodeError(message: "\(url.path): \(bytesPerPixel) bytes per pixel; the loader reads RGB or RGBA")
        }
        var rgb = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0 ..< height {
            for x in 0 ..< width {
                let s = y * bytesPerRow + x * bytesPerPixel
                let o = (y * width + x) * 3
                rgb[o] = bytes[s + offsets[0]]
                rgb[o + 1] = bytes[s + offsets[1]]
                rgb[o + 2] = bytes[s + offsets[2]]
            }
        }
        return try EyeImage(camera: camera, width: width, height: height, rgb: rgb)
    }
}

/// A fixture PNG the loader cannot read as RGB 8-bit; thrown so that the fixture test fails
/// with the file's path and the reason.
private struct FixtureDecodeError: Error, CustomStringConvertible {
    /// The file and what was found in it.
    let message: String

    var description: String { message }
}
