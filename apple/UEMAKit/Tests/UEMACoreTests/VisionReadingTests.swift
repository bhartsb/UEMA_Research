// Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
// Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
//
// VisionReadingTests.swift: UEMACore's VisionReadingLayout (Sources/UEMACore/VisionReading.swift),
// the vision SenseReading layout shared by UEMASenses' VisionReader and UEMATokenizer's
// VisionPatcher.

import Foundation
import UEMACore
import XCTest

final class VisionReadingTests: XCTestCase {
    func image(_ camera: EyeCamera, value: UInt8, size: Int? = nil) throws -> EyeImage {
        let n = size ?? camera.pixelSize
        return try EyeImage(camera: camera, width: n, height: n, rgb: [UInt8](repeating: value, count: n * n * 3))
    }

    func testReleasedLayoutHoldsTheFourPatchedCameras() {
        XCTAssertEqual(VisionReadingLayout.released.cameras, [.eyeLeft, .eyeRight, .foveaLeft, .foveaRight])
        XCTAssertEqual(VisionReadingLayout.released.valueCount, 2 * 256 * 256 * 3 + 2 * 288 * 288 * 3)
        XCTAssertEqual(VisionReadingLayout.released.valueCount, 890_880)
        XCTAssertEqual(VisionReadingLayout.gazeShape, [2, 2])
    }

    func testReadingRoundTripsImagesAndGaze() throws {
        var images: [EyeCamera: EyeImage] = [:]
        for (k, camera) in VisionReadingLayout.released.cameras.enumerated() {
            images[camera] = try image(camera, value: UInt8(10 * (k + 1)))
        }
        let left = EyeGaze(yaw: 0.25, pitch: -0.125)
        let right = EyeGaze(yaw: -0.5, pitch: 0.0625)
        let reading = try VisionReadingLayout.released.reading(images: images, left: left, right: right, isFresh: false)
        XCTAssertEqual(reading.sense, .vision)
        XCTAssertFalse(reading.isFresh)
        XCTAssertEqual(reading.primary.shape, [890_880])
        XCTAssertEqual(reading.primary.values[0], 10)
        XCTAssertEqual(reading.primary.values[2 * 256 * 256 * 3], 30)
        XCTAssertEqual(reading.auxiliary?.values, [0.25, -0.125, -0.5, 0.0625])
        let back = try VisionReadingLayout.released.images(of: reading)
        XCTAssertEqual(back.images, images)
        XCTAssertEqual(back.left, left)
        XCTAssertEqual(back.right, right)
    }

    func testMissingCameraWrongSizeAndWrongShapeThrow() throws {
        let layout = VisionReadingLayout.released
        XCTAssertThrowsError(try layout.reading(images: [:], left: .level, right: .level)) { error in
            XCTAssertEqual(error as? UEMACoreError, .missingCamera("eye_left"))
        }
        var images: [EyeCamera: EyeImage] = [:]
        for camera in layout.cameras { images[camera] = try image(camera, value: 0) }
        images[.foveaLeft] = try image(.foveaLeft, value: 0, size: 256)
        XCTAssertThrowsError(try layout.reading(images: images, left: .level, right: .level)) { error in
            XCTAssertEqual(error as? UEMACoreError, .wrongShape(type: "EyeImage fovea_left", expected: [288, 288], found: [256, 256]))
        }
        let wrong = SenseReading(sense: .vision, primary: try ShapedArray(shape: [3], values: [0, 0, 0]),
                                 auxiliary: try ShapedArray(shape: [2, 2], values: [0, 0, 0, 0]))
        XCTAssertThrowsError(try layout.images(of: wrong))
    }

    func testSmallLayoutAndNonFiniteBytes() throws {
        let layout = VisionReadingLayout(entries: [.init(.eyeLeft, pixelSize: 1), .init(.eyeRight, pixelSize: 1)])
        XCTAssertEqual(layout.valueCount, 6)
        let reading = SenseReading(sense: .vision, primary: try ShapedArray(shape: [6], values: [.nan, 300, -4, 1.6, 2, 3]),
                                   auxiliary: try ShapedArray(shape: [2, 2], values: [0, 0, 0, 0]))
        let back = try layout.images(of: reading)
        XCTAssertEqual(back.images[.eyeLeft]?.rgb, [0, 255, 0])
        XCTAssertEqual(back.images[.eyeRight]?.rgb, [2, 2, 3])
    }
}
