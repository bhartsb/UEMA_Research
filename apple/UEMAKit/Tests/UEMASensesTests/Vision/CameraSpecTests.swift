// CameraSpecTests.swift: tests of Sources/UEMASenses/Vision/CameraSpec.swift and of the
// downsampling arithmetic in SceneKitEyeRenderer.swift (work item WI-29). They use no GPU and
// draw nothing. Nothing here was compiled on the machine that wrote it (no Swift toolchain).
//
// FIRST BUILD ON A MAC: `cd apple/UEMAKit && xcodebuild -scheme UEMAKit-Package test
// -destination 'platform=macOS' -only-testing:UEMASensesTests/CameraSpecTests` : exit 0, 13 tests.

import Foundation
import XCTest
import UEMACore
import UEMASenses

final class CameraSpecTests: XCTestCase {
    /// The five cameras, their sizes and angles (args.json via cameras.json "world"; body XML
    /// fovy 60), in the fixture's order.
    func testReleasedCameras() {
        XCTAssertEqual(CameraSpec.released.map(\.name),
                       ["eye_left", "eye_right", "fovea_left", "fovea_right", "ring_left"])
        XCTAssertEqual(CameraSpec.released.map(\.sizePx), [256, 256, 288, 288, 256])
        XCTAssertEqual(CameraSpec.released.map(\.fovyDeg), [60, 60, 24, 24, 150])
        XCTAssertEqual(CameraSpec(camera: .foveaLeft).bytesPerImage, 288 * 288 * 3)
        XCTAssertEqual(CameraSpec(camera: .foveaRight).pixelsPerDegree, 12.0, accuracy: 1e-12)
        XCTAssertEqual(CameraSpec(camera: .ringLeft).fixtureFileName, "ring_left.png")
        XCTAssertEqual(CameraSpec.released(named: "eye_right")?.camera, .eyeRight)
        XCTAssertNil(CameraSpec.released(named: "eye_centre"))
    }

    /// Identical images differ by 0 in every channel.
    func testDifferenceOfIdenticalImagesIsZero() throws {
        let a = try EyeImage(camera: .eyeLeft, width: 2, height: 2, rgb: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
        let d = try a.difference(to: a)
        XCTAssertEqual(d.meanAbsolute, SIMD3(0, 0, 0))
        XCTAssertEqual(d.maxAbsolute, SIMD3(0, 0, 0))
        XCTAssertEqual(d.differingPixelShare, 0)
        XCTAssertEqual(d.pixelCount, 4)
    }

    /// Black against white differs by 255 in every channel: the far end of the scale.
    func testDifferenceOfOppositeImagesIs255() throws {
        let black = try EyeImage(camera: .eyeLeft, width: 3, height: 1, rgb: [UInt8](repeating: 0, count: 9))
        let white = try EyeImage(camera: .eyeLeft, width: 3, height: 1, rgb: [UInt8](repeating: 255, count: 9))
        let d = try black.difference(to: white)
        XCTAssertEqual(d.meanAbsolute, SIMD3(255, 255, 255))
        XCTAssertEqual(d.largestChannelMean, 255)
        XCTAssertEqual(d.differingPixelShare, 1)
    }

    /// One channel of one pixel out of two: mean 50 / 2 = 25 in red, 0 elsewhere, share 0.5.
    func testDifferencePerChannel() throws {
        let a = try EyeImage(camera: .eyeLeft, width: 2, height: 1, rgb: [100, 0, 0, 10, 20, 30])
        let b = try EyeImage(camera: .eyeLeft, width: 2, height: 1, rgb: [50, 0, 0, 10, 20, 30])
        let d = try a.difference(to: b)
        XCTAssertEqual(d.meanAbsolute.x, 25, accuracy: 1e-12)
        XCTAssertEqual(d.meanAbsolute.y, 0, accuracy: 1e-12)
        XCTAssertEqual(d.maxAbsolute, SIMD3(50, 0, 0))
        XCTAssertEqual(d.differingPixelShare, 0.5, accuracy: 1e-12)
    }

    /// Images of different sizes are not compared.
    func testDifferenceRejectsSizeMismatch() throws {
        let a = try EyeImage(camera: .eyeLeft, width: 1, height: 1, rgb: [0, 0, 0])
        let b = try EyeImage(camera: .eyeLeft, width: 2, height: 1, rgb: [0, 0, 0, 0, 0, 0])
        XCTAssertThrowsError(try a.difference(to: b)) { error in
            XCTAssertEqual(error as? EyeImageError,
                           .sizeMismatch(width: 1, height: 1, otherWidth: 2, otherHeight: 1))
        }
    }

    /// pixel(x:y:) reads rows top-down, columns left-to-right; meanIntensity is the byte mean.
    func testPixelLayoutAndMeanIntensity() throws {
        let image = try EyeImage(camera: .eyeLeft, width: 2, height: 2,
                                 rgb: [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        XCTAssertEqual(image.pixel(x: 1, y: 0).red, 1)
        XCTAssertEqual(image.pixel(x: 0, y: 1).green, 5)
        XCTAssertEqual(image.pixel(x: 1, y: 1).blue, 9)
        XCTAssertEqual(image.meanIntensity, 45.0 / 12.0, accuracy: 1e-12)
    }

    /// The released cadence renders every model step after its 5th env step (train_mimo.py:3245
    /// `render=(rep == reps - 1)`); the plan's every-2nd cadence is available by name.
    func testRenderCadence() {
        let c = EyeRenderCadence.released
        XCTAssertTrue(c.renders(atModelStep: ModelStep(count: 0)))
        XCTAssertTrue(c.renders(atModelStep: ModelStep(count: 7)))
        XCTAssertTrue(c.renders(atModelStep: ModelStep(count: 7), afterEnvStep: 4))
        XCTAssertFalse(c.renders(atModelStep: ModelStep(count: 7), afterEnvStep: 3))
        let half = EyeRenderCadence.everySecondModelStep
        XCTAssertTrue(half.renders(atModelStep: ModelStep(count: 4)))
        XCTAssertFalse(half.renders(atModelStep: ModelStep(count: 5)))
    }

    /// A cadence decodes through the same limits as init; the env step index defaults to the
    /// last of the 5 env steps when absent.
    func testRenderCadenceDecodes() throws {
        let decoder = JSONDecoder()
        let full = try decoder.decode(EyeRenderCadence.self,
                                      from: Data(#"{"everyModelSteps": 2, "envStepIndex": 1}"#.utf8))
        XCTAssertEqual(full, EyeRenderCadence(everyModelSteps: 2, envStepIndex: 1))
        let short = try decoder.decode(EyeRenderCadence.self, from: Data(#"{"everyModelSteps": 1}"#.utf8))
        XCTAssertEqual(short, EyeRenderCadence.released)
        let encoded = try JSONEncoder().encode(EyeRenderCadence.everySecondModelStep)
        XCTAssertEqual(try decoder.decode(EyeRenderCadence.self, from: encoded), .everySecondModelStep)
    }

    /// A decoded cadence of 0 model steps (a division by 0 in renders(atModelStep:)) or an env
    /// step index outside 0 ..< 5 throws instead of stopping the process.
    func testRenderCadenceDecodingRejectsOutOfRangeValues() {
        let decoder = JSONDecoder()
        XCTAssertThrowsError(try decoder.decode(EyeRenderCadence.self,
                                                from: Data(#"{"everyModelSteps": 0}"#.utf8))) { error in
            XCTAssertEqual(error as? EyeRenderCadence.DecodingProblem, .everyModelStepsBelowOne(0))
        }
        XCTAssertThrowsError(try decoder.decode(EyeRenderCadence.self,
                                                from: Data(#"{"everyModelSteps": 1, "envStepIndex": 5}"#.utf8))) { error in
            XCTAssertEqual(error as? EyeRenderCadence.DecodingProblem, .envStepIndexOutOfRange(5))
        }
    }

    /// The pass reads the geom list once, one pose per camera, and renders each camera once in
    /// EyeCamera.allCases order.
    func testRenderPassUsesSceneOnceAndEveryCamera() throws {
        let scene = EyeFakeScene()
        let renderer = EyeFakeRenderer()
        let images = try EyeRenderPass(renderer: renderer).render(from: scene)
        XCTAssertEqual(scene.geomReads, 1)
        XCTAssertEqual(scene.poseReads, EyeCamera.allCases)
        XCTAssertEqual(renderer.rendered, EyeCamera.allCases)
        XCTAssertEqual(Set(images.keys), Set(EyeCamera.allCases))
        XCTAssertEqual(images[.foveaLeft]?.width, 288)
        XCTAssertEqual(renderer.geomCounts, [Int](repeating: 1, count: 5))
    }

    /// A renderer error stops the pass and reaches the caller unchanged.
    func testRenderPassPassesErrorsOn() {
        let renderer = EyeFakeRenderer()
        renderer.failOn = .foveaRight
        XCTAssertThrowsError(try EyeRenderPass(renderer: renderer).render(from: EyeFakeScene())) { error in
            XCTAssertEqual(error as? EyeFakeError, .failed)
        }
        XCTAssertEqual(renderer.rendered, [.eyeLeft, .eyeRight, .foveaLeft])
    }

    /// 2 x 2 blocks average with rounding half up: (0 + 1 + 1 + 1) / 4 = 0.75 -> 1.
    func testDownsampleAveragesBlocks() {
        // 2 x 2 RGBA image -> 1 RGB pixel.
        let rgba: [UInt8] = [0, 10, 200, 255, 1, 10, 201, 255, 1, 11, 202, 255, 1, 11, 203, 255]
        let out = rgba.withUnsafeBufferPointer { eyeDownsample(rgba: $0, width: 2, factor: 2) }
        XCTAssertEqual(out, [1, 11, 202])
    }

    /// Factor 1 keeps every pixel and drops alpha, rows top-down.
    func testDownsampleFactorOneDropsAlpha() {
        let rgba: [UInt8] = [1, 2, 3, 99, 4, 5, 6, 99, 7, 8, 9, 99, 10, 11, 12, 99]
        let out = rgba.withUnsafeBufferPointer { eyeDownsample(rgba: $0, width: 2, factor: 1) }
        XCTAssertEqual(out, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    }
}

// MARK: - Fakes

private enum EyeFakeError: Error, Equatable {
    case failed
}

/// A scene with one sphere and the identity camera pose for every camera; it counts its reads.
private final class EyeFakeScene: SceneGeomReading {
    var geomReads = 0
    var poseReads: [EyeCamera] = []

    func sceneGeoms() throws -> [SceneGeom] {
        geomReads += 1
        return [try SceneGeom(type: 2, size: [0.1, 0, 0], position: [0, 0, -1],
                              rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], rgba: [1, 0, 0, 1])]
    }

    func cameraPose(_ camera: EyeCamera) throws -> CameraPose {
        poseReads.append(camera)
        return try CameraPose(position: [0, 0, 0], rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1])
    }
}

/// A renderer that returns a black image of the camera's size and records each call.
private final class EyeFakeRenderer: EyeRendering {
    var rendered: [EyeCamera] = []
    var geomCounts: [Int] = []
    var failOn: EyeCamera?

    func render(camera: EyeCamera, pose: CameraPose, geoms: [SceneGeom]) throws -> EyeImage {
        if camera == failOn { throw EyeFakeError.failed }
        rendered.append(camera)
        geomCounts.append(geoms.count)
        let n = camera.pixelSize
        return try EyeImage(camera: camera, width: n, height: n, rgb: [UInt8](repeating: 0, count: n * n * 3))
    }
}
