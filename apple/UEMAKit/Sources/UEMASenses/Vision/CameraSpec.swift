// CameraSpec.swift: the five eye cameras of the released world as the eye renderer draws them,
// the difference measurement between two eye images, the render cadence, and the pass that
// renders the five cameras of one state (work item WI-29 of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-6). Nothing in
// this file was compiled on the machine that wrote it (no Swift toolchain; plan section 3, fact 1).
//
// WHAT THIS FILE IS. Plain value types and one pass type without SceneKit or Metal, so that
// they compile and test on any platform: CameraSpec gives one UEMACore.EyeCamera's square image
// size in pixels and vertical field of view in degrees; the extension on UEMACore.EyeImage
// measures the per-channel mean absolute difference against another image (WI-29's acceptance
// measurement: "the mean absolute difference per channel against the Python PNG (0 =
// identical, 255 = every pixel opposite) is recorded"); EyeRenderCadence says on which model
// steps the eyes are rendered; EyeRenderPass renders the five cameras of one state through any
// EyeRendering from any SceneGeomReading (UEMACore/Protocols.swift).
//
// INPUT. The released run's settings (weights/b1_continue_B1K_skip13_492800/run/args.json,
// copied by WI-14 into every Tests/UEMASensesTests/Fixtures/eyes/<state>/cameras.json under
// "world"): eye_size_px 256, fovea_size_px 288, fovea_fov_deg 24.0, ring_fov_deg 150.0,
// ring_size_px 256 (BridgeEnv renders the ring through the wide eye renderer at eye_size:
// Source/Python/mimo_bridge/env.py:1494-1502; args.json's ring_size 128 is not used for the
// image size). The wide cameras' 60 degrees come from the body XML
// (Sources/UEMAScene/Resources/crib_age24_trunk_v1/MIMo_modelv2_grown24_fovea.xml:337 and 345,
// `<camera name="eye_left" mode="fixed" fovy="60" pos="0.01125 0 0" euler="90 -90 0" />`).
// UEMACore.EyeCamera (Protocols.swift, "The eyes") holds the same five sizes and angles; this
// file reads them from there and adds nothing that could disagree.
//
// THE RENDER CADENCE. The plan's row WI-29 says "Every 2nd model step"; the Python the released
// run trained with renders on EVERY model step. Source/Python/mimo_bridge/train_mimo.py:3243-3245
// passes `render=(rep == reps - 1)` to each of the model step's env steps (reps = 5 env steps per
// model step), and env.py:3148-3149 then renders when `want_render` is true: `do_render =
// (force_render or self.frame % self.render_every == 0) if want_render is None else
// bool(want_render or force_render)`. render_every 2 (args.json) applies only when want_render is
// None, which the trainer path never passes. env.py:3140-3146 states the reason: "every render is
// used and every model step gets a fresh image". The Python is the truth for this port, so
// EyeRenderCadence.released renders on every model step, after the last of its 5 env steps.
//
// OPERATION. Lookups; the difference: for two images of equal size, the mean over all pixels of
// |a - b| per channel in 8-bit steps, the largest single difference per channel, and the share
// of pixels that differ in any channel (the same three numbers as the fixture exporter's mad(),
// Tests/UEMASensesTests/Fixtures/eyes/export_abstract_scene.py); the pass: one sceneGeoms() read
// per state, then one cameraPose and one render per camera in EyeCamera.allCases order.
//
// OUTPUT. CameraSpec.released, the five cameras in the fixture's order (eye_left, eye_right,
// fovea_left, fovea_right, ring_left: apple/scripts/export_eye_fixtures.py CAMERA_FILES);
// EyeImageDifference; EyeRenderCadence.released; EyeRenderPass.render(from:) -> one EyeImage
// per camera, rgb bytes row-major, rows top-down, columns left-to-right, 3 bytes per pixel (the
// fixture PNGs' layout: cameras.json "image_format": "PNG, RGB 8-bit, rows top-down, columns
// left-to-right").
//
// FIRST BUILD ON A MAC (this file's part of WI-29's acceptance).
//  1. Run WI-17's steps 1 to 3 (MuJoCo XCFramework in place, `swift package resolve` exit 0).
//  2. `cd apple/UEMAKit && xcodebuild -scheme UEMAKit-Package test -destination
//     'platform=macOS' -only-testing:UEMASensesTests/CameraSpecTests` : exit 0. These tests use
//     no GPU and no SceneKit.
//  3. A compile error naming EyeImage, EyeCamera, CameraPose, SceneGeom or SceneGeomReading
//     means UEMACore's Protocols.swift changed: UEMACore wins, and this file follows it.
//  4. Record the command, exit code and clock time (AM/PM Pacific) in apple/FIRST_BUILD_RECORD.md.
//
// SOURCES READ (November 4, 2026): the files named above, read entire where cited by line.

import Foundation
import UEMACore

/// One eye camera as the eye renderer draws it: the camera, its square image size in pixels and
/// its vertical field of view in degrees (MuJoCo's `cam_fovy`). The image is square, so the
/// horizontal field of view equals the vertical one. The numbers come from
/// `UEMACore.EyeCamera`.
public struct CameraSpec: Hashable, Sendable {
    /// The camera.
    public let camera: EyeCamera
    /// Width and height of the rendered image in pixels.
    public let sizePx: Int
    /// Vertical field of view in degrees.
    public let fovyDeg: Double

    /// The released description of `camera`.
    public init(camera: EyeCamera) {
        self.camera = camera
        self.sizePx = camera.pixelSize
        self.fovyDeg = camera.fieldOfViewDegrees
    }

    /// The camera's name in the scene XML (`mj_name2id(model, mjOBJ_CAMERA, name)` finds it).
    public var name: String { camera.mujocoName }

    /// The five cameras in the fixture's order (export_eye_fixtures.py CAMERA_FILES; equal to
    /// `EyeCamera.allCases`).
    public static let released: [CameraSpec] = EyeCamera.allCases.map { CameraSpec(camera: $0) }

    /// The released camera with this scene XML name, or nil for any other name.
    public static func released(named name: String) -> CameraSpec? {
        EyeCamera(rawValue: name).map { CameraSpec(camera: $0) }
    }

    /// Pixels per degree of field of view (256 / 60 = 4.27 for the wide eyes, 288 / 24 = 12.0
    /// for the foveas, 256 / 150 = 1.71 for the ring).
    public var pixelsPerDegree: Double { Double(sizePx) / fovyDeg }

    /// Bytes of one RGB image: sizePx x sizePx x 3.
    public var bytesPerImage: Int { sizePx * sizePx * 3 }

    /// The fixture file name of this camera's image (export_eye_fixtures.py CAMERA_FILES:
    /// "<name>.png").
    public var fixtureFileName: String { "\(name).png" }
}

/// Why two eye images could not be compared.
public enum EyeImageError: Error, Equatable, CustomStringConvertible {
    /// Two images of different sizes were compared.
    case sizeMismatch(width: Int, height: Int, otherWidth: Int, otherHeight: Int)

    public var description: String {
        switch self {
        case let .sizeMismatch(width, height, otherWidth, otherHeight):
            return "eye image: \(width)x\(height) compared with \(otherWidth)x\(otherHeight)"
        }
    }
}

/// The per-channel difference between two RGB images of equal size.
public struct EyeImageDifference: Equatable, Sendable {
    /// Mean of |a - b| over every pixel, per channel (red, green, blue), in 8-bit steps:
    /// 0 = identical, 255 = every pixel at the opposite end.
    public let meanAbsolute: SIMD3<Double>
    /// The largest |a - b| per channel, 0 to 255.
    public let maxAbsolute: SIMD3<Int>
    /// Share of pixels that differ in at least one channel, 0 (none) to 1 (every pixel).
    public let differingPixelShare: Double
    /// Pixels compared (width x height).
    public let pixelCount: Int

    /// The largest of the three channel means, the single number WI-29's pass line is set from.
    public var largestChannelMean: Double { max(meanAbsolute.x, max(meanAbsolute.y, meanAbsolute.z)) }
}

extension EyeImage {
    /// The three bytes of the pixel at column `x` (0 = left) and row `y` (0 = top).
    public func pixel(x: Int, y: Int) -> (red: UInt8, green: UInt8, blue: UInt8) {
        precondition(x >= 0 && x < width && y >= 0 && y < height, "pixel (\(x), \(y)) outside \(width)x\(height)")
        let i = (y * width + x) * 3
        return (rgb[i], rgb[i + 1], rgb[i + 2])
    }

    /// Mean of all bytes over the whole image, 0 to 255 (a one-number brightness for logs; the
    /// fixture test prints it beside the fixture's).
    public var meanIntensity: Double {
        guard !rgb.isEmpty else { return 0 }
        var sum = 0
        for b in rgb { sum += Int(b) }
        return Double(sum) / Double(rgb.count)
    }

    /// The difference against `other` (WI-29's acceptance measurement). The cameras are not
    /// compared, only the pixels.
    /// - Throws: `EyeImageError.sizeMismatch` when the sizes differ.
    public func difference(to other: EyeImage) throws -> EyeImageDifference {
        guard width == other.width, height == other.height else {
            throw EyeImageError.sizeMismatch(width: width, height: height,
                                             otherWidth: other.width, otherHeight: other.height)
        }
        var sums = SIMD3<Int>(repeating: 0)
        var maxima = SIMD3<Int>(repeating: 0)
        var differing = 0
        let pixelCount = width * height
        var i = 0
        for _ in 0 ..< pixelCount {
            let dr = abs(Int(rgb[i]) - Int(other.rgb[i]))
            let dg = abs(Int(rgb[i + 1]) - Int(other.rgb[i + 1]))
            let db = abs(Int(rgb[i + 2]) - Int(other.rgb[i + 2]))
            sums &+= SIMD3(dr, dg, db)
            maxima = SIMD3(max(maxima.x, dr), max(maxima.y, dg), max(maxima.z, db))
            if dr != 0 || dg != 0 || db != 0 { differing += 1 }
            i += 3
        }
        let n = Double(max(pixelCount, 1))
        return EyeImageDifference(
            meanAbsolute: SIMD3(Double(sums.x) / n, Double(sums.y) / n, Double(sums.z) / n),
            maxAbsolute: maxima,
            differingPixelShare: Double(differing) / n,
            pixelCount: pixelCount)
    }
}

/// On which model steps the eyes are rendered: every `everyModelSteps`-th one, after the env
/// step `envStepIndex` (0 to 4) of that model step. The released trainer renders on every model
/// step after its last env step (train_mimo.py:3243-3245 `render=(rep == reps - 1)`; this file's
/// header gives the full reading and the plan row's different "every 2nd").
public struct EyeRenderCadence: Hashable, Codable, Sendable {
    /// Render on model steps whose count is a multiple of this number.
    public let everyModelSteps: Int
    /// The env step within the model step after which the render happens, 0 to
    /// `ModelStep.envStepsPerModelStep - 1`.
    public let envStepIndex: Int

    /// Creates a cadence; `everyModelSteps` is at least 1 and `envStepIndex` lies in 0 ..< 5.
    public init(everyModelSteps: Int, envStepIndex: Int = ModelStep.envStepsPerModelStep - 1) {
        precondition(everyModelSteps >= 1, "render cadence needs at least 1 model step")
        precondition(envStepIndex >= 0 && envStepIndex < ModelStep.envStepsPerModelStep,
                     "envStepIndex \(envStepIndex) outside 0 ..< \(ModelStep.envStepsPerModelStep)")
        self.everyModelSteps = everyModelSteps
        self.envStepIndex = envStepIndex
    }

    /// The released cadence: every model step, after its last (5th) env step.
    public static let released = EyeRenderCadence(everyModelSteps: 1)

    /// The cadence the plan row and env.py's fallback name (render_every 2), for a session
    /// configured for half the renders; not what the released trainer did.
    public static let everySecondModelStep = EyeRenderCadence(everyModelSteps: 2)

    private enum CodingKeys: String, CodingKey {
        case everyModelSteps
        case envStepIndex
    }

    /// Why a decoded cadence was refused.
    public enum DecodingProblem: Error, Equatable, CustomStringConvertible {
        /// `everyModelSteps` below 1 (a cadence of 0 would divide by 0 in `renders`).
        case everyModelStepsBelowOne(Int)
        /// `envStepIndex` outside 0 ..< `ModelStep.envStepsPerModelStep` (0 ..< 5).
        case envStepIndexOutOfRange(Int)

        public var description: String {
            switch self {
            case let .everyModelStepsBelowOne(value):
                return "render cadence: everyModelSteps \(value) is below 1"
            case let .envStepIndexOutOfRange(value):
                return "render cadence: envStepIndex \(value) outside 0 ..< \(ModelStep.envStepsPerModelStep)"
            }
        }
    }

    /// Decodes `{"everyModelSteps": n, "envStepIndex": k}` through the same limits as `init`, but
    /// throws `DecodingProblem` on a value outside them instead of stopping the process (the
    /// synthesized decoder would assign the fields without any check).
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let every = try container.decode(Int.self, forKey: .everyModelSteps)
        let index = try container.decodeIfPresent(Int.self, forKey: .envStepIndex)
            ?? ModelStep.envStepsPerModelStep - 1
        guard every >= 1 else { throw DecodingProblem.everyModelStepsBelowOne(every) }
        guard index >= 0 && index < ModelStep.envStepsPerModelStep else {
            throw DecodingProblem.envStepIndexOutOfRange(index)
        }
        self.everyModelSteps = every
        self.envStepIndex = index
    }

    /// True when the eyes are rendered on this model step (model step 0 renders).
    public func renders(atModelStep step: ModelStep) -> Bool {
        step.count % everyModelSteps == 0
    }

    /// True when the render happens after env step `envStep` (0 to 4) of `step`.
    public func renders(atModelStep step: ModelStep, afterEnvStep envStep: Int) -> Bool {
        renders(atModelStep: step) && envStep == envStepIndex
    }
}

/// Renders the cameras of one state: reads the geom list once from a `SceneGeomReading`, then
/// each camera's pose, and hands the three to an `EyeRendering` per camera. Both are injected,
/// so a test runs the pass with fakes and no graphics library.
public struct EyeRenderPass {
    /// The renderer each camera goes through.
    public let renderer: any EyeRendering
    /// The cameras rendered, in this sequence.
    public let cameras: [EyeCamera]

    /// A pass over `cameras` (default: all five, fixture order) through `renderer`.
    public init(renderer: any EyeRendering, cameras: [EyeCamera] = EyeCamera.allCases) {
        self.renderer = renderer
        self.cameras = cameras
    }

    /// One image per camera of the current state of `scene`.
    ///
    /// The geom list is read once and used for every camera. MuJoCo's mjv_updateScene moves only
    /// one thing per camera: an infinite plane (size 0, the released floor at scene.xml:37) is
    /// recentred under the camera (the exporter's note in export_abstract_scene.py, "planes of
    /// size 0 are recentred on the camera"). On the three WI-14 states the exporter found 0 geoms
    /// whose record differs between the five cameras (abstract_scene.json
    /// cameras.<name>.geom_differences, November 4, 2026): the floor's record keeps size 0 and
    /// position (0, 0, 0). GeomMirror draws such a plane at a half extent of the far clipping
    /// distance (96.9 m for the released crib) around its given centre.
    /// - Throws: whatever the scene or the renderer throws, unchanged.
    public func render(from scene: any SceneGeomReading) throws -> [EyeCamera: EyeImage] {
        let geoms = try scene.sceneGeoms()
        var images: [EyeCamera: EyeImage] = [:]
        for camera in cameras {
            let pose = try scene.cameraPose(camera)
            images[camera] = try renderer.render(camera: camera, pose: pose, geoms: geoms)
        }
        return images
    }
}
