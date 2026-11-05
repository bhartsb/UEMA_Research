// VisionPatcher.swift: the vision tokenizer of the UEMA Apple port (work item WI-30, target B
// of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-6, the
// patcher half: "the patcher cuts 16 x 16 patches, / 255, into (N,768) with (N,4) angles
// (azimuth, elevation, eye, span); N = 1,160 for wide plus fovea").
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1).
//
// WHAT THIS FILE IS. The port of `BinocularPatcher.__call__` (Source/Python/mimo_bridge/
// vision_adapter.py:136-214) with its helpers `patch_grid_rays` (:68-83), `compose_gaze`
// (:86-101) and `image_to_patches` (:104-114).
//
// INPUT. Four RGB 8-bit images (rows top-down, columns left-to-right, 3 bytes per pixel; the
// WI-14 fixture's cameras.json "image_format"): eye_left and eye_right, 256 x 256 px over a
// 60 degree vertical field of view; fovea_left and fovea_right, 288 x 288 px over 24 degrees
// (the released run's `vision_layout` [["wide", 256, 60.0], ["fovea", 288, 24.0]], fixture
// patches.json "world"). ring_left is rendered for the snap reflex only and is not patched
// (env.py:1494-1502 per UEMASenses/Vision/CameraSpec.swift; it is not in `vision_layout`).
// Plus each eye's gaze: yaw and pitch in radians, the eye joints' qpos with the right yaw
// negated (env.eye_yaw_pitch, env.py:2212-2220, as patches.json "gaze" records it).
//
// OPERATION, per source (wide first, then fovea; vision_adapter.py:194-212):
//  1. Each image is cropped to whole 16-pixel patches, every byte divided by 255 in float32,
//     and cut into rows of 768 = 16 rows x 16 columns x 3 colours, row-major inside the patch
//     (y, x, rgb), patches row-major over the image's grid (:110-113): 16 x 16 = 256 patches
//     per 256 px image, 18 x 18 = 324 per 288 px image.
//  2. Each patch centre's ray in the eye's rest frame, columns (forward, left, up): with
//     ty = tan(fovy / 2), tx = ty x width / height, x = (2 ((c 16 + 8) / width) - 1) tx,
//     y = (1 - 2 ((r 16 + 8) / height)) ty, n = sqrt(1 + x^2 + y^2), ray = (1/n, -x/n, y/n)
//     (:72-83), in float64.
//  3. The gaze composed onto each ray (:93-101), float64, stored as float32: f = ray_f cos(pitch)
//     - ray_u sin(pitch), l = ray_l, u = ray_f sin(pitch) + ray_u cos(pitch); azimuth =
//     atan2(f sin(yaw) + l cos(yaw), f cos(yaw) - l sin(yaw)) (positive = eye-left), elevation
//     = asin(clip(u, -1, 1)). The yaw and pitch are first rounded to float32 (:191-192).
//  4. Left and right interleaved: row 2k = left patch k, row 2k + 1 = right patch k (:202-209).
//     Angle columns: azimuth, elevation, eye (-1 left, +1 right), span = radians(fovy) x 16 /
//     size, stored as float32 (:171, :205-210): 0.06544985 rad for the wide pair, 0.023271056
//     for the fovea pair.
//
// OUTPUT. `patches` [N, 768] float32 in 0..1 and `angles` [N, 4] float32, N = 2 x (256 + 324) =
// 1,160 for the released layout (patches.json "n_total" 1160, "expected_n_rule"). These are the
// vision encoder's inputs "patches" and "angular_pos" (lem_model.py:2396-2409).
//
// THE SenseReading LAYOUT this tokenizer reads (`VisionPatcher.reading(images:left:right:)`
// builds it): `primary` shape [V], the patched cameras' bytes as float32 0..255 laid end to end
// in source order, left camera before right camera (eye_left, eye_right, fovea_left,
// fovea_right), each width x height x 3 row-major; V = 2 x 256^2 x 3 + 2 x 288^2 x 3 =
// 890,880. `auxiliary` shape [2, 2]: [[left yaw, left pitch], [right yaw, right pitch]] in
// radians. The UEMACore comment on `SenseReading` names "[5, height, width, 3]" for vision; one
// array of that shape cannot hold images of 256 and 288 px, so this layout replaces it.
//
// SOLID. `VisionPatcher` has one job (images and gaze to patch rows); the camera set is data
// (`VisionSource`), so another layout is another source list with no change here.
//
// FIRST BUILD ON A MAC (work item WI-30's acceptance: "VisionPatcher on the WI-14 PNGs gives N
// equal to the recorded count and the first 4 angle rows within 1e-5; each other tokenizer
// reproduces one hand-computed row"). After the package resolves (Package.swift header, steps 1
// to 3; the MuJoCo XCFramework must be in place for the manifest to load):
//  1. `cd apple/UEMAKit && xcodebuild -scheme UEMAKit-Package build -destination
//     'platform=macOS'` : exit code 0. Constructs to check first if this module fails: the
//     existential `any Tokenizer` in TokenizerBank.swift (Swift 5.6 or newer); the tuple return
//     of `VisionPatcher.patch(images:left:right:)`; `Set<Int>` literals in TokenizerSupport.swift.
//  2. `xcodebuild -scheme UEMAKit-Package test -destination 'platform=macOS'
//     -only-testing:UEMATokenizerTests` : exit code 0, every test passing:
//     - VisionPatcherTests: N = 1,160 on the released layout; a 32 x 32 synthetic image pair
//       gives the hand-computed patch bytes and the angles 0.2810349 / 0.2705498 rad (patch 0,
//       gaze 0) and 0.5713975 / 0.0780811 rad (gaze yaw 0.3, pitch -0.2); and, on each of the
//       three WI-14 states under Tests/UEMASensesTests/Fixtures/eyes/, N equal to patches.json
//       "n_total", the first 4 angle rows within 1e-5 of "first_angle_rows", the first 12
//       patch numbers equal to "first_patch_row_head", and the SHA-256 of the [1160, 768]
//       float32 bytes equal to "patches_sha256" (the Python patcher reproduced both digests
//       from the same PNGs on November 4, 2026). The test prints whether the angle digest
//       "angles_sha256" also matches: Darwin's atan2 and asin may differ from numpy's in the
//       last float64 bit, so that line is recorded, not asserted. The fixture test reads the
//       PNGs by path from #filePath (the sources must be on disk; it skips with a message
//       otherwise) and decodes them with ImageIO.
//     - TactileTokenizerTests, AudioTokenizerTests, ProprioceptionTokenizerTests,
//       MotorTokenizerTests, VestibularNormaliserTests, TokenizerBankTests: one hand-computed
//       row each (values in the test files, computed with numpy on November 4, 2026); the
//       tactile tests also feed the WI-28 reader's [256, 10] + [256, 11] layout and check that its
//       253 unused slots stay out of the token, and the motor tests scale the motor reader's
//       auxiliary [24].
//     The fixture PNGs are 8-bit truecolour (IHDR colour type 2, read November 4, 2026), with no
//     gAMA, sRGB or iCCP chunk, so ImageIO's data provider gives the stored bytes; a loader error
//     names the file and the bytes per pixel it found.
//  3. If the fixture test fails on the patch digest but passes "first_patch_row_head", the PNG
//     decode changed bytes past the first pixels (a colour-managed decode): the test helper reads
//     the provider's raw bytes, so check the CGImage's bitsPerPixel and alphaInfo first.
//  4. Write each step's command, exit code and clock time (AM/PM Pacific) into
//     apple/FIRST_BUILD_RECORD.md (WI-41), with the angle digest line of step 2.

import Foundation
import UEMACore

/// One camera pair the patcher cuts: a left and a right camera of the same square size and
/// field of view (vision_adapter.py:139-140, :164-172).
public struct VisionSource: Equatable, Hashable, Sendable {
    /// The source's name in the run's `vision_layout`: "wide" or "fovea".
    public let name: String
    /// The left camera.
    public let leftCamera: EyeCamera
    /// The right camera.
    public let rightCamera: EyeCamera
    /// Width and height of each image in pixels.
    public let sizePixels: Int
    /// Vertical (and horizontal, the image being square) field of view in degrees.
    public let fieldOfViewDegrees: Double

    /// A source; the patcher's init checks the ranges.
    public init(name: String, leftCamera: EyeCamera, rightCamera: EyeCamera, sizePixels: Int,
                fieldOfViewDegrees: Double) {
        self.name = name
        self.leftCamera = leftCamera
        self.rightCamera = rightCamera
        self.sizePixels = sizePixels
        self.fieldOfViewDegrees = fieldOfViewDegrees
    }

    /// The wide pair: eye_left and eye_right, 256 px over 60 degrees (patches.json "sources").
    public static let wide = VisionSource(
        name: "wide", leftCamera: .eyeLeft, rightCamera: .eyeRight, sizePixels: 256, fieldOfViewDegrees: 60)

    /// The fovea pair: fovea_left and fovea_right, 288 px over 24 degrees (patches.json "sources").
    public static let fovea = VisionSource(
        name: "fovea", leftCamera: .foveaLeft, rightCamera: .foveaRight, sizePixels: 288, fieldOfViewDegrees: 24)

    /// The released run's layout, coarsest first (patches.json "vision_layout").
    public static let released: [VisionSource] = [wide, fovea]
}

/// One eye's gaze: the eye joints' yaw and pitch in radians (env.eye_yaw_pitch,
/// env.py:2212-2220; the right yaw is negated there, so both eyes use the same sign rule:
/// positive yaw = eye-left).
public struct EyeGaze: Equatable, Hashable, Codable, Sendable {
    /// Rotation about the head's up axis, radians, positive = eye-left.
    public let yaw: Double
    /// Rotation about the eye's left axis, radians.
    public let pitch: Double

    /// A gaze of `yaw` and `pitch` radians.
    public init(yaw: Double, pitch: Double) {
        self.yaw = yaw
        self.pitch = pitch
    }

    /// The eye looking straight ahead: yaw 0, pitch 0.
    public static let level = EyeGaze(yaw: 0, pitch: 0)
}

/// The unit ray through one patch centre in the eye's rest frame, columns (forward, left, up)
/// (vision_adapter.py:68-83).
public struct PatchRay: Equatable, Hashable, Sendable {
    /// The component along the eye's forward axis.
    public let forward: Double
    /// The component along the eye's left axis.
    public let left: Double
    /// The component along the eye's up axis.
    public let up: Double

    /// A ray of the three components given (the caller keeps it of unit length).
    public init(forward: Double, left: Double, up: Double) {
        self.forward = forward
        self.left = left
        self.up = up
    }
}

/// The vision tokenizer: four eye images and the two gazes into the vision encoder's patch rows
/// [N, 768] and angle rows [N, 4] (vision_adapter.py:136-214). The ray tables are computed once
/// at init (vision_adapter.py:164-174).
public struct VisionPatcher: Tokenizer, Sendable {
    /// Pixels per patch side: 16 (`PATCH`, vision_adapter.py:26; tokenizer/config.py:251).
    public static let releasedPatchSize = 16

    /// Numbers per patch row: 16 x 16 x 3 = 768.
    public var patchRowLength: Int { patchSize * patchSize * 3 }

    /// The sense served: vision.
    public var sense: SenseName { .vision }

    /// The camera pairs, in output block sequence.
    public let sources: [VisionSource]

    /// Pixels per patch side.
    public let patchSize: Int

    /// Per source, the ray of each patch centre, row-major over the grid.
    public let rays: [[PatchRay]]

    /// Per source, the span angle of one patch in radians as float32 (vision_adapter.py:171).
    public let spans: [Float]

    /// Patches per eye summed over the sources: 256 + 324 = 580 on the released layout.
    public let patchesPerEye: Int

    /// Patch rows in the output: 2 x `patchesPerEye` = 1,160 on the released layout.
    public var patchCount: Int { 2 * patchesPerEye }

    /// A patcher over `sources`; throws `TokenizerError.invalidSource` for a source smaller
    /// than one patch, a field of view outside (0, 180) degrees, or no source.
    public init(sources: [VisionSource] = VisionSource.released, patchSize: Int = VisionPatcher.releasedPatchSize) throws {
        guard patchSize > 0 else {
            throw TokenizerError.invalidSource(reason: "patch size \(patchSize) is not positive")
        }
        guard !sources.isEmpty else {
            throw TokenizerError.invalidSource(reason: "no vision source")
        }
        for source in sources {
            guard source.sizePixels >= patchSize else {
                throw TokenizerError.invalidSource(
                    reason: "source \(source.name): \(source.sizePixels) px is smaller than one \(patchSize) px patch")
            }
            guard source.fieldOfViewDegrees > 0, source.fieldOfViewDegrees < 180 else {
                throw TokenizerError.invalidSource(
                    reason: "source \(source.name): field of view \(source.fieldOfViewDegrees) degrees is outside (0, 180)")
            }
        }
        self.init(validatedSources: sources, patchSize: patchSize)
    }

    private init(validatedSources sources: [VisionSource], patchSize: Int) {
        self.sources = sources
        self.patchSize = patchSize
        var rays: [[PatchRay]] = []
        var spans: [Float] = []
        var perEye = 0
        for source in sources {
            let table = VisionPatcher.patchGridRays(
                width: source.sizePixels, height: source.sizePixels,
                fieldOfViewDegrees: source.fieldOfViewDegrees, patchSize: patchSize)
            rays.append(table)
            perEye += table.count
            // span = math.radians(fovy) * patch / float(size), stored into a float32 column
            // (vision_adapter.py:171, :210); math.radians(x) is x * (pi / 180).
            let span = source.fieldOfViewDegrees * (Double.pi / 180.0) * Double(patchSize) / Double(source.sizePixels)
            spans.append(Float(span))
        }
        self.rays = rays
        self.spans = spans
        self.patchesPerEye = perEye
    }

    /// The released patcher: the wide and fovea pairs, 16 px patches, 1,160 rows.
    public static func released() -> VisionPatcher {
        VisionPatcher(validatedSources: VisionSource.released, patchSize: releasedPatchSize)
    }

    // MARK: - The geometry

    /// `patch_grid_rays` (vision_adapter.py:68-83): the unit ray of every patch centre of a
    /// pinhole camera, row-major over the grid of (height / patch) x (width / patch) patches,
    /// in float64.
    public static func patchGridRays(width: Int, height: Int, fieldOfViewDegrees: Double,
                                     patchSize: Int = releasedPatchSize) -> [PatchRay] {
        let rows = height / patchSize
        let cols = width / patchSize
        let fovy = fieldOfViewDegrees * (Double.pi / 180.0)
        let ty = tan(fovy / 2.0)
        let tx = ty * (Double(width) / Double(height))
        let half = Double(patchSize) / 2.0
        var out: [PatchRay] = []
        out.reserveCapacity(rows * cols)
        for r in 0 ..< rows {
            for c in 0 ..< cols {
                let x = (2.0 * ((Double(c * patchSize) + half) / Double(width)) - 1.0) * tx
                let y = (1.0 - 2.0 * ((Double(r * patchSize) + half) / Double(height))) * ty
                let n = (1.0 + x * x + y * y).squareRoot()
                out.append(PatchRay(forward: 1.0 / n, left: -x / n, up: y / n))
            }
        }
        return out
    }

    /// `compose_gaze` (vision_adapter.py:86-101): each ray turned by the eye's pitch (inner
    /// joint, about the left axis) then its yaw (outer joint, about the up axis), as (azimuth,
    /// elevation) radians referenced to the head, computed in float64 and rounded to float32.
    public static func composeGaze(rays: [PatchRay], yaw: Double, pitch: Double) -> [(azimuth: Float, elevation: Float)] {
        let cosPitch = cos(pitch)
        let sinPitch = sin(pitch)
        let cosYaw = cos(yaw)
        let sinYaw = sin(yaw)
        return rays.map { ray -> (azimuth: Float, elevation: Float) in
            let f = ray.forward * cosPitch - ray.up * sinPitch
            let l = ray.left
            let u = ray.forward * sinPitch + ray.up * cosPitch
            let azimuth = atan2(f * sinYaw + l * cosYaw, f * cosYaw - l * sinYaw)
            let elevation = asin(min(max(u, -1.0), 1.0))
            return (Float(azimuth), Float(elevation))
        }
    }

    // MARK: - The patches

    /// `image_to_patches` (vision_adapter.py:104-114): the image cropped to whole patches, each
    /// byte / 255 in float32, written into `output` from `rowOffset` with `rowStride` rows
    /// between consecutive patches (2 for the interleaved eyes). Returns the patch count.
    @discardableResult
    static func writePatches(of image: EyeImage, patchSize: Int, into output: inout [Float],
                             firstRow: Int, rowStride: Int) -> Int {
        let rows = image.height / patchSize
        let cols = image.width / patchSize
        let rowLength = patchSize * patchSize * 3
        let width = image.width
        image.rgb.withUnsafeBufferPointer { pixels in
            for r in 0 ..< rows {
                for c in 0 ..< cols {
                    let patchIndex = r * cols + c
                    var o = (firstRow + patchIndex * rowStride) * rowLength
                    for py in 0 ..< patchSize {
                        let y = r * patchSize + py
                        var p = (y * width + c * patchSize) * 3
                        for _ in 0 ..< patchSize * 3 {
                            output[o] = Float(pixels[p]) / 255.0
                            o += 1
                            p += 1
                        }
                    }
                }
            }
        }
        return rows * cols
    }

    /// INPUT: `images` holding each source's left and right camera (other cameras, the ring
    /// among them, are ignored); the two gazes in radians. OPERATION: the four steps of this
    /// file's header. OUTPUT: `patches` [N, 768] and `angles` [N, 4], N = `patchCount`.
    /// Throws `TokenizerError.missingImage` or `.imageSizeMismatch`.
    public func patch(images: [EyeCamera: EyeImage], left: EyeGaze, right: EyeGaze) throws
        -> (patches: ShapedArray, angles: ShapedArray) {
        let rowLength = patchRowLength
        var patches = [Float](repeating: 0, count: patchCount * rowLength)
        var angles = [Float](repeating: 0, count: patchCount * 4)
        // lyp = np.asarray(left_yaw_pitch, dtype=np.float32), then float(lyp[i]) (:191-192, :206).
        let leftYaw = Double(Float(left.yaw))
        let leftPitch = Double(Float(left.pitch))
        let rightYaw = Double(Float(right.yaw))
        let rightPitch = Double(Float(right.pitch))
        var blockStart = 0
        for (index, source) in sources.enumerated() {
            let table = rays[index]
            let n = table.count
            let leftImage = try image(for: source.leftCamera, in: images, source: source)
            let rightImage = try image(for: source.rightCamera, in: images, source: source)
            VisionPatcher.writePatches(of: leftImage, patchSize: patchSize, into: &patches,
                                       firstRow: blockStart, rowStride: 2)
            VisionPatcher.writePatches(of: rightImage, patchSize: patchSize, into: &patches,
                                       firstRow: blockStart + 1, rowStride: 2)
            let leftAngles = VisionPatcher.composeGaze(rays: table, yaw: leftYaw, pitch: leftPitch)
            let rightAngles = VisionPatcher.composeGaze(rays: table, yaw: rightYaw, pitch: rightPitch)
            let span = spans[index]
            for k in 0 ..< n {
                let l = (blockStart + 2 * k) * 4
                angles[l] = leftAngles[k].azimuth
                angles[l + 1] = leftAngles[k].elevation
                angles[l + 2] = -1.0
                angles[l + 3] = span
                let r = l + 4
                angles[r] = rightAngles[k].azimuth
                angles[r + 1] = rightAngles[k].elevation
                angles[r + 2] = 1.0
                angles[r + 3] = span
            }
            blockStart += 2 * n
        }
        let patchArray = try ShapedArray(shape: [patchCount, rowLength], values: patches)
        let angleArray = try ShapedArray(shape: [patchCount, 4], values: angles)
        return (patchArray, angleArray)
    }

    /// The image of `camera`, checked against the source's patch grid: `image_to_patches` crops
    /// to whole patches, so any size with the same grid passes, as in Python (:198-201).
    private func image(for camera: EyeCamera, in images: [EyeCamera: EyeImage], source: VisionSource) throws -> EyeImage {
        guard let image = images[camera] else {
            throw TokenizerError.missingImage(camera: camera.mujocoName)
        }
        let grid = source.sizePixels / patchSize
        guard image.width / patchSize == grid, image.height / patchSize == grid else {
            throw TokenizerError.imageSizeMismatch(
                camera: camera.mujocoName, expectedSize: source.sizePixels, width: image.width, height: image.height)
        }
        return image
    }

    // MARK: - The SenseReading layout

    /// The cameras the reading carries, in their sequence inside `primary`.
    public var readingCameras: [EyeCamera] {
        sources.flatMap { [$0.leftCamera, $0.rightCamera] }
    }

    /// The length of `primary` in the reading layout: each camera's size^2 x 3 bytes, summed
    /// (890,880 on the released layout).
    public var readingValueCount: Int {
        sources.reduce(0) { $0 + 2 * $1.sizePixels * $1.sizePixels * 3 }
    }

    /// Packs the patched cameras' images and the two gazes into the `SenseReading` layout of
    /// this file's header (`primary` [V] bytes as float32, `auxiliary` [2, 2] yaw and pitch).
    /// Each image must be exactly its source's size.
    public func reading(images: [EyeCamera: EyeImage], left: EyeGaze, right: EyeGaze,
                        isFresh: Bool = true) throws -> SenseReading {
        var values: [Float] = []
        values.reserveCapacity(readingValueCount)
        for source in sources {
            for camera in [source.leftCamera, source.rightCamera] {
                guard let image = images[camera] else {
                    throw TokenizerError.missingImage(camera: camera.mujocoName)
                }
                guard image.width == source.sizePixels, image.height == source.sizePixels else {
                    throw TokenizerError.imageSizeMismatch(
                        camera: camera.mujocoName, expectedSize: source.sizePixels,
                        width: image.width, height: image.height)
                }
                values.append(contentsOf: image.rgb.map { Float($0) })
            }
        }
        let gaze: [Float] = [Float(left.yaw), Float(left.pitch), Float(right.yaw), Float(right.pitch)]
        return SenseReading(
            sense: .vision,
            primary: try ShapedArray(shape: [values.count], values: values),
            auxiliary: try ShapedArray(shape: [2, 2], values: gaze),
            isFresh: isFresh)
    }

    /// Reads the `SenseReading` layout of this file's header back into images and gazes and
    /// patches them. OUTPUT: `primary` [N, 768], `auxiliary` [N, 4]; `isFresh` carried.
    public func tokenize(_ reading: SenseReading) throws -> EncoderInput {
        try TokenizerChecks.requireSense(reading, .vision)
        guard reading.primary.rank == 1, reading.primary.count == readingValueCount else {
            throw TokenizerError.wrongShape(
                sense: .vision, array: "primary",
                expected: "[\(readingValueCount)]: the patched cameras' RGB bytes end to end", found: reading.primary.shape)
        }
        guard let gaze = reading.auxiliary, gaze.shape == [2, 2] else {
            throw TokenizerError.wrongShape(
                sense: .vision, array: "auxiliary", expected: "[2, 2]: left and right yaw, pitch in radians",
                found: reading.auxiliary?.shape ?? [])
        }
        var images: [EyeCamera: EyeImage] = [:]
        var offset = 0
        for source in sources {
            let length = source.sizePixels * source.sizePixels * 3
            for camera in [source.leftCamera, source.rightCamera] {
                let bytes = reading.primary.values[offset ..< offset + length].map { value -> UInt8 in
                    value.isFinite ? UInt8(min(max(value.rounded(), 0), 255)) : 0
                }
                images[camera] = try EyeImage(
                    camera: camera, width: source.sizePixels, height: source.sizePixels, rgb: bytes)
                offset += length
            }
        }
        let left = EyeGaze(yaw: Double(gaze.values[0]), pitch: Double(gaze.values[1]))
        let right = EyeGaze(yaw: Double(gaze.values[2]), pitch: Double(gaze.values[3]))
        let result = try patch(images: images, left: left, right: right)
        return EncoderInput(sense: .vision, primary: result.patches, auxiliary: result.angles, isFresh: reading.isFresh)
    }
}
