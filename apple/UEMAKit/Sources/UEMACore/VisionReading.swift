// Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
// Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
//
// VisionReading.swift: the one layout of the vision sense's `SenseReading`, shared by the reader
// that renders it (UEMASenses VisionReader) and the tokenizer that patches it (UEMATokenizer
// VisionPatcher). Written November 5, 2026 for the cross-module consistency pass (item 1 of
// docs/apple_port/HANDOFF_2026-11-04.md): before it, the `SenseReading` comment named vision as
// [5, height, width, 3], an array shape that cannot hold images of 256 and 288 px, while
// VisionPatcher read the layout below, and no vision reader existed.
//
// THE LAYOUT.
//   primary   [V] float32: the RGB bytes (0..255) of each camera in `cameras`, in that sequence,
//             each width x height x 3 row-major, laid end to end. On the released layout the
//             cameras are eye_left, eye_right (256 px), fovea_left, fovea_right (288 px), so
//             V = 2 x 256^2 x 3 + 2 x 288^2 x 3 = 890,880. ring_left is rendered by the scene
//             but is not in the released run's vision_layout, so it is not carried.
//   auxiliary [2, 2] float32: [[left yaw, left pitch], [right yaw, right pitch]] in radians,
//             the eye joints' angles (env.eye_yaw_pitch, Source/Python/mimo_bridge/env.py:2212-2219;
//             the right yaw is negated there so both eyes use one sign rule).

/// One eye's gaze: the eye joints' yaw and pitch in radians (env.eye_yaw_pitch,
/// env.py:2212-2219; the right yaw is negated there, so both eyes use the same sign rule:
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

/// The vision `SenseReading` layout of this file's header: which cameras, at which square size,
/// in which sequence.
public struct VisionReadingLayout: Equatable, Hashable, Sendable {
    /// One camera of the layout and its square image size in pixels.
    public struct Entry: Equatable, Hashable, Sendable {
        /// The camera.
        public let camera: EyeCamera
        /// Width and height of its image in pixels.
        public let pixelSize: Int

        /// An entry; `pixelSize` defaults to the camera's released size.
        public init(_ camera: EyeCamera, pixelSize: Int? = nil) {
            self.camera = camera
            self.pixelSize = pixelSize ?? camera.pixelSize
        }
    }

    /// The cameras in their sequence inside `primary`.
    public let entries: [Entry]

    /// A layout of `entries`.
    public init(entries: [Entry]) {
        self.entries = entries
    }

    /// The released run's vision_layout: eye_left, eye_right (256 px), fovea_left, fovea_right
    /// (288 px).
    public static let released = VisionReadingLayout(
        entries: [Entry(.eyeLeft), Entry(.eyeRight), Entry(.foveaLeft), Entry(.foveaRight)])

    /// The cameras, in sequence.
    public var cameras: [EyeCamera] { entries.map(\.camera) }

    /// The length of `primary`: each entry's pixelSize^2 x 3, summed (890,880 for `released`).
    public var valueCount: Int {
        entries.reduce(0) { $0 + $1.pixelSize * $1.pixelSize * 3 }
    }

    /// The shape of `auxiliary`: [2, 2].
    public static let gazeShape = [2, 2]

    /// Packs `images` (one per entry, each of the entry's size) and the two gazes into the
    /// layout. A missing camera throws `UEMACoreError.missingCamera`; an image of another size
    /// throws `UEMACoreError.wrongShape`.
    public func reading(images: [EyeCamera: EyeImage], left: EyeGaze, right: EyeGaze,
                        isFresh: Bool = true) throws -> SenseReading {
        var values: [Float] = []
        values.reserveCapacity(valueCount)
        for entry in entries {
            let name = entry.camera.mujocoName
            guard let image = images[entry.camera] else { throw UEMACoreError.missingCamera(name) }
            guard image.width == entry.pixelSize, image.height == entry.pixelSize else {
                throw UEMACoreError.wrongShape(type: "EyeImage \(name)", expected: [entry.pixelSize, entry.pixelSize],
                                               found: [image.height, image.width])
            }
            values.append(contentsOf: image.rgb.map { Float($0) })
        }
        let gaze: [Float] = [Float(left.yaw), Float(left.pitch), Float(right.yaw), Float(right.pitch)]
        return SenseReading(
            sense: .vision,
            primary: try ShapedArray(shape: [values.count], values: values),
            auxiliary: try ShapedArray(shape: Self.gazeShape, values: gaze),
            isFresh: isFresh)
    }

    /// The images and gazes of a reading in the layout, the inverse of `reading`. A reading of
    /// another sense throws `UEMACoreError.missingSense(.vision)`; a primary or auxiliary of
    /// another shape throws `UEMACoreError.wrongShape`; each value is rounded to the nearest
    /// byte and clamped to 0..255, and a NaN or infinite value reads as 0.
    public func images(of reading: SenseReading) throws -> (images: [EyeCamera: EyeImage], left: EyeGaze, right: EyeGaze) {
        guard reading.sense == .vision else { throw UEMACoreError.missingSense(.vision) }
        guard reading.primary.shape == [valueCount] else {
            throw UEMACoreError.wrongShape(type: "vision primary", expected: [valueCount], found: reading.primary.shape)
        }
        guard let gaze = reading.auxiliary, gaze.shape == Self.gazeShape else {
            throw UEMACoreError.wrongShape(type: "vision auxiliary", expected: Self.gazeShape,
                                           found: reading.auxiliary?.shape ?? [])
        }
        var images: [EyeCamera: EyeImage] = [:]
        var offset = 0
        for entry in entries {
            let length = entry.pixelSize * entry.pixelSize * 3
            let bytes = reading.primary.values[offset ..< offset + length].map { value -> UInt8 in
                value.isFinite ? UInt8(Swift.min(Swift.max(value.rounded(), 0), 255)) : 0
            }
            images[entry.camera] = try EyeImage(camera: entry.camera, width: entry.pixelSize,
                                                height: entry.pixelSize, rgb: bytes)
            offset += length
        }
        let g = gaze.values
        return (images, EyeGaze(yaw: Double(g[0]), pitch: Double(g[1])), EyeGaze(yaw: Double(g[2]), pitch: Double(g[3])))
    }
}
