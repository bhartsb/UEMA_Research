// EyesView.swift: the five eye camera images of the vision sense. Work item WI-38,
// target B of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html (row WI-38:
// "EyesView: five camera images"); specification REQ-B-14: "the 5 eye images".
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). SwiftUI plus CoreGraphics for the pixel buffer; no
// UIKit or AppKit type: no platform image class is used (specification T-B-7).
//
// INPUT: one `EyeImage` per `EyeCamera` (apple/UEMAKit/Sources/UEMACore/Protocols.swift:
// 832-874 for EyeCamera, 970-996 for EyeImage): RGB 8-bit, rows top-down, columns left-to-right, width x height x 3
// bytes (cameras.json image_format of WI-14); eye_left and eye_right 256 px over 60
// degrees, fovea_left and fovea_right 288 px over 24 degrees, ring_left 256 px over 150
// degrees. A camera with no image yet (before the first render) shows a placeholder.
// `renderedAt` is the model step of the render when the caller passes one.
// OPERATION: `EyeImageRaster.cgImage(from:)` wraps the bytes in a CGImage of 8 bits per
// component, 24 bits per pixel, no alpha, device RGB; SwiftUI draws it with
// `Image(decorative:scale:)` and no interpolation, so each rendered pixel stays one
// square. OUTPUT: pixels on screen; nothing is written back.
//
// The "First build on a Mac" checklist of this work item is in UEMAMindApp.swift's header.

import CoreGraphics
import Foundation
import SwiftUI
import UEMACore

/// Converts an `EyeImage` into a CGImage for display. One responsibility: the pixel
/// layout; the view does the drawing.
enum EyeImageRaster {
    /// Bits per colour component: 8 (RGB 8-bit, cameras.json image_format).
    static let bitsPerComponent = 8

    /// Bytes per pixel: 3 (red, green, blue; no alpha).
    static let bytesPerPixel = 3

    /// A CGImage of `image`'s width and height over its RGB bytes, or nil when
    /// CoreGraphics refuses the buffer (a zero width or height).
    static func cgImage(from image: EyeImage) -> CGImage? {
        guard image.width > 0, image.height > 0 else { return nil }
        guard let provider = CGDataProvider(data: Data(image.rgb) as CFData) else { return nil }
        return CGImage(
            width: image.width,
            height: image.height,
            bitsPerComponent: bitsPerComponent,
            bitsPerPixel: bitsPerComponent * bytesPerPixel,
            bytesPerRow: image.width * bytesPerPixel,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent)
    }

    /// The panel heading: "Eye images rendered at model step 12" when the step is known,
    /// "Eye images of the latest render" when images exist without it, "No eye render
    /// yet" otherwise.
    static func heading(renderedAt: ModelStep?, hasImages: Bool) -> String {
        if let renderedAt {
            return "Eye images rendered at model step \(renderedAt.count)"
        }
        return hasImages ? "Eye images of the latest render" : "No eye render yet"
    }

    /// The caption of one camera: its scene name, pixel size and vertical field of view,
    /// such as "fovea_left, 288 px, 24 degrees".
    static func caption(for camera: EyeCamera) -> String {
        "\(camera.mujocoName), \(camera.pixelSize) px, \(Int(camera.fieldOfViewDegrees)) degrees"
    }
}

/// The five eye images in `EyeCamera.allCases` sequence (eye_left, eye_right, fovea_left,
/// fovea_right, ring_left), each with its caption.
struct EyesView: View {
    /// The latest image per camera; a missing camera shows a placeholder.
    let images: [EyeCamera: EyeImage]

    /// The model step of the latest render; nil before the first.
    let renderedAt: ModelStep?

    private let columns = [GridItem(.adaptive(minimum: 140, maximum: 288), spacing: 8)]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(EyeImageRaster.heading(renderedAt: renderedAt, hasImages: !images.isEmpty))
                .font(.headline)
            LazyVGrid(columns: columns, alignment: .leading, spacing: 8) {
                ForEach(EyeCamera.allCases, id: \.self) { camera in
                    VStack(alignment: .leading, spacing: 4) {
                        eyePicture(camera)
                            .aspectRatio(1, contentMode: .fit)
                            .frame(minWidth: 128, maxWidth: 288)
                        Text(EyeImageRaster.caption(for: camera))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    @ViewBuilder
    private func eyePicture(_ camera: EyeCamera) -> some View {
        if let image = images[camera], let cgImage = EyeImageRaster.cgImage(from: image) {
            Image(decorative: cgImage, scale: 1)
                .interpolation(.none)
                .resizable()
        } else {
            Rectangle()
                .fill(Color.gray.opacity(0.25))
                .overlay {
                    Text("no image").font(.caption).foregroundStyle(.secondary)
                }
        }
    }
}

#if DEBUG
/// Synthetic images for the preview: each camera a horizontal ramp from black to one
/// colour, at the camera's own pixel size. Not a render of the scene.
enum EyesPreviewImages {
    static func ramps() -> [EyeCamera: EyeImage] {
        var out: [EyeCamera: EyeImage] = [:]
        for (index, camera) in EyeCamera.allCases.enumerated() {
            let size = camera.pixelSize
            var rgb = [UInt8](repeating: 0, count: size * size * 3)
            for row in 0 ..< size {
                for column in 0 ..< size {
                    let level = UInt8(column * 255 / max(size - 1, 1))
                    let base = (row * size + column) * 3
                    rgb[base + index % 3] = level
                }
            }
            if let image = try? EyeImage(camera: camera, width: size, height: size, rgb: rgb) {
                out[camera] = image
            }
        }
        return out
    }
}

#Preview("Eyes, synthetic ramps") {
    ScrollView {
        EyesView(images: EyesPreviewImages.ramps(), renderedAt: ModelStep(count: 2))
            .padding()
    }
}
#endif
