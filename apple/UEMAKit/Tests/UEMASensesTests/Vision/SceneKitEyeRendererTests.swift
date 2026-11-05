// SceneKitEyeRendererTests.swift: tests of Sources/UEMASenses/Vision/SceneKitEyeRenderer.swift
// (work item WI-29). They need a Metal device and skip without one. Nothing here was compiled
// on the machine that wrote it (no Swift toolchain).
//
// WHAT THEY CHECK.
// - Synthetic scenes (no fixture needed): the image size of each of the five cameras; an
//   emissive red box straight ahead fills the centre and leaves the corners at the background;
//   a box to the camera's right lands in the right half, a box above lands in the top rows (rows
//   top-down, no flip); a grey box lit only by MuJoCo's headlight gives MuJoCo's fixed-function
//   value 0.5 x 0.1 (ambient) + 0.5 x 0.4 (diffuse, face toward the light) = 0.25, which is 64 of
//   255, within 6 steps (the colour-space check of the renderer's header); a near box in front of
//   a far box hides it (the depth buffer's clear value agrees with its direction).
// - Without a GPU: the released crib's clipping distances (0.005 x 3.23 m and 30 x 3.23 m,
//   scene.xml:9 and the exporter's extent) and the released lights (scene.xml:38-39 plus
//   MuJoCo's default headlight).
// - The WI-14 fixture states (WI-29's acceptance): for each state folder with an
//   abstract_scene.json (written by Fixtures/eyes/export_abstract_scene.py), every camera is
//   rendered from that file's geom list, lights and extent at cameras.json's recorded pose and
//   compared with the Python PNG; one line per state and camera is printed and all lines go to
//   NSTemporaryDirectory()/uema_eye_differences.json. No pass line is asserted: the plan sets it
//   in section 9 from these numbers. Without abstract_scene.json the test skips and names the
//   exporter.
//
// FIRST BUILD ON A MAC: `cd apple/UEMAKit && xcodebuild -scheme UEMAKit-Package test
// -destination 'platform=macOS' -only-testing:UEMASensesTests/SceneKitEyeRendererTests` :
// exit 0, 7 tests (the fixture test skipped until step 2 of the renderer's checklist has run).

import Foundation
import ImageIO
import Metal
import simd
import XCTest
import UEMACore
import UEMASenses

final class SceneKitEyeRendererTests: XCTestCase {
    private let identity: [Double] = [1, 0, 0, 0, 1, 0, 0, 0, 1]

    private func makeRenderer(lights: [EyeLight] = [], extent: Double = 1, supersampling: Int = 2) throws -> SceneKitEyeRenderer {
        guard MTLCreateSystemDefaultDevice() != nil else {
            throw XCTSkip("no Metal device on this machine")
        }
        return try SceneKitEyeRenderer(settings: EyeRenderSettings(extentMetres: extent, supersampling: supersampling),
                                       lights: FixedEyeLights(lights))
    }

    private func box(at position: [Float], half: Float, rgba: [Float]) throws -> SceneGeom {
        try SceneGeom(type: 6, size: [half, half, half], position: position,
                      rotation: [1, 0, 0, 0, 1, 0, 0, 0, 1], rgba: rgba)
    }

    private var originPose: CameraPose {
        get throws { try CameraPose(position: [0, 0, 0], rotation: identity) }
    }

    /// Every camera renders at its own square size; an empty scene is the background (black).
    func testImageSizePerCameraAndEmptyScene() throws {
        let renderer = try makeRenderer()
        for camera in EyeCamera.allCases {
            let image = try renderer.render(camera: camera, pose: try originPose, geoms: [])
            XCTAssertEqual(image.camera, camera)
            XCTAssertEqual(image.width, camera.pixelSize)
            XCTAssertEqual(image.height, camera.pixelSize)
            XCTAssertEqual(image.rgb.count, camera.pixelSize * camera.pixelSize * 3)
            XCTAssertEqual(image.rgb.max() ?? 0, 0, "\(camera.rawValue): empty scene must be black")
        }
    }

    /// An emissive red box 1 m ahead (the camera looks along its local -z) fills the centre; the
    /// corners stay black (the box spans atan(0.2 / 0.8) = 14 degrees of the 30-degree half view).
    func testEmissiveBoxAhead() throws {
        let renderer = try makeRenderer()
        let image = try renderer.render(camera: .eyeLeft, pose: try originPose,
                                        geoms: [try box(at: [0, 0, -1], half: 0.2, rgba: [1, 0, 0, 1])],
                                        appearances: [GeomAppearance(emission: 1, specular: 0, shininess: 0)])
        let centre = image.pixel(x: 128, y: 128)
        XCTAssertGreaterThan(centre.red, 240)
        XCTAssertLessThan(centre.green, 15)
        XCTAssertLessThan(centre.blue, 15)
        let corner = image.pixel(x: 0, y: 0)
        XCTAssertLessThan(Int(corner.red) + Int(corner.green) + Int(corner.blue), 15)
        XCTAssertEqual(renderer.lastReport?.drawnCount, 1)
    }

    /// Right is +x and up is +y of the camera; rows are top-down. A box at x = 0.4 m, 1 m ahead
    /// sits at column 128 + 128 x 0.4 / tan(30 degrees) = 217; a box at y = 0.4 m at row 39.
    func testImageOrientation() throws {
        let renderer = try makeRenderer()
        let look = [GeomAppearance(emission: 1, specular: 0, shininess: 0)]
        let right = try renderer.render(camera: .eyeLeft, pose: try originPose,
                                        geoms: [try box(at: [0.4, 0, -1], half: 0.1, rgba: [1, 0, 0, 1])],
                                        appearances: look)
        XCTAssertGreaterThan(right.pixel(x: 216, y: 128).red, 200)
        XCTAssertLessThan(right.pixel(x: 40, y: 128).red, 15)
        let up = try renderer.render(camera: .eyeLeft, pose: try originPose,
                                     geoms: [try box(at: [0, 0.4, -1], half: 0.1, rgba: [1, 0, 0, 1])],
                                     appearances: look)
        XCTAssertGreaterThan(up.pixel(x: 128, y: 39).red, 200)
        XCTAssertLessThan(up.pixel(x: 128, y: 216).red, 15)
    }

    /// A grey (0.5) box face toward the camera lit only by MuJoCo's default headlight gives
    /// 0.5 x 0.1 + 0.5 x 0.4 = 0.25 = 64 of 255 under the classic renderer's arithmetic.
    func testHeadlightShadingMatchesFixedFunctionValue() throws {
        let renderer = try makeRenderer(lights: [EyeLight.mujocoDefaultHeadlight])
        let image = try renderer.render(camera: .eyeLeft, pose: try originPose,
                                        geoms: [try box(at: [0, 0, -1], half: 0.2, rgba: [0.5, 0.5, 0.5, 1])],
                                        appearances: [GeomAppearance(emission: 0, specular: 0, shininess: 0.5)])
        let centre = image.pixel(x: 128, y: 128)
        print("headlight shading: centre pixel \(centre.red) \(centre.green) \(centre.blue); classic arithmetic 64")
        XCTAssertEqual(Double(centre.red), 64, accuracy: 6)
        XCTAssertEqual(Double(centre.green), 64, accuracy: 6)
        XCTAssertEqual(Double(centre.blue), 64, accuracy: 6)
    }

    /// A red emissive box 1 m ahead in front of a green emissive box 2 m ahead: the centre pixel
    /// is red. Under a depth buffer whose clear value disagreed with its direction every fragment
    /// would fail the depth test and the centre would stay black (the renderer's header,
    /// OPERATION step 4).
    func testNearBoxHidesFarBox() throws {
        let renderer = try makeRenderer(extent: 3.23)
        let look = [GeomAppearance(emission: 1, specular: 0, shininess: 0),
                    GeomAppearance(emission: 1, specular: 0, shininess: 0)]
        // The far box first in the list, so drawing order alone cannot put red in front.
        let image = try renderer.render(camera: .eyeLeft, pose: try originPose,
                                        geoms: [try box(at: [0, 0, -2], half: 0.4, rgba: [0, 1, 0, 1]),
                                                try box(at: [0, 0, -1], half: 0.1, rgba: [1, 0, 0, 1])],
                                        appearances: look)
        let centre = image.pixel(x: 128, y: 128)
        XCTAssertGreaterThan(centre.red, 240, "near red box must be drawn")
        XCTAssertLessThan(centre.green, 15, "far green box must be hidden behind it")
        // Off the red box (its front face at 0.9 m spans columns 128 +/- 128 x (0.1 / 0.9) /
        // tan(30 degrees) = 128 +/- 25) but on the green box's front face at 1.6 m, 0.25 m right:
        // column 128 + 128 x (0.25 / 1.6) / tan(30 degrees) = 163.
        let side = image.pixel(x: 163, y: 128)
        XCTAssertGreaterThan(side.green, 240)
        XCTAssertLessThan(side.red, 15)
    }

    /// The released crib's clipping distances in metres: znear 0.005 x 3.23 = 0.01615 m and
    /// zfar 30 x 3.23 = 96.9 m (scene.xml:9; extent from the exporter's abstract_scene.json), and
    /// its three lights: MuJoCo's default headlight, the light at (0, 0, 10) m pointing down
    /// without shadow and the light at (0, 0, 5) m aimed at the given point with a shadow.
    func testReleasedSettingsAndLightsWithoutGPU() throws {
        let settings = EyeRenderSettings.releasedCrib
        XCTAssertEqual(settings.znearMetres, 0.01615, accuracy: 1e-9)
        XCTAssertEqual(settings.zfarMetres, 96.9, accuracy: 1e-9)
        XCTAssertEqual(settings.supersampling, 2)
        XCTAssertEqual(settings.shadowMapSize, 2048)
        let pose = try originPose
        let lights = try FixedEyeLights.releasedCrib(upperBodyCentreOfMass: SIMD3<Float>(0, 3, 1))
            .lights(for: .eyeLeft, pose: pose)
        XCTAssertEqual(lights.count, 3)
        XCTAssertEqual(lights[0], EyeLight.mujocoDefaultHeadlight)
        XCTAssertFalse(lights[1].castsShadow)
        XCTAssertEqual(lights[1].position, SIMD3<Float>(0, 0, 10))
        XCTAssertTrue(lights[2].castsShadow)
        // Aim from (0, 0, 5) to (0, 3, 1): (0, 3, -4) / 5 = (0, 0.6, -0.8).
        XCTAssertLessThan(simd_length(lights[2].direction - SIMD3<Float>(0, 0.6, -0.8)), 1e-6)
        let straightDown = try FixedEyeLights.releasedCrib(upperBodyCentreOfMass: nil).lights(for: .eyeLeft, pose: pose)
        XCTAssertEqual(straightDown[2].direction, SIMD3<Float>(0, 0, -1))
    }

    /// WI-29's acceptance: the WI-14 states rendered and compared with the Python PNGs.
    func testFixtureStatesAgainstPythonImages() throws {
        guard MTLCreateSystemDefaultDevice() != nil else { throw XCTSkip("no Metal device on this machine") }
        let states = ["step_000492800", "step_000493674", "step_000493980"]
        var rows: [[String: Any]] = []
        var rendered = 0
        for state in states {
            let sub = "Fixtures/eyes/\(state)"
            guard let sceneURL = Bundle.module.url(forResource: "abstract_scene", withExtension: "json", subdirectory: sub),
                  let camerasURL = Bundle.module.url(forResource: "cameras", withExtension: "json", subdirectory: sub)
            else { continue }
            let abstract = try EyeFixtureScene(url: sceneURL)
            let cameraRows = try EyeFixtureCameras(url: camerasURL)
            let renderer = try SceneKitEyeRenderer(settings: EyeRenderSettings(extentMetres: abstract.extentMetres),
                                                   lights: abstract)
            for camera in EyeCamera.allCases {
                guard let pngURL = Bundle.module.url(forResource: camera.rawValue, withExtension: "png", subdirectory: sub) else {
                    XCTFail("\(state): \(camera.rawValue).png missing")
                    continue
                }
                let pose = try cameraRows.pose(camera)
                let image = try renderer.render(camera: camera, pose: pose, geoms: try abstract.geoms(for: camera),
                                                appearances: abstract.appearances)
                let reference = try eyeImageFromPNG(url: pngURL, camera: camera)
                XCTAssertEqual(reference.width, camera.pixelSize, "\(state) \(camera.rawValue): PNG size")
                let d = try image.difference(to: reference)
                rendered += 1
                let line = String(format: "%@ %@: mean abs per channel %.3f %.3f %.3f, max %ld %ld %ld, differing share %.4f, mean intensity swift %.2f vs python %.2f",
                                  state, camera.rawValue, d.meanAbsolute.x, d.meanAbsolute.y, d.meanAbsolute.z,
                                  d.maxAbsolute.x, d.maxAbsolute.y, d.maxAbsolute.z, d.differingPixelShare,
                                  image.meanIntensity, reference.meanIntensity)
                print(line)
                rows.append([
                    "state": state, "camera": camera.rawValue,
                    "mean_abs_per_channel": [d.meanAbsolute.x, d.meanAbsolute.y, d.meanAbsolute.z],
                    "max_abs_per_channel": [d.maxAbsolute.x, d.maxAbsolute.y, d.maxAbsolute.z],
                    "differing_pixel_share": d.differingPixelShare,
                    "mean_intensity_swift": image.meanIntensity,
                    "mean_intensity_python": reference.meanIntensity,
                    "geoms_drawn": renderer.lastReport?.drawnCount ?? -1,
                    "geoms_skipped": renderer.lastReport?.skippedCount ?? -1,
                ])
            }
        }
        if rendered == 0 {
            throw XCTSkip("no Fixtures/eyes/<state>/abstract_scene.json: run Tests/UEMASensesTests/Fixtures/eyes/export_abstract_scene.py (its docstring) and rebuild")
        }
        let doc: [String: Any] = [
            "measurement": "mean absolute difference per channel, Swift SceneKit render against the Python classic-renderer PNG, 8-bit steps (0 = identical, 255 = every pixel opposite)",
            "rows": rows,
        ]
        let out = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("uema_eye_differences.json")
        let data = try JSONSerialization.data(withJSONObject: doc, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: out)
        print("eye render differences written to \(out.path) (\(rendered) images)")
    }
}

// MARK: - Fixture readers

/// Why a fixture file could not be read.
private enum EyeFixtureError: Error, CustomStringConvertible {
    case missing(String)
    case unreadablePNG(String)

    var description: String {
        switch self {
        case let .missing(what): return "eye fixture: \(what) missing or of the wrong type"
        case let .unreadablePNG(path): return "eye fixture: cannot decode \(path)"
        }
    }
}

private func numbers(_ value: Any?, _ what: String) throws -> [Double] {
    guard let list = value as? [Any] else { throw EyeFixtureError.missing(what) }
    return try list.map { (item: Any) throws -> Double in
        guard let n = item as? NSNumber else { throw EyeFixtureError.missing(what) }
        return n.doubleValue
    }
}

private func vector3(_ value: Any?, _ what: String) throws -> SIMD3<Float> {
    let v = try numbers(value, what)
    guard v.count == 3 else { throw EyeFixtureError.missing(what) }
    return SIMD3(Float(v[0]), Float(v[1]), Float(v[2]))
}

/// abstract_scene.json of one state (export_abstract_scene.py, format
/// "uema.eye_fixture.abstract_scene.v1"): the eye_left geom list, the per-camera plane
/// positions, the per-camera mjvLight rows, and the model extent. It is also the light source.
private struct EyeFixtureScene: EyeLightSource {
    let extentMetres: Double
    let baseGeoms: [SceneGeom]
    let appearances: [GeomAppearance]
    let planePositions: [EyeCamera: [Int: [Float]]]
    let lightsPerCamera: [EyeCamera: [EyeLight]]

    init(url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let doc = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw EyeFixtureError.missing("abstract_scene.json object")
        }
        guard let model = doc["model"] as? [String: Any], let stat = model["stat"] as? [String: Any],
              let extent = (stat["extent_m"] as? NSNumber)?.doubleValue else {
            throw EyeFixtureError.missing("model.stat.extent_m")
        }
        extentMetres = extent
        guard let geomRows = doc["geoms"] as? [[String: Any]] else { throw EyeFixtureError.missing("geoms") }
        var geoms: [SceneGeom] = []
        var looks: [GeomAppearance] = []
        for row in geomRows {
            guard let type = (row["type"] as? NSNumber)?.intValue else { throw EyeFixtureError.missing("geom type") }
            geoms.append(try SceneGeom(type: type,
                                       size: try numbers(row["size"], "geom size").map { Float($0) },
                                       position: try numbers(row["pos"], "geom pos").map { Float($0) },
                                       rotation: try numbers(row["mat"], "geom mat").map { Float($0) },
                                       rgba: try numbers(row["rgba"], "geom rgba").map { Float($0) }))
            looks.append(GeomAppearance(emission: (row["emission"] as? NSNumber)?.floatValue ?? 0,
                                        specular: (row["specular"] as? NSNumber)?.floatValue ?? 0.5,
                                        shininess: (row["shininess"] as? NSNumber)?.floatValue ?? 0.5,
                                        reflectance: (row["reflectance"] as? NSNumber)?.floatValue ?? 0))
        }
        baseGeoms = geoms
        appearances = looks
        guard let cameras = doc["cameras"] as? [String: Any] else { throw EyeFixtureError.missing("cameras") }
        var planes: [EyeCamera: [Int: [Float]]] = [:]
        var lights: [EyeCamera: [EyeLight]] = [:]
        for camera in EyeCamera.allCases {
            guard let entry = cameras[camera.rawValue] as? [String: Any] else {
                throw EyeFixtureError.missing("cameras.\(camera.rawValue)")
            }
            var moved: [Int: [Float]] = [:]
            for diff in (entry["geom_differences"] as? [[String: Any]]) ?? [] {
                guard let index = (diff["index"] as? NSNumber)?.intValue else { continue }
                moved[index] = try numbers(diff["pos"], "geom_differences pos").map { Float($0) }
            }
            planes[camera] = moved
            var list: [EyeLight] = []
            for row in (entry["lights"] as? [[String: Any]]) ?? [] {
                list.append(EyeLight(
                    directional: ((row["directional"] as? NSNumber)?.intValue ?? 0) != 0,
                    headlight: ((row["headlight"] as? NSNumber)?.intValue ?? 0) != 0,
                    position: try vector3(row["pos"], "light pos"),
                    direction: try vector3(row["dir"], "light dir"),
                    ambient: try vector3(row["ambient"], "light ambient"),
                    diffuse: try vector3(row["diffuse"], "light diffuse"),
                    specular: try vector3(row["specular"], "light specular"),
                    castsShadow: ((row["castshadow"] as? NSNumber)?.intValue ?? 0) != 0,
                    cutoffDegrees: (row["cutoff"] as? NSNumber)?.floatValue ?? 45))
            }
            lights[camera] = list
        }
        planePositions = planes
        lightsPerCamera = lights
    }

    /// The geom list as mjv_updateScene gave it for `camera` (planes recentred per camera).
    func geoms(for camera: EyeCamera) throws -> [SceneGeom] {
        guard let moved = planePositions[camera], !moved.isEmpty else { return baseGeoms }
        var out: [SceneGeom] = []
        out.reserveCapacity(baseGeoms.count)
        for (index, g) in baseGeoms.enumerated() {
            if let pos = moved[index] {
                out.append(try SceneGeom(type: g.type, size: g.size, position: pos, rotation: g.rotation, rgba: g.rgba))
            } else {
                out.append(g)
            }
        }
        return out
    }

    func lights(for camera: EyeCamera, pose: CameraPose) throws -> [EyeLight] {
        lightsPerCamera[camera] ?? []
    }
}

/// cameras.json of one state (export_eye_fixtures.py): each camera's recorded world pose.
private struct EyeFixtureCameras {
    let poses: [String: CameraPose]

    init(url: URL) throws {
        let data = try Data(contentsOf: url)
        guard let doc = try JSONSerialization.jsonObject(with: data) as? [String: Any],
              let rows = doc["cameras"] as? [[String: Any]] else {
            throw EyeFixtureError.missing("cameras.json cameras")
        }
        var poses: [String: CameraPose] = [:]
        for row in rows {
            guard let name = row["name"] as? String else { throw EyeFixtureError.missing("camera name") }
            poses[name] = try CameraPose(position: try numbers(row["pos_m"], "pos_m"),
                                         rotation: try numbers(row["xmat_row_major"], "xmat_row_major"))
        }
        self.poses = poses
    }

    func pose(_ camera: EyeCamera) throws -> CameraPose {
        guard let pose = poses[camera.rawValue] else { throw EyeFixtureError.missing("pose of \(camera.rawValue)") }
        return pose
    }
}

/// The PNG's RGB bytes, rows top-down, read without colour conversion: the decoded bytes when
/// the image is 8 bits per channel RGB or RGBA, else a draw into a context of the image's own
/// colour space.
private func eyeImageFromPNG(url: URL, camera: EyeCamera) throws -> EyeImage {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
        throw EyeFixtureError.unreadablePNG(url.path)
    }
    let width = image.width
    let height = image.height
    var rgb = [UInt8](repeating: 0, count: width * height * 3)
    let alpha = image.alphaInfo
    let alphaFirst = alpha == .first || alpha == .premultipliedFirst || alpha == .noneSkipFirst
    let littleEndian = image.bitmapInfo.contains(.byteOrder32Little)
    if image.bitsPerComponent == 8, image.bitsPerPixel == 24 || image.bitsPerPixel == 32, !littleEndian,
       let data = image.dataProvider?.data, let base = CFDataGetBytePtr(data) {
        let step = image.bitsPerPixel / 8
        let offset = (step == 4 && alphaFirst) ? 1 : 0
        for y in 0 ..< height {
            for x in 0 ..< width {
                let s = y * image.bytesPerRow + x * step + offset
                let o = (y * width + x) * 3
                rgb[o] = base[s]
                rgb[o + 1] = base[s + 1]
                rgb[o + 2] = base[s + 2]
            }
        }
    } else {
        let space = image.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let drawn: Bool = rgba.withUnsafeMutableBytes { (buffer: UnsafeMutableRawBufferPointer) -> Bool in
            guard let context = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8,
                                          bytesPerRow: width * 4, space: space,
                                          bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw EyeFixtureError.unreadablePNG(url.path) }
        for i in 0 ..< width * height {
            rgb[i * 3] = rgba[i * 4]
            rgb[i * 3 + 1] = rgba[i * 4 + 1]
            rgb[i * 3 + 2] = rgba[i * 4 + 2]
        }
    }
    return try EyeImage(camera: camera, width: width, height: height, rgb: rgb)
}
