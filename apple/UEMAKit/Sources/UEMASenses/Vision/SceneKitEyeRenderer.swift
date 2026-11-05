// SceneKitEyeRenderer.swift: the eye renderer of the Apple port: the geom list mirrored as
// SceneKit nodes (GeomMirror.swift), one SceneKit camera at the MuJoCo camera's pose and field of
// view, MuJoCo's lights, and an SCNRenderer on the default Metal device drawing offscreen into a
// Metal texture that is read back as RGB 8-bit (work item WI-29 of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html; specification REQ-B-6). Nothing in
// this file was compiled on the machine that wrote it (no Swift toolchain; plan section 3, fact 1).
//
// WHAT THIS FILE IS. SceneKitEyeRenderer implements UEMACore.EyeRendering. EyeRenderSettings holds
// the clipping distances, the supersampling factor, the background and the shadow map size;
// EyeLight, EyeLightSource and FixedEyeLights describe and supply the lights (UEMACore's
// SceneGeomReading carries no lights, so the lights are injected beside it); eyeDownsample
// averages the supersampled texture down to the camera's size. The classic renderer (mjr_, OpenGL
// 1.5) is never called: iOS has no desktop OpenGL and MuJoCo's Filament renderer declares no Metal
// value (plan row WI-29 evidence).
//
// INPUT. render(camera:pose:geoms:): the camera (UEMACore.EyeCamera: 256 px over 60 degrees for
// eye_left and eye_right, 288 px over 24 degrees for fovea_left and fovea_right, 256 px over 150
// degrees for ring_left), its world pose (CameraPose: position 3 numbers in metres; rotation 9
// numbers row-major whose columns are the camera's x (right), y (up) and z axes; the camera looks
// along its -z, as MuJoCo's mjv_cameraInModel reads it: forward = -column 2, up = column 1), and
// the geom list mjv_updateScene filled (SceneGeom; GeomMirror.swift gives each field's meaning).
// The Python reference renders through env.py:625-638: mjv_updateScene(model, data, MjvOption(),
// None, fixed camera, mjCAT_ALL, scene); mjr_render; mjr_readPixels; np.flipud.
//
// OPERATION, per render:
//  1. GeomMirror.update(geoms) sets one node per geom (GeomMirror.swift, OPERATION).
//  2. The camera node's simdTransform = T(position) x R(rotation) (eyeWorldTransform). Its
//     SCNCamera: fieldOfView = fovy degrees, projectionDirection vertical, zNear = znear x extent
//     and zFar = zfar x extent in metres (scene.xml:9 `<map ... zfar="30" znear="0.005" />`;
//     MuJoCo's map znear and zfar are fractions of the model's stat.extent, which the settings
//     take as `extentMetres`), automaticallyAdjustsZRange off.
//  3. Lights from the injected EyeLightSource: MuJoCo's headlight (a directional light at the
//     camera along its forward axis; MuJoCo's defaults ambient 0.1, diffuse 0.4, specular 0.5,
//     active 1, since scene.xml sets no <headlight>) and the scene's two lights (scene.xml:38-39:
//     a light at (0, 0, 10) m pointing down, diffuse 0.4, specular 0, no shadow; a light at
//     (0, 0, 5) m aimed at upper_body's centre of mass, diffuse 0.8, specular 0.3, casting a
//     shadow). MuJoCo's non-directional lights are spot lights with cutoff 45 degrees (half
//     angle) and exponent 10 (MuJoCo light defaults); SceneKit's .spot light takes the full cone
//     angle: spotOuterAngle 90 degrees, spotInnerAngle 0, a linear fall-off between them instead
//     of MuJoCo's cos^10. SceneKit lights have one colour for diffuse and specular; the light's
//     diffuse colour is used. All ambient parts are summed into one .ambient light.
//  4. SCNRenderer(device:options:) renders at time 0 into a .rgba8Unorm texture of
//     (size x supersampling) pixels square with a .depth32Float depth texture, through
//     render(atTime:viewport:commandBuffer:passDescriptor:); a blit copies the texture into a
//     shared MTLBuffer; the command buffer is committed and waited on. The renderer's
//     usesReverseZ is set to false and the depth attachment is cleared to 1 (the far end of a
//     forward depth buffer): SceneKit's reverse depth buffer stores the near end as 1 and passes
//     a fragment only when its depth is greater than the stored one, so a clear to 1 under
//     reverse depth would reject every fragment and leave the image at the background. The
//     clear value is read from usesReverseZ at each render, so the two cannot disagree.
//  5. eyeDownsample averages each supersampling x supersampling block of the RGBA bytes into one
//     RGB pixel, rounding half up: supersampling 2 gives 4 samples per pixel, the count of MuJoCo's
//     default offscreen multisampling (vis.quality offsamples default 4; the scene does not set
//     it). Metal's texture row 0 is the top row, so the bytes come out rows top-down, columns
//     left-to-right, the fixture PNGs' layout, with no flip (the Python flips because
//     mjr_readPixels gives bottom-up rows, env.py:638).
//
// WHY LINEAR COLOURS AND A NON-sRGB TARGET. MuJoCo's classic renderer computes fixed-function
// lighting on the rgba numbers as given and writes the result to an 8-bit buffer with no gamma
// step. SceneKit lights in linear space. With every colour built in the linear sRGB colour space
// (GeomMirror.swift LinearColor) SceneKit converts no input, and a .rgba8Unorm (not _srgb) target
// stores the shader's output with no encoding, so the arithmetic is the classic renderer's.
//
// WHAT IS NOT REPRODUCED, each one measured by the fixture exporter's ablations
// (export_abstract_scene.py "feature_ablations": the mean absolute difference per channel of the
// classic render with that one feature off against the full render, for eye_left and ring_left
// of each state): textures (play mat, pen walls, ceiling; UEMACore.SceneGeom carries no material
// or texture id), floor reflections (play mat reflectance 0.05, scene.xml:17), the gradient
// skybox (scene.xml:28) and haze (scene.xml:10). The fixture test prints the measured difference;
// the pass line is set from it in the plan's section 9.
// Measured by that exporter on November 4, 2026 (mujoco 3.3.0, the three WI-14 states; numbers
// from its printed output, also in each abstract_scene.json "feature_ablations"), as mean absolute
// difference per channel red, green, blue in 8-bit steps (0 = identical, 255 = opposite):
//   textures off     eye_left 24.3 to 34.3 red, 20.0 to 21.9 green, 15.1 to 16.8 blue;
//                    ring_left 25.6 to 34.0 red, 20.2 to 25.6 green, 17.4 to 25.7 blue
//   skybox off       eye_left 0.0 to 5.0; ring_left 2.8 to 24.2 (largest in blue)
//   shadows off      eye_left 0.0 to 0.6; ring_left 5.2 to 10.4
//   reflections off  0.0 to 0.24; haze off 0.0; back-face culling off 0.0
//   headlight off    eye_left 43.1 to 59.8; ring_left 36.0 to 55.7 (reproduced here)
// So the missing textures are the largest expected part of the Swift render's difference (up to
// 34.3 of 255 in red on the wide eye), then the skybox on the ring camera (up to 24.2 in blue).
// The same run re-rendered each WI-14 state with the classic renderer from the shipped bundle and
// compared it with WI-14's PNG: 0.000 to 0.029 per channel on step_000492800's eye and fovea
// cameras, but up to 5.95 (eye_left) and 29.9 (ring_left) on step_000493674 and up to 12.4
// (ring_left) on step_000493980: for those two states the bundle world set from cameras.json is
// not pixel-identical to the world WI-14 rendered, so their Swift differences carry that part too
// (step_000492800 is the clean reference).
//
// OUTPUT. UEMACore.EyeImage of camera.pixelSize x camera.pixelSize x 3 bytes; `lastReport`, the
// GeomMirror report of the last render.
//
// FIRST BUILD ON A MAC (WI-29's acceptance: "renders the WI-14 states; the mean absolute
// difference per channel against the Python PNG (0 = identical, 255 = every pixel opposite) is
// recorded, and the pass line is set from it in section 9 before any planner result is judged").
//  1. Run WI-17's steps 1 to 3 (MuJoCo XCFramework in place, `swift package resolve` exit 0).
//  2. On the Windows machine (or any machine with mujoco 3.3.0 for Python): run
//     Tests/UEMASensesTests/Fixtures/eyes/export_abstract_scene.py as its docstring says; it
//     writes Fixtures/eyes/<state>/abstract_scene.json for the three WI-14 states (the geom list,
//     lights, extent and each geom's emission, specular, shininess) and exits 0 only when the
//     classic renderer reproduces every fixture PNG exactly. Commit those files with the
//     integration commit. Without them the fixture test skips and names this step.
//  3. `cd apple/UEMAKit && xcodebuild -scheme UEMAKit-Package test -destination
//     'platform=macOS' -only-testing:UEMASensesTests/SceneKitEyeRendererTests` : exit 0. The
//     synthetic tests check size, layout and colour; the fixture test prints one line per state
//     and camera: "<state> <camera>: mean abs per channel r g b, max r g b, differing share s,
//     mean intensity swift vs python" and writes them to
//     NSTemporaryDirectory()/uema_eye_differences.json. Copy that file to
//     apple/records/eye_render_difference.json and the largest channel means into section 9.
//  4. If the images are uniformly darker or brighter than the fixture PNGs (mean intensity
//     printed beside the fixture's), the colour-space step above is wrong for this SDK: try a
//     .bgra8Unorm_srgb target, record both numbers, keep the smaller difference.
//  5. If SceneKit draws nothing (every pixel the background), check first that
//     `renderer.usesReverseZ` reads false after init and the depth clear is 1 (OPERATION step
//     4), then that the depth attachment format is accepted (try .depth32Float_stencil8) and
//     that `pointOfView` is set.
//  6. If the shading differs most on specular highlights, SCNMaterial.shininess may not be the
//     raw exponent: retry with shininess x 1 instead of x 128 (GeomMirror.apply) and record both.
//  7. SceneKit is marked deprecated in the 26.0 SDKs (developer.apple.com, SCNRenderer, read
//     November 4, 2026): warnings are expected. Record the command, exit code, warnings count and
//     clock time (AM/PM Pacific) in apple/FIRST_BUILD_RECORD.md.
//
// SOURCES READ (November 4, 2026):
// - https://developer.apple.com/tutorials/data/documentation/scenekit/scnrenderer.json (fetched
//   November 4, 2026): "convenience init(device: (any MTLDevice)?, options: [AnyHashable : Any]?)";
//   "func render(atTime: CFTimeInterval, viewport: CGRect, commandBuffer: any MTLCommandBuffer,
//   passDescriptor: MTLRenderPassDescriptor)"; "var scene: SCNScene?"; "Deprecated: 26.0".
//   `pointOfView` and `autoenablesDefaultLighting` belong to the SCNSceneRenderer protocol that
//   SCNRenderer conforms to (same page: "Conforms to ... SCNSceneRenderer").
// - https://developer.apple.com/tutorials/data/documentation/scenekit/scnscenerenderer/usesreversez.json
//   (fetched November 4, 2026): "var usesReverseZ: Bool { get set }"; the page states no
//   default value, so init sets it explicitly.
// - https://developer.apple.com/tutorials/data/documentation/scenekit/scnmaterial/shininess.json
//   (fetched November 4, 2026): "var shininess: CGFloat { get set }", "The sharpness of
//   specular highlights"; the page does not say how the value maps to a specular exponent,
//   hence checklist step 6.
// - Source/Python/mimo_bridge/env.py:590-638 (the shared GL context and EyeRenderer).
// - The bundle's scene.xml:8-39 (visual block, materials, lights).
// - UEMACore/Protocols.swift (EyeRendering, EyeCamera, CameraPose, SceneGeom, EyeImage).

import CoreGraphics
import Foundation
import Metal
import SceneKit
import simd
import UEMACore

// MARK: - Settings

/// The numbers of one eye render that are not in the geom list: clipping distances, sampling,
/// background and shadow map size.
public struct EyeRenderSettings: Equatable, Sendable {
    /// The model's stat.extent in metres (MuJoCo computes it at compile time; the fixture
    /// exporter writes it as abstract_scene.json model.stat.extent_m).
    public let extentMetres: Double
    /// Near clipping distance as a fraction of the extent: 0.005 (scene.xml:9).
    public let znearFraction: Double
    /// Far clipping distance as a fraction of the extent: 30 (scene.xml:9).
    public let zfarFraction: Double
    /// Render at this many times the camera's size per side, then average: 2 gives 4 samples
    /// per pixel (MuJoCo's default offsamples 4).
    public let supersampling: Int
    /// The colour where no geom is drawn, linear 0 to 1 per channel (MuJoCo clears to black).
    public let background: SIMD3<Float>
    /// Shadow map size in pixels per side: 2048 (scene.xml:11 `<quality shadowsize="2048" />`).
    public let shadowMapSize: Int

    /// Settings; `extentMetres` above 0, `supersampling` 1 to 4.
    public init(extentMetres: Double, znearFraction: Double = 0.005, zfarFraction: Double = 30,
                supersampling: Int = 2, background: SIMD3<Float> = SIMD3(0, 0, 0), shadowMapSize: Int = 2048) {
        precondition(extentMetres > 0, "extent must be above 0 m")
        precondition(znearFraction > 0 && zfarFraction > znearFraction, "need 0 < znear < zfar")
        precondition(supersampling >= 1 && supersampling <= 4, "supersampling \(supersampling) outside 1 ... 4")
        self.extentMetres = extentMetres
        self.znearFraction = znearFraction
        self.zfarFraction = zfarFraction
        self.supersampling = supersampling
        self.background = background
        self.shadowMapSize = shadowMapSize
    }

    /// The released crib: extent 3.23 m (mujoco 3.3.0 compiling the shipped scene.xml, printed by
    /// export_abstract_scene.py on November 4, 2026: "extent 3.2300 m"; abstract_scene.json
    /// model.stat.extent_m), so znear 0.016 m and zfar 96.9 m.
    public static let releasedCrib = EyeRenderSettings(extentMetres: 3.23)

    /// Near clipping distance in metres.
    public var znearMetres: Double { znearFraction * extentMetres }
    /// Far clipping distance in metres.
    public var zfarMetres: Double { zfarFraction * extentMetres }
}

// MARK: - Lights

/// One light as MuJoCo's mjvLight describes it, in world coordinates (metres) and linear colour
/// (0 to 1 per channel).
public struct EyeLight: Equatable, Sendable {
    /// True for a directional light (no position fall-off, MuJoCo `directional`).
    public let directional: Bool
    /// True for MuJoCo's headlight, which sits at the camera and points along its forward axis;
    /// its position and direction are ignored and taken from the camera.
    public let headlight: Bool
    /// Position in metres, world frame.
    public let position: SIMD3<Float>
    /// Direction the light points along, world frame (normalised at use).
    public let direction: SIMD3<Float>
    /// Ambient colour.
    public let ambient: SIMD3<Float>
    /// Diffuse colour (SceneKit's one light colour).
    public let diffuse: SIMD3<Float>
    /// Specular colour (recorded; SceneKit has no separate specular light colour).
    public let specular: SIMD3<Float>
    /// True when the light casts shadows.
    public let castsShadow: Bool
    /// Spot cut-off half angle in degrees (MuJoCo `cutoff`, default 45).
    public let cutoffDegrees: Float

    /// A light.
    public init(directional: Bool, headlight: Bool = false, position: SIMD3<Float>, direction: SIMD3<Float>,
                ambient: SIMD3<Float> = .zero, diffuse: SIMD3<Float>, specular: SIMD3<Float>,
                castsShadow: Bool, cutoffDegrees: Float = 45) {
        self.directional = directional
        self.headlight = headlight
        self.position = position
        self.direction = direction
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.castsShadow = castsShadow
        self.cutoffDegrees = cutoffDegrees
    }

    /// MuJoCo's default headlight (visual/headlight defaults: ambient 0.1, diffuse 0.4,
    /// specular 0.5; scene.xml sets none).
    public static let mujocoDefaultHeadlight = EyeLight(
        directional: true, headlight: true, position: .zero, direction: SIMD3(0, 0, -1),
        ambient: SIMD3(repeating: 0.1), diffuse: SIMD3(repeating: 0.4), specular: SIMD3(repeating: 0.5),
        castsShadow: false)
}

/// Supplies the lights of one render. Injected into the renderer, because the geom reading
/// (UEMACore.SceneGeomReading) carries no lights.
public protocol EyeLightSource {
    /// The lights for `camera` at `pose`.
    func lights(for camera: EyeCamera, pose: CameraPose) throws -> [EyeLight]
}

/// A fixed list of lights, the same for every camera.
public struct FixedEyeLights: EyeLightSource, Equatable, Sendable {
    /// The lights.
    public let list: [EyeLight]

    /// A fixed list.
    public init(_ list: [EyeLight]) {
        self.list = list
    }

    public func lights(for camera: EyeCamera, pose: CameraPose) throws -> [EyeLight] { list }

    /// The released crib's lights (scene.xml:38-39) plus MuJoCo's default headlight. The second
    /// scene light aims at upper_body's centre of mass, which moves with the body: pass it in
    /// metres (world frame), or nil to aim straight down (0, 0, -1).
    public static func releasedCrib(upperBodyCentreOfMass: SIMD3<Float>?) -> FixedEyeLights {
        let topPosition = SIMD3<Float>(0, 0, 5)
        var aim = SIMD3<Float>(0, 0, -1)
        if let target = upperBodyCentreOfMass {
            let d = target - topPosition
            if simd_length(d) > 1e-6 { aim = simd_normalize(d) }
        }
        return FixedEyeLights([
            EyeLight.mujocoDefaultHeadlight,
            EyeLight(directional: false, position: SIMD3(0, 0, 10), direction: SIMD3(0, 0, -1),
                     diffuse: SIMD3(repeating: 0.4), specular: SIMD3(repeating: 0), castsShadow: false),
            EyeLight(directional: false, position: topPosition, direction: aim,
                     diffuse: SIMD3(repeating: 0.8), specular: SIMD3(repeating: 0.3), castsShadow: true),
        ])
    }
}

/// A rotation whose local -z points along `direction` (SceneKit lights and cameras point along
/// their local -z).
func eyeOrientation(pointingAlong direction: SIMD3<Float>) -> simd_float4x4 {
    let length = simd_length(direction)
    let forward = length > 1e-9 ? direction / length : SIMD3<Float>(0, 0, -1)
    let z = -forward
    let up: SIMD3<Float> = abs(z.z) < 0.99 ? SIMD3(0, 0, 1) : SIMD3(0, 1, 0)
    let x = simd_normalize(simd_cross(up, z))
    let y = simd_cross(z, x)
    return simd_float4x4(columns: (
        SIMD4<Float>(x.x, x.y, x.z, 0), SIMD4<Float>(y.x, y.y, y.z, 0), SIMD4<Float>(z.x, z.y, z.z, 0), SIMD4<Float>(0, 0, 0, 1)))
}

// MARK: - Errors

/// Why an eye render failed.
public enum EyeRenderError: Error, Equatable, CustomStringConvertible {
    /// MTLCreateSystemDefaultDevice() returned nil.
    case noMetalDevice
    /// The Metal device could not make the named object.
    case metalAllocationFailed(String)
    /// The command buffer ended with an error.
    case commandBufferFailed(String)

    public var description: String {
        switch self {
        case .noMetalDevice:
            return "eye renderer: this machine has no Metal device"
        case let .metalAllocationFailed(what):
            return "eye renderer: Metal could not make \(what)"
        case let .commandBufferFailed(message):
            return "eye renderer: the command buffer failed: \(message)"
        }
    }
}

// MARK: - Downsampling

/// Averages each `factor` x `factor` block of an RGBA image (`width` x `width` pixels, 4 bytes
/// per pixel, rows top-down) into one RGB pixel, rounding half up; the result is
/// (width / factor) squared pixels x 3 bytes, rows top-down. `width` must be a multiple of
/// `factor`.
public func eyeDownsample(rgba: UnsafeBufferPointer<UInt8>, width: Int, factor: Int) -> [UInt8] {
    precondition(factor >= 1 && width % factor == 0, "width \(width) is not a multiple of \(factor)")
    precondition(rgba.count >= width * width * 4, "RGBA buffer holds \(rgba.count) bytes, need \(width * width * 4)")
    let outWidth = width / factor
    let samples = factor * factor
    var out = [UInt8](repeating: 0, count: outWidth * outWidth * 3)
    for oy in 0 ..< outWidth {
        for ox in 0 ..< outWidth {
            var r = 0, g = 0, b = 0
            for sy in 0 ..< factor {
                let row = (oy * factor + sy) * width
                for sx in 0 ..< factor {
                    let i = (row + ox * factor + sx) * 4
                    r += Int(rgba[i])
                    g += Int(rgba[i + 1])
                    b += Int(rgba[i + 2])
                }
            }
            let o = (oy * outWidth + ox) * 3
            out[o] = UInt8((r + samples / 2) / samples)
            out[o + 1] = UInt8((g + samples / 2) / samples)
            out[o + 2] = UInt8((b + samples / 2) / samples)
        }
    }
    return out
}

// MARK: - The renderer

/// The Apple port's eye renderer: SceneKit on the default Metal device, offscreen, RGB 8-bit
/// (this file's header gives input, operation and output). One instance keeps its scene, nodes
/// and render targets between renders; it is not safe to call from two threads at once.
public final class SceneKitEyeRenderer: EyeRendering {
    /// The settings every render uses.
    public let settings: EyeRenderSettings
    /// The GeomMirror report of the last render; nil before the first.
    public private(set) var lastReport: GeomMirrorReport?

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let renderer: SCNRenderer
    private let scene = SCNScene()
    private let mirror: GeomMirror
    private let cameraNode = SCNNode()
    private let lightRoot = SCNNode()
    private let lightSource: any EyeLightSource
    private var targets: [Int: (color: MTLTexture, depth: MTLTexture, readback: MTLBuffer)] = [:]

    /// A renderer on `device` with `settings`, lit by `lights`.
    /// - Throws: `EyeRenderError.metalAllocationFailed` when the device makes no command queue.
    public init(device: MTLDevice, settings: EyeRenderSettings, lights: any EyeLightSource) throws {
        guard let queue = device.makeCommandQueue() else {
            throw EyeRenderError.metalAllocationFailed("a command queue")
        }
        self.device = device
        self.queue = queue
        self.settings = settings
        self.lightSource = lights
        self.mirror = GeomMirror(infinitePlaneHalfExtent: Float(settings.zfarMetres))
        self.renderer = SCNRenderer(device: device, options: nil)

        let camera = SCNCamera()
        camera.projectionDirection = .vertical
        camera.automaticallyAdjustsZRange = false
        camera.zNear = settings.znearMetres
        camera.zFar = settings.zfarMetres
        camera.wantsHDR = false
        cameraNode.camera = camera
        cameraNode.name = "eye_camera"

        scene.background.contents = LinearColor.make(settings.background.x, settings.background.y, settings.background.z)
        scene.rootNode.addChildNode(mirror.rootNode)
        scene.rootNode.addChildNode(cameraNode)
        lightRoot.name = "eye_lights"
        scene.rootNode.addChildNode(lightRoot)

        renderer.scene = scene
        renderer.pointOfView = cameraNode
        renderer.autoenablesDefaultLighting = false
        // A forward depth buffer, matched by the depth clear in render(...) (header, step 4).
        renderer.usesReverseZ = false
    }

    /// A renderer on the system's default Metal device.
    /// - Throws: `EyeRenderError.noMetalDevice` when the machine has none.
    public convenience init(settings: EyeRenderSettings, lights: any EyeLightSource) throws {
        guard let device = MTLCreateSystemDefaultDevice() else { throw EyeRenderError.noMetalDevice }
        try self.init(device: device, settings: settings, lights: lights)
    }

    /// The image of `camera` with MuJoCo's default material numbers for every geom.
    public func render(camera: EyeCamera, pose: CameraPose, geoms: [SceneGeom]) throws -> EyeImage {
        try render(camera: camera, pose: pose, geoms: geoms, appearances: nil)
    }

    /// The image of `camera`, with one `GeomAppearance` per geom when given.
    public func render(camera: EyeCamera, pose: CameraPose, geoms: [SceneGeom],
                       appearances: [GeomAppearance]?) throws -> EyeImage {
        lastReport = mirror.update(geoms: geoms, appearances: appearances)

        let position = pose.position.map { Float($0) }
        let rotation = pose.rotation.map { Float($0) }
        cameraNode.simdTransform = eyeWorldTransform(position: position, rotation: rotation)
        cameraNode.camera?.fieldOfView = CGFloat(camera.fieldOfViewDegrees)

        try placeLights(try lightSource.lights(for: camera, pose: pose), cameraPosition: position, cameraRotation: rotation)

        let size = camera.pixelSize
        let width = size * settings.supersampling
        let target = try renderTarget(width: width)

        let pass = MTLRenderPassDescriptor()
        pass.colorAttachments[0].texture = target.color
        pass.colorAttachments[0].loadAction = .clear
        pass.colorAttachments[0].storeAction = .store
        pass.colorAttachments[0].clearColor = MTLClearColor(
            red: Double(settings.background.x), green: Double(settings.background.y),
            blue: Double(settings.background.z), alpha: 1)
        pass.depthAttachment.texture = target.depth
        pass.depthAttachment.loadAction = .clear
        pass.depthAttachment.storeAction = .dontCare
        // The far end of the depth buffer: 1 for a forward buffer, 0 for SceneKit's reverse one.
        pass.depthAttachment.clearDepth = renderer.usesReverseZ ? 0 : 1

        guard let commandBuffer = queue.makeCommandBuffer() else {
            throw EyeRenderError.metalAllocationFailed("a command buffer")
        }
        renderer.render(atTime: 0, viewport: CGRect(x: 0, y: 0, width: width, height: width),
                        commandBuffer: commandBuffer, passDescriptor: pass)
        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw EyeRenderError.metalAllocationFailed("a blit command encoder")
        }
        let bytesPerRow = width * 4
        blit.copy(from: target.color, sourceSlice: 0, sourceLevel: 0,
                  sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                  sourceSize: MTLSize(width: width, height: width, depth: 1),
                  to: target.readback, destinationOffset: 0,
                  destinationBytesPerRow: bytesPerRow, destinationBytesPerImage: bytesPerRow * width)
        blit.endEncoding()
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw EyeRenderError.commandBufferFailed(String(describing: error))
        }

        let byteCount = bytesPerRow * width
        let pointer = target.readback.contents().bindMemory(to: UInt8.self, capacity: byteCount)
        let rgb = eyeDownsample(rgba: UnsafeBufferPointer(start: pointer, count: byteCount),
                                width: width, factor: settings.supersampling)
        return try EyeImage(camera: camera, width: size, height: size, rgb: rgb)
    }

    /// The colour texture, depth texture and read-back buffer for one square width, made once.
    private func renderTarget(width: Int) throws -> (color: MTLTexture, depth: MTLTexture, readback: MTLBuffer) {
        if let existing = targets[width] { return existing }
        let colorDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: width, height: width, mipmapped: false)
        colorDescriptor.usage = [.renderTarget, .shaderRead]
        colorDescriptor.storageMode = .private
        guard let color = device.makeTexture(descriptor: colorDescriptor) else {
            throw EyeRenderError.metalAllocationFailed("a \(width) x \(width) colour texture")
        }
        let depthDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .depth32Float, width: width, height: width, mipmapped: false)
        depthDescriptor.usage = [.renderTarget]
        depthDescriptor.storageMode = .private
        guard let depth = device.makeTexture(descriptor: depthDescriptor) else {
            throw EyeRenderError.metalAllocationFailed("a \(width) x \(width) depth texture")
        }
        guard let readback = device.makeBuffer(length: width * width * 4, options: .storageModeShared) else {
            throw EyeRenderError.metalAllocationFailed("a \(width * width * 4)-byte read-back buffer")
        }
        let made = (color: color, depth: depth, readback: readback)
        targets[width] = made
        return made
    }

    /// Replaces the light nodes with `lights` (this file's header, OPERATION step 3).
    private func placeLights(_ lights: [EyeLight], cameraPosition: [Float], cameraRotation: [Float]) throws {
        for child in lightRoot.childNodes { child.removeFromParentNode() }
        var ambient = SIMD3<Float>(repeating: 0)
        let cameraTransform = eyeWorldTransform(position: cameraPosition, rotation: cameraRotation)
        for light in lights {
            ambient += light.ambient
            let scnLight = SCNLight()
            scnLight.color = LinearColor.make(light.diffuse.x, light.diffuse.y, light.diffuse.z)
            let node = SCNNode()
            if light.headlight {
                scnLight.type = .directional
                node.simdTransform = cameraTransform
            } else if light.directional {
                scnLight.type = .directional
                node.simdTransform = eyeOrientation(pointingAlong: light.direction)
            } else {
                scnLight.type = .spot
                scnLight.spotInnerAngle = 0
                scnLight.spotOuterAngle = CGFloat(2 * light.cutoffDegrees)
                var transform = eyeOrientation(pointingAlong: light.direction)
                transform.columns.3 = SIMD4<Float>(light.position.x, light.position.y, light.position.z, 1)
                node.simdTransform = transform
            }
            scnLight.castsShadow = light.castsShadow
            if light.castsShadow {
                scnLight.shadowMapSize = CGSize(width: settings.shadowMapSize, height: settings.shadowMapSize)
                scnLight.shadowMode = .forward
            }
            node.light = scnLight
            lightRoot.addChildNode(node)
        }
        if ambient.x > 0 || ambient.y > 0 || ambient.z > 0 {
            let scnLight = SCNLight()
            scnLight.type = .ambient
            scnLight.color = LinearColor.make(ambient.x, ambient.y, ambient.z)
            let node = SCNNode()
            node.light = scnLight
            lightRoot.addChildNode(node)
        }
    }
}
