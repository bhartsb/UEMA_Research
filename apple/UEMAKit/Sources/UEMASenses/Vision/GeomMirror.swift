// GeomMirror.swift: MuJoCo's abstract geom list (mjvGeom records after mjv_updateScene) mirrored
// as SceneKit nodes (work item WI-29 of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html;
// specification REQ-B-6). Nothing in this file was compiled on the machine that wrote it (no
// Swift toolchain; plan section 3, fact 1).
//
// WHAT THIS FILE IS. Three parts, one responsibility each: EyeGeomShape turns one
// UEMACore.SceneGeom (type, size, position, rotation, rgba) into the SceneKit primitive and the
// 4 x 4 transform that draws the same solid; GeomAppearance holds the material numbers MuJoCo
// keeps beside the geom (emission, specular, shininess); GeomMirror keeps one SCNNode per geom
// index under one root node and updates their geometry, transform and material from the list.
// The classic renderer (mjr_, OpenGL 1.5) is never called: iOS has no desktop OpenGL (plan row
// WI-29 evidence).
//
// INPUT. `[SceneGeom]` from `SceneGeomReading.sceneGeoms()` (UEMACore/Protocols.swift): per geom
// `type` (mjtGeom: 0 plane, 1 hfield, 2 sphere, 3 capsule, 4 ellipsoid, 5 cylinder, 6 box, 7 mesh,
// 8 sdf; decor types from 100; MuJoCo mjtGeom enum, the same numbers as
// export_abstract_scene.py GEOM_TYPE_NAMES), `size` (3 numbers, metres), `position` (3, metres,
// world frame, z up), `rotation` (9, row-major, world = R x local), `rgba` (4, 0 to 1). Optional
// `[GeomAppearance]` of the same count (the mjvGeom emission, specular, shininess fields that
// export_abstract_scene.py geom_row() writes; UEMACore.SceneGeom does not carry them).
//
// THE SIZE MEANING PER TYPE (MuJoCo XML reference, geom/size, as the classic renderer draws it):
//   plane      size[0], size[1] half extents along local x and y; the visible face points along
//              local +z; 0 in either means infinite (scene.xml:37, the floor "size 0 0 .25")
//   sphere     size[0] radius
//   capsule    size[0] radius, size[1] half length of the cylinder part, axis local z
//   ellipsoid  size[0], size[1], size[2] radii along local x, y, z
//   cylinder   size[0] radius, size[1] half height, axis local z
//   box        size[0], size[1], size[2] half extents along local x, y, z
// The released crib bundle's three XML files name no other geom type (`type="..."` attributes
// counted in scene.xml, MIMo_modelv2_grown24_fovea.xml and MIMo_metav2_lite.xml: plane, sphere,
// capsule, ellipsoid, cylinder and box, besides joint, texture and skybox types; no <mesh> or
// <hfield> element; bundle.json "mesh_files_referenced": 0). hfield, mesh, sdf and decor geoms
// are skipped and counted.
//
// OPERATION. Per geom: the SceneKit primitive whose own axes match MuJoCo's after one fixed
// correction (SCNCapsule and SCNCylinder run along their local y, MuJoCo's along local z: a
// rotation of +90 degrees about x takes y to z; SCNPlane lies in its local x-y plane facing +z,
// SCNBox's width, height and length run along x, y, z, as in MuJoCo); an ellipsoid is a unit
// SCNSphere scaled by its three radii. The node's simdTransform = T(position) x R(rotation) x
// correction x scale. Material: MuJoCo's fixed-function lighting in the classic renderer takes
// rgba as both ambient and diffuse colour, emission x rgba as emitted colour, specular as a grey
// specular colour and shininess x 128 as the specular exponent (OpenGL GL_SHININESS ranges 0 to
// 128); SceneKit's .blinn model gets the same four inputs. Colours are built in the linear sRGB
// colour space so that SceneKit converts none of them (the renderer's header gives the reason).
// Tessellation follows MuJoCo's vis.quality defaults: numslices 28, numstacks 16 (mjvisual
// documentation defaults; scene.xml:11 sets only shadowsize).
//
// OUTPUT. GeomMirror.rootNode (add it to the SCNScene once) with one child per geom index; the
// node of an index beyond the current list, or of a skipped type, or of rgba alpha 0, is hidden.
// update(...) returns GeomMirrorReport: counts drawn per type and skipped per type.
//
// FIRST BUILD ON A MAC (this file's part of WI-29's acceptance).
//  1. `cd apple/UEMAKit && xcodebuild -scheme UEMAKit test -destination
//     'platform=macOS' -only-testing:UEMASensesTests/GeomMirrorTests` : exit 0. These tests
//     build nodes and read transforms; they draw nothing.
//  2. If SCNCapsule's `capRadius`/`height`, SCNCylinder's `radius`/`height` or
//     `radialSegmentCount` fail to compile, the SceneKit SDK changed: read the SDK header and
//     keep the size meaning above.
//  3. SceneKit's framework is marked deprecated in the 26.0 SDKs (developer.apple.com SCNRenderer
//     page, read November 4, 2026: "Deprecated: 26.0"): deprecation warnings are expected and are
//     not errors; record their count in apple/FIRST_BUILD_RECORD.md.
//  4. Record the command, exit code and clock time (AM/PM Pacific) in apple/FIRST_BUILD_RECORD.md.
//
// SOURCES READ (November 4, 2026): UEMACore/Protocols.swift (SceneGeom, SceneGeomReading,
// EyeRendering); Source/Python/mimo_bridge/env.py:603-638 (EyeRenderer: mjv_updateScene with a
// default MjvOption and mjCAT_ALL, then mjr_render); the bundle's scene.xml:8-39 (visual, assets,
// lights); Tests/UEMASensesTests/Fixtures/eyes/export_abstract_scene.py (the mjvGeom fields it
// writes); https://developer.apple.com/tutorials/data/documentation/scenekit/scnrenderer.json
// (fetched November 4, 2026, for the renderer's methods; cited in SceneKitEyeRenderer.swift).

import CoreGraphics
import Foundation
import SceneKit
import simd
import UEMACore

/// The MuJoCo geom types by their mjtGeom number (MuJoCo's mjtGeom enum; 100 and above are
/// decor types, which have no case here).
public enum EyeGeomType: Int, CaseIterable, Sendable {
    case plane = 0
    case hfield = 1
    case sphere = 2
    case capsule = 3
    case ellipsoid = 4
    case cylinder = 5
    case box = 6
    case mesh = 7
    case sdf = 8
}

/// The SceneKit primitive that draws one MuJoCo geom, with sizes in metres.
public enum EyeGeomShape: Equatable, Sendable {
    /// Half extents along local x and y; faces local +z.
    case plane(halfWidth: Float, halfHeight: Float)
    /// Radius.
    case sphere(radius: Float)
    /// Radius and half length of the cylinder part, axis local z.
    case capsule(radius: Float, halfLength: Float)
    /// Radii along local x, y, z.
    case ellipsoid(radii: SIMD3<Float>)
    /// Radius and half height, axis local z.
    case cylinder(radius: Float, halfHeight: Float)
    /// Half extents along local x, y, z.
    case box(halfExtents: SIMD3<Float>)

    /// The shape of `geom`, or nil for a type the mirror does not draw (hfield, mesh, sdf,
    /// decor). A plane with size[0] or size[1] at 0 or below (MuJoCo's infinite plane) gets
    /// `infinitePlaneHalfExtent` metres in that direction.
    public init?(geom: SceneGeom, infinitePlaneHalfExtent: Float) {
        let s = geom.size
        switch EyeGeomType(rawValue: geom.type) {
        case .plane?:
            self = .plane(halfWidth: s[0] > 0 ? s[0] : infinitePlaneHalfExtent,
                          halfHeight: s[1] > 0 ? s[1] : infinitePlaneHalfExtent)
        case .sphere?:
            self = .sphere(radius: s[0])
        case .capsule?:
            self = .capsule(radius: s[0], halfLength: s[1])
        case .ellipsoid?:
            self = .ellipsoid(radii: SIMD3(s[0], s[1], s[2]))
        case .cylinder?:
            self = .cylinder(radius: s[0], halfHeight: s[1])
        case .box?:
            self = .box(halfExtents: SIMD3(s[0], s[1], s[2]))
        case .hfield?, .mesh?, .sdf?, nil:
            return nil
        }
    }

    /// The fixed transform from the SceneKit primitive's own frame to MuJoCo's local geom frame:
    /// +90 degrees about x for capsule and cylinder (SceneKit axis y to MuJoCo axis z); the
    /// radii as a scale for the ellipsoid (drawn as a unit sphere); identity otherwise.
    public var primitiveCorrection: simd_float4x4 {
        switch self {
        case .capsule, .cylinder:
            // Rotation about x by +90 degrees: y -> z, z -> -y.
            return simd_float4x4(columns: (
                SIMD4<Float>(1, 0, 0, 0),
                SIMD4<Float>(0, 0, 1, 0),
                SIMD4<Float>(0, -1, 0, 0),
                SIMD4<Float>(0, 0, 0, 1)))
        case let .ellipsoid(radii):
            return simd_float4x4(diagonal: SIMD4<Float>(radii.x, radii.y, radii.z, 1))
        case .plane, .sphere, .box:
            return matrix_identity_float4x4
        }
    }

    /// A key that changes exactly when the SceneKit geometry must be rebuilt (the ellipsoid's
    /// radii live in the transform, so every ellipsoid shares one unit-sphere key).
    var geometryKey: EyeGeomShape {
        if case .ellipsoid = self { return .ellipsoid(radii: SIMD3(1, 1, 1)) }
        return self
    }
}

/// The 4 x 4 world transform of a MuJoCo pose: rotation `rotation` (9 numbers, row-major,
/// world = R x local) and translation `position` (3 numbers, metres), as a column-vector
/// matrix (simd and SceneKit's `simdTransform` take column vectors).
public func eyeWorldTransform(position: [Float], rotation: [Float]) -> simd_float4x4 {
    precondition(position.count == 3 && rotation.count == 9, "a pose is 3 position and 9 rotation numbers")
    let r = rotation
    return simd_float4x4(columns: (
        SIMD4<Float>(r[0], r[3], r[6], 0),
        SIMD4<Float>(r[1], r[4], r[7], 0),
        SIMD4<Float>(r[2], r[5], r[8], 0),
        SIMD4<Float>(position[0], position[1], position[2], 1)))
}

/// The material numbers MuJoCo keeps beside a geom in its mjvGeom record (emission, specular,
/// shininess, reflectance; each 0 to 1). The defaults are MuJoCo's material defaults (XML
/// reference, material: specular "0.5", shininess "0.5", emission "0", reflectance "0").
public struct GeomAppearance: Equatable, Hashable, Codable, Sendable {
    /// Emitted colour as a multiple of rgba, 0 to 1.
    public let emission: Float
    /// Grey specular colour, 0 to 1.
    public let specular: Float
    /// Specular sharpness, 0 to 1 (x 128 gives the exponent).
    public let shininess: Float
    /// Mirror reflectance, 0 to 1. Recorded; the mirror draws no reflections (the fixture
    /// exporter's "reflections_off" ablation measures what this costs).
    public let reflectance: Float

    /// An appearance.
    public init(emission: Float = 0, specular: Float = 0.5, shininess: Float = 0.5, reflectance: Float = 0) {
        self.emission = emission
        self.specular = specular
        self.shininess = shininess
        self.reflectance = reflectance
    }

    /// MuJoCo's material defaults.
    public static let mujocoDefault = GeomAppearance()

    /// The specular exponent OpenGL receives: shininess x 128.
    public var specularExponent: Float { shininess * 128 }
}

/// What one `GeomMirror.update` drew and skipped.
public struct GeomMirrorReport: Equatable, Sendable {
    /// Geoms drawn, per mjtGeom number.
    public let drawnByType: [Int: Int]
    /// Geoms skipped for their type, per mjtGeom number.
    public let skippedByType: [Int: Int]
    /// Geoms hidden for rgba alpha 0.
    public let transparentCount: Int

    /// Geoms drawn in all.
    public var drawnCount: Int { drawnByType.values.reduce(0, +) }
    /// Geoms skipped for their type in all.
    public var skippedCount: Int { skippedByType.values.reduce(0, +) }
}

/// Colours in the linear sRGB colour space, so that SceneKit's linear-space lighting receives the
/// numbers as given (MuJoCo's fixed-function lighting applies no gamma).
enum LinearColor {
    /// The linear sRGB colour space, or nil where the platform has none.
    static let space: CGColorSpace? = CGColorSpace(name: CGColorSpace.linearSRGB)

    /// A colour of the given components, each clamped to 0 ... 1.
    static func make(_ red: Float, _ green: Float, _ blue: Float, _ alpha: Float = 1) -> CGColor {
        let c: [CGFloat] = [red, green, blue, alpha].map { CGFloat(min(max($0, 0), 1)) }
        if let space = space, let color = CGColor(colorSpace: space, components: c) {
            return color
        }
        return CGColor(red: c[0], green: c[1], blue: c[2], alpha: c[3])
    }
}

/// One SceneKit node per MuJoCo geom index, kept under `rootNode` and updated from the geom list
/// of each state. Nodes are reused between updates; a node's geometry is rebuilt only when its
/// shape key changes.
public final class GeomMirror {
    /// Slices around a round primitive: MuJoCo vis.quality numslices default 28.
    public static let radialSegments = 28
    /// Stacks along a round primitive: MuJoCo vis.quality numstacks default 16.
    public static let stackSegments = 16

    /// The parent of every geom node; add it to the scene once.
    public let rootNode = SCNNode()

    /// The half extent in metres given to an infinite plane (the renderer passes its far
    /// clipping distance).
    public let infinitePlaneHalfExtent: Float

    private var nodes: [SCNNode] = []
    private var keys: [EyeGeomShape?] = []

    /// A mirror whose infinite planes get `infinitePlaneHalfExtent` metres (above 0).
    public init(infinitePlaneHalfExtent: Float) {
        precondition(infinitePlaneHalfExtent > 0, "infinite plane half extent must be above 0 m")
        self.infinitePlaneHalfExtent = infinitePlaneHalfExtent
        rootNode.name = "geom_mirror"
    }

    /// The nodes in geom-index sequence (for tests and the scene view); a hidden node draws
    /// nothing.
    public var geomNodes: [SCNNode] { nodes }

    /// Updates every node from `geoms`. `appearances`, when given, holds one entry per geom;
    /// a missing or short list gives MuJoCo's material defaults to the rest.
    @discardableResult
    public func update(geoms: [SceneGeom], appearances: [GeomAppearance]? = nil) -> GeomMirrorReport {
        var drawn: [Int: Int] = [:]
        var skipped: [Int: Int] = [:]
        var transparent = 0
        while nodes.count < geoms.count {
            let node = SCNNode()
            node.name = "geom_\(nodes.count)"
            node.isHidden = true
            rootNode.addChildNode(node)
            nodes.append(node)
            keys.append(nil)
        }
        for (i, geom) in geoms.enumerated() {
            let node = nodes[i]
            guard let shape = EyeGeomShape(geom: geom, infinitePlaneHalfExtent: infinitePlaneHalfExtent) else {
                skipped[geom.type, default: 0] += 1
                node.isHidden = true
                continue
            }
            guard geom.rgba[3] > 0 else {
                transparent += 1
                node.isHidden = true
                continue
            }
            let key = shape.geometryKey
            if keys[i] != key || node.geometry == nil {
                node.geometry = GeomMirror.makeGeometry(for: key)
                node.geometry?.firstMaterial = SCNMaterial()
                keys[i] = key
            }
            node.simdTransform = eyeWorldTransform(position: geom.position, rotation: geom.rotation)
                * shape.primitiveCorrection
            let appearance: GeomAppearance
            if let list = appearances, i < list.count {
                appearance = list[i]
            } else {
                appearance = GeomAppearance.mujocoDefault
            }
            if let material = node.geometry?.firstMaterial {
                GeomMirror.apply(rgba: geom.rgba, appearance: appearance, to: material)
            }
            node.isHidden = false
            drawn[geom.type, default: 0] += 1
        }
        if geoms.count < nodes.count {
            for i in geoms.count ..< nodes.count { nodes[i].isHidden = true }
        }
        return GeomMirrorReport(drawnByType: drawn, skippedByType: skipped, transparentCount: transparent)
    }

    /// The SceneKit geometry of one shape key, at MuJoCo's tessellation.
    static func makeGeometry(for shape: EyeGeomShape) -> SCNGeometry {
        switch shape {
        case let .plane(halfWidth, halfHeight):
            let plane = SCNPlane(width: CGFloat(2 * halfWidth), height: CGFloat(2 * halfHeight))
            return plane
        case let .sphere(radius):
            let sphere = SCNSphere(radius: CGFloat(radius))
            sphere.segmentCount = radialSegments
            return sphere
        case .ellipsoid:
            let sphere = SCNSphere(radius: 1)
            sphere.segmentCount = radialSegments
            return sphere
        case let .capsule(radius, halfLength):
            // SCNCapsule's height is the whole length, both caps included.
            let capsule = SCNCapsule(capRadius: CGFloat(radius), height: CGFloat(2 * halfLength + 2 * radius))
            capsule.radialSegmentCount = radialSegments
            capsule.capSegmentCount = stackSegments
            return capsule
        case let .cylinder(radius, halfHeight):
            let cylinder = SCNCylinder(radius: CGFloat(radius), height: CGFloat(2 * halfHeight))
            cylinder.radialSegmentCount = radialSegments
            return cylinder
        case let .box(halfExtents):
            return SCNBox(width: CGFloat(2 * halfExtents.x), height: CGFloat(2 * halfExtents.y),
                          length: CGFloat(2 * halfExtents.z), chamferRadius: 0)
        }
    }

    /// Writes MuJoCo's four lighting inputs into a SceneKit material (this file's header,
    /// OPERATION).
    static func apply(rgba: [Float], appearance: GeomAppearance, to material: SCNMaterial) {
        material.lightingModel = .blinn
        material.diffuse.contents = LinearColor.make(rgba[0], rgba[1], rgba[2])
        material.ambient.contents = LinearColor.make(rgba[0], rgba[1], rgba[2])
        material.emission.contents = LinearColor.make(rgba[0] * appearance.emission,
                                                      rgba[1] * appearance.emission,
                                                      rgba[2] * appearance.emission)
        material.specular.contents = LinearColor.make(appearance.specular, appearance.specular, appearance.specular)
        material.shininess = CGFloat(appearance.specularExponent)
        material.transparency = CGFloat(min(max(rgba[3], 0), 1))
        material.isDoubleSided = false
    }
}
