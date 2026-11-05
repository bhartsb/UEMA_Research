// GeomMirrorTests.swift: tests of Sources/UEMASenses/Vision/GeomMirror.swift (work item WI-29):
// the size meaning per MuJoCo geom type, the transform that puts each SceneKit primitive where
// MuJoCo draws the geom, node reuse, and the skipped types. They build SceneKit nodes and draw
// nothing. Nothing here was compiled on the machine that wrote it (no Swift toolchain).
//
// FIRST BUILD ON A MAC: `cd apple/UEMAKit && xcodebuild -scheme UEMAKit test
// -destination 'platform=macOS' -only-testing:UEMASensesTests/GeomMirrorTests` : exit 0, 10 tests.

import Foundation
import SceneKit
import simd
import XCTest
import UEMACore
import UEMASenses

final class GeomMirrorTests: XCTestCase {
    private let identity: [Float] = [1, 0, 0, 0, 1, 0, 0, 0, 1]

    private func geom(_ type: Int, size: [Float], position: [Float] = [0, 0, 0],
                      rotation: [Float]? = nil, rgba: [Float] = [0.5, 0.5, 0.5, 1]) throws -> SceneGeom {
        try SceneGeom(type: type, size: size, position: position, rotation: rotation ?? identity, rgba: rgba)
    }

    private func apply(_ m: simd_float4x4, _ p: SIMD3<Float>) -> SIMD3<Float> {
        let v = m * SIMD4<Float>(p.x, p.y, p.z, 1)
        return SIMD3(v.x, v.y, v.z)
    }

    private func assertClose(_ a: SIMD3<Float>, _ b: SIMD3<Float>, file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertLessThan(simd_length(a - b), 1e-5, "\(a) against \(b)", file: file, line: line)
    }

    /// Each released type maps to its primitive with MuJoCo's size meaning.
    func testShapePerType() throws {
        XCTAssertEqual(EyeGeomShape(geom: try geom(2, size: [0.1, 0, 0]), infinitePlaneHalfExtent: 9),
                       .sphere(radius: 0.1))
        XCTAssertEqual(EyeGeomShape(geom: try geom(3, size: [0.02, 0.05, 0]), infinitePlaneHalfExtent: 9),
                       .capsule(radius: 0.02, halfLength: 0.05))
        XCTAssertEqual(EyeGeomShape(geom: try geom(4, size: [0.1, 0.2, 0.3]), infinitePlaneHalfExtent: 9),
                       .ellipsoid(radii: SIMD3(0.1, 0.2, 0.3)))
        XCTAssertEqual(EyeGeomShape(geom: try geom(5, size: [0.03, 0.04, 0]), infinitePlaneHalfExtent: 9),
                       .cylinder(radius: 0.03, halfHeight: 0.04))
        XCTAssertEqual(EyeGeomShape(geom: try geom(6, size: [0.1, 0.2, 0.3]), infinitePlaneHalfExtent: 9),
                       .box(halfExtents: SIMD3(0.1, 0.2, 0.3)))
        XCTAssertEqual(EyeGeomShape(geom: try geom(0, size: [0.72, 0.12, 0.01]), infinitePlaneHalfExtent: 9),
                       .plane(halfWidth: 0.72, halfHeight: 0.12))
    }

    /// MuJoCo's infinite plane (the released floor, scene.xml:37 "size 0 0 .25") gets the given
    /// half extent.
    func testInfinitePlaneGetsHalfExtent() throws {
        XCTAssertEqual(EyeGeomShape(geom: try geom(0, size: [0, 0, 0.25]), infinitePlaneHalfExtent: 7.5),
                       .plane(halfWidth: 7.5, halfHeight: 7.5))
    }

    /// hfield (1), mesh (7), sdf (8) and decor types (100 and above) are not drawn.
    func testUnsupportedTypesHaveNoShape() throws {
        for type in [1, 7, 8, 100, 104] {
            XCTAssertNil(EyeGeomShape(geom: try geom(type, size: [1, 1, 1]), infinitePlaneHalfExtent: 9), "type \(type)")
        }
    }

    /// A capsule's SceneKit axis (local y) lands on the geom's MuJoCo z axis: with a rotation
    /// whose z column is world x, the cap tip at SceneKit (0, h, 0) lands at position + h x.
    func testCapsuleAxisFollowsMuJoCoZ() throws {
        // Columns: local x -> world y, local y -> world z, local z -> world x (row-major rows).
        let rotation: [Float] = [0, 0, 1,
                                 1, 0, 0,
                                 0, 1, 0]
        let g = try geom(3, size: [0.02, 0.05, 0], position: [1, 2, 3], rotation: rotation)
        let shape = try XCTUnwrap(EyeGeomShape(geom: g, infinitePlaneHalfExtent: 9))
        let transform = eyeWorldTransform(position: g.position, rotation: g.rotation) * shape.primitiveCorrection
        assertClose(apply(transform, SIMD3(0, 0.07, 0)), SIMD3(1.07, 2, 3))
        assertClose(apply(transform, SIMD3(0, 0, 0)), SIMD3(1, 2, 3))
    }

    /// An ellipsoid is a unit sphere scaled by its radii along MuJoCo's local axes.
    func testEllipsoidScale() throws {
        let g = try geom(4, size: [0.1, 0.2, 0.3], position: [0, 0, 1])
        let shape = try XCTUnwrap(EyeGeomShape(geom: g, infinitePlaneHalfExtent: 9))
        let transform = eyeWorldTransform(position: g.position, rotation: g.rotation) * shape.primitiveCorrection
        assertClose(apply(transform, SIMD3(1, 0, 0)), SIMD3(0.1, 0, 1))
        assertClose(apply(transform, SIMD3(0, 1, 0)), SIMD3(0, 0.2, 1))
        assertClose(apply(transform, SIMD3(0, 0, 1)), SIMD3(0, 0, 1.3))
    }

    /// The pose matrix reads the 9 numbers row-major: world = R x local.
    func testWorldTransformIsRowMajor() {
        // Rotation by +90 degrees about z: local x -> world y.
        let rz: [Float] = [0, -1, 0,
                           1, 0, 0,
                           0, 0, 1]
        let m = eyeWorldTransform(position: [0.5, 0, 0], rotation: rz)
        assertClose(apply(m, SIMD3(1, 0, 0)), SIMD3(0.5, 1, 0))
        assertClose(apply(m, SIMD3(0, 1, 0)), SIMD3(-0.5, 0, 0))
    }

    /// Box and capsule sizes reach SceneKit as full lengths (2 x half sizes; a capsule's
    /// SceneKit height includes both caps).
    func testMirrorBuildsGeometryWithFullLengths() throws {
        let mirror = GeomMirror(infinitePlaneHalfExtent: 9)
        let report = mirror.update(geoms: [try geom(6, size: [0.1, 0.2, 0.3]), try geom(3, size: [0.02, 0.05, 0])])
        XCTAssertEqual(report.drawnCount, 2)
        let box = try XCTUnwrap(mirror.geomNodes[0].geometry as? SCNBox)
        XCTAssertEqual(Double(box.width), 0.2, accuracy: 1e-6)
        XCTAssertEqual(Double(box.height), 0.4, accuracy: 1e-6)
        XCTAssertEqual(Double(box.length), 0.6, accuracy: 1e-6)
        let capsule = try XCTUnwrap(mirror.geomNodes[1].geometry as? SCNCapsule)
        XCTAssertEqual(Double(capsule.capRadius), 0.02, accuracy: 1e-6)
        XCTAssertEqual(Double(capsule.height), 0.14, accuracy: 1e-6)
        XCTAssertEqual(capsule.radialSegmentCount, GeomMirror.radialSegments)
    }

    /// Plane, cylinder and ellipsoid reach SceneKit with MuJoCo's size meaning: a plane of half
    /// extents 0.72 m x 0.12 m is a 1.44 m x 0.24 m SCNPlane, MuJoCo's infinite plane gets twice
    /// the mirror's half extent, a cylinder of half height 0.04 m is 0.08 m tall, and every
    /// ellipsoid is a unit SCNSphere at 28 segments whose radii sit in the node's transform.
    func testMirrorBuildsPlaneCylinderAndEllipsoid() throws {
        let mirror = GeomMirror(infinitePlaneHalfExtent: 9)
        let report = mirror.update(geoms: [
            try geom(0, size: [0.72, 0.12, 0.01]),
            try geom(0, size: [0, 0, 0.25]),
            try geom(5, size: [0.03, 0.04, 0]),
            try geom(4, size: [0.1, 0.2, 0.3], position: [0, 0, 1]),
        ])
        XCTAssertEqual(report.drawnByType, [0: 2, 5: 1, 4: 1])
        let plane = try XCTUnwrap(mirror.geomNodes[0].geometry as? SCNPlane)
        XCTAssertEqual(Double(plane.width), 1.44, accuracy: 1e-6)
        XCTAssertEqual(Double(plane.height), 0.24, accuracy: 1e-6)
        let floor = try XCTUnwrap(mirror.geomNodes[1].geometry as? SCNPlane)
        XCTAssertEqual(Double(floor.width), 18, accuracy: 1e-6)
        XCTAssertEqual(Double(floor.height), 18, accuracy: 1e-6)
        let cylinder = try XCTUnwrap(mirror.geomNodes[2].geometry as? SCNCylinder)
        XCTAssertEqual(Double(cylinder.radius), 0.03, accuracy: 1e-6)
        XCTAssertEqual(Double(cylinder.height), 0.08, accuracy: 1e-6)
        XCTAssertEqual(cylinder.radialSegmentCount, GeomMirror.radialSegments)
        let sphere = try XCTUnwrap(mirror.geomNodes[3].geometry as? SCNSphere)
        XCTAssertEqual(Double(sphere.radius), 1, accuracy: 1e-6)
        XCTAssertEqual(sphere.segmentCount, GeomMirror.radialSegments)
        assertClose(apply(mirror.geomNodes[3].simdTransform, SIMD3(0, 0, 1)), SIMD3(0, 0, 1.3))
        // A second ellipsoid of other radii keeps the same unit sphere.
        mirror.update(geoms: [
            try geom(0, size: [0.72, 0.12, 0.01]),
            try geom(0, size: [0, 0, 0.25]),
            try geom(5, size: [0.03, 0.04, 0]),
            try geom(4, size: [0.5, 0.5, 0.5]),
        ])
        XCTAssertTrue(mirror.geomNodes[3].geometry === sphere, "every ellipsoid shares one unit-sphere key")
        // A list without appearances gives MuJoCo's default shininess 0.5 x 128 = 64.
        let material = try XCTUnwrap(mirror.geomNodes[0].geometry?.firstMaterial)
        XCTAssertEqual(Double(material.shininess), 64, accuracy: 1e-4)
    }

    /// Nodes are reused; a shorter list hides the extra nodes; skipped types and alpha 0 are
    /// hidden and counted.
    func testMirrorReuseHidingAndReport() throws {
        let mirror = GeomMirror(infinitePlaneHalfExtent: 9)
        let first = mirror.update(geoms: [
            try geom(2, size: [0.1, 0, 0]),
            try geom(7, size: [1, 1, 1]),
            try geom(6, size: [0.1, 0.1, 0.1], rgba: [1, 1, 1, 0]),
            try geom(5, size: [0.1, 0.1, 0]),
        ])
        XCTAssertEqual(first.drawnByType, [2: 1, 5: 1])
        XCTAssertEqual(first.skippedByType, [7: 1])
        XCTAssertEqual(first.transparentCount, 1)
        XCTAssertEqual(mirror.geomNodes.map(\.isHidden), [false, true, true, false])
        let sphereNode = mirror.geomNodes[0]
        let sphereGeometry = sphereNode.geometry

        mirror.update(geoms: [try geom(2, size: [0.1, 0, 0], position: [0, 0, 2])])
        XCTAssertEqual(mirror.geomNodes.count, 4)
        XCTAssertTrue(mirror.geomNodes[0] === sphereNode)
        XCTAssertTrue(mirror.geomNodes[0].geometry === sphereGeometry, "same shape key keeps the geometry")
        XCTAssertEqual(mirror.geomNodes.map(\.isHidden), [false, true, true, true])
        XCTAssertEqual(mirror.geomNodes[0].simdPosition.z, 2, accuracy: 1e-6)
        XCTAssertEqual(mirror.rootNode.childNodes.count, 4)
    }

    /// MuJoCo's lighting inputs reach the material: blinn, specular grey, shininess x 128,
    /// alpha as transparency.
    func testMaterialFollowsAppearance() throws {
        let mirror = GeomMirror(infinitePlaneHalfExtent: 9)
        mirror.update(geoms: [try geom(2, size: [0.1, 0, 0], rgba: [0.2, 0.4, 0.6, 0.5])],
                      appearances: [GeomAppearance(emission: 0.75, specular: 0.03, shininess: 0.03)])
        let material = try XCTUnwrap(mirror.geomNodes[0].geometry?.firstMaterial)
        XCTAssertEqual(material.lightingModel, .blinn)
        XCTAssertEqual(Double(material.shininess), 0.03 * 128, accuracy: 1e-4)
        XCTAssertEqual(Double(material.transparency), 0.5, accuracy: 1e-6)
        XCTAssertEqual(GeomAppearance.mujocoDefault.specularExponent, 64, accuracy: 1e-6)
    }
}
