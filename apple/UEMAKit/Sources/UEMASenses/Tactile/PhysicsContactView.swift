// PhysicsContactView.swift: UEMACore's `PhysicsReading` presented as this module's
// `ContactPhysicsReading` (work item WI-28, target B; specification REQ-B-3 "a tactile read
// every physics substep" and REQ-B-5; plan row WI-25 "readers for ... contacts (ncon,
// mj_contactForce)").
//
// WHAT THIS FILE IS. The join between the shared physics contract and the contact reader of
// ContactReader.swift. `PhysicsReading` (UEMACore/Protocols.swift) gives per physics step:
// `contacts` (one `Contact` per `data.contact[i]`, i < ncon, each with the two geom ids, the
// signed distance, the midpoint position, the 9-number frame and the 6 numbers of
// mj_contactForce), `xpos` (nbody x 3) and `xmat` (nbody x 9). Three model constants the
// reader also needs are not in `PhysicsReading` (its doc comment: "The time-invariant model
// constants ... are read through UEMAPhysics's own model-description protocol (WI-25)"), so
// they come in as `ContactModelTables`: `model.geom_bodyid` (ngeom), `model.body_rootid`
// (nbody) and the body names (nbody). The velocity of a contact point comes from an injected
// `PointVelocityReading` (env.py:2337-2341: `mj_jac` at the contact point for each body, times
// `qvel`).
//
// INPUT. `any PhysicsReading` after a physics step; `ContactModelTables`; a
// `PointVelocityReading`. OPERATION. Copies `contacts`, `xpos` and `xmat` once at init (each
// access of a `PhysicsReading` array property may copy the whole array out of MuJoCo), then
// answers the `ContactPhysicsReading` calls by index into those copies. OUTPUT. A value the
// `ContactReader` reads; it holds no state between physics steps.
//
// SOLID. Single responsibility: index arithmetic over copied arrays. Dependency inversion:
// the physics, the tables and the velocity rule are all injected.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain); the "First build on a Mac" checklist is in TouchPooler.swift's header.

import Foundation
import UEMACore

/// The three model constants the contact reader needs beyond `PhysicsReading`, copied once
/// from the loaded MuJoCo model by the physics module (WI-25).
public struct ContactModelTables: Equatable {
    /// `model.geom_bodyid[g]`: "id of geom's body", one per geom (ngeom entries).
    public let geomBodyIds: [Int]
    /// `model.body_rootid[b]`: "ancestor that is direct child of world", one per body.
    public let bodyRootIds: [Int]
    /// `model.body(b).name`, one per body.
    public let bodyNames: [String]

    /// Makes the tables; the arrays are stored as given.
    public init(geomBodyIds: [Int], bodyRootIds: [Int], bodyNames: [String]) {
        self.geomBodyIds = geomBodyIds
        self.bodyRootIds = bodyRootIds
        self.bodyNames = bodyNames
    }

    /// The id of the body named `name`, or nil (env.py:1201 `model.body("hip").id`).
    public func bodyId(named name: String) -> Int? {
        bodyNames.firstIndex(of: name)
    }

    /// The infant's root body: `body_rootid` of the `hip` body (env.py:1200-1202), or nil
    /// when the model has no body of that name.
    public var infantRootBodyId: Int? {
        guard let hip = bodyId(named: "hip"), hip < bodyRootIds.count else { return nil }
        return bodyRootIds[hip]
    }
}

/// The world-frame linear velocity, m/s, of a world point taken as attached to a body.
public protocol PointVelocityReading {
    /// env.py:2337-2341: the 3 x nv translational Jacobian `mj_jac(m, d, jacp, NULL, point,
    /// bodyId)` times `data.qvel`. Body 0 (the world) gives zero.
    func pointVelocity(ofBody bodyId: Int, atWorldPoint point: Vec3, in physics: any PhysicsReading) -> Vec3
}

/// The rigid-body form of the same velocity: v = v_ref + omega x (point - ref), with
/// (omega, v_ref) from `PhysicsReading.bodyVelocity(bodyId:)` (rotational 3 then linear 3,
/// world frame) and `ref` the point that velocity is referenced at. For one rigid body the
/// Jacobian product `mj_jac @ qvel` equals this formula exactly. Which point
/// `bodyVelocity` is referenced at is the physics module's choice (mj_objectVelocity with
/// mjOBJ_XBODY refers to `xpos`, with mjOBJ_BODY to `xipos`, the centre of mass), so the
/// reference point is injected; `atBodyOrigin` uses `xpos`.
public struct RigidPointVelocity: PointVelocityReading {
    /// The point, world frame, metres, at which `bodyVelocity(bodyId:)` is referenced.
    public let referencePoint: (any PhysicsReading, Int) -> Vec3

    /// Makes the rule over a reference-point function.
    public init(referencePoint: @escaping (any PhysicsReading, Int) -> Vec3) {
        self.referencePoint = referencePoint
    }

    /// The rule for a `bodyVelocity` referenced at the body frame origin `xpos[b]`.
    public static let atBodyOrigin = RigidPointVelocity { physics, bodyId in
        let xpos = physics.xpos
        let i = 3 * bodyId
        guard bodyId >= 0, i + 2 < xpos.count else { return Vec3.zero }
        return Vec3(xpos[i], xpos[i + 1], xpos[i + 2])
    }

    /// v_ref + omega x (point - ref); zero for a body whose velocity has fewer than 6 numbers.
    public func pointVelocity(ofBody bodyId: Int, atWorldPoint point: Vec3, in physics: any PhysicsReading) -> Vec3 {
        let six = physics.bodyVelocity(bodyId: bodyId)
        guard six.count >= 6 else { return Vec3.zero }
        let omega = Vec3(six[0], six[1], six[2])
        let linear = Vec3(six[3], six[4], six[5])
        return linear + omega.cross(point - referencePoint(physics, bodyId))
    }
}

/// One physics step's contacts and poses, read through `ContactPhysicsReading`.
public struct PhysicsContactView: ContactPhysicsReading {
    private let physics: any PhysicsReading
    private let tables: ContactModelTables
    private let velocity: PointVelocityReading
    private let contacts: [Contact]
    private let xpos: [Double]
    private let xmat: [Double]

    /// Copies `contacts`, `xpos` and `xmat` of `physics` once.
    public init(physics: any PhysicsReading, tables: ContactModelTables, velocity: PointVelocityReading) {
        self.physics = physics
        self.tables = tables
        self.velocity = velocity
        contacts = physics.contacts
        xpos = physics.xpos
        xmat = physics.xmat
    }

    /// `data.ncon`.
    public var contactCount: Int { contacts.count }

    /// `data.contact[index]`: geom ids, midpoint, frame and distance.
    public func contact(at index: Int) -> RawContact {
        let c = contacts[index]
        let position = Vec3(array: c.position) ?? Vec3.zero
        let frame = Mat3(rowMajor: c.frame) ?? Mat3.identity
        return RawContact(geom1: c.geomIds[0], geom2: c.geomIds[1], position: position,
                          frame: frame, distance: c.distance)
    }

    /// The six numbers of `mj_contactForce` carried in the `Contact`.
    public func contactForce(at index: Int) -> ContactForce {
        ContactForce(array: contacts[index].force) ?? ContactForce(normal: 0, tangent1: 0, tangent2: 0)
    }

    /// `model.geom_bodyid[geomId]`; -1 for a flex contact (geom id -1) or an id out of range.
    public func bodyId(ofGeom geomId: Int) -> Int {
        guard geomId >= 0, geomId < tables.geomBodyIds.count else { return -1 }
        return tables.geomBodyIds[geomId]
    }

    /// `model.body_rootid[bodyId]`; -1 out of range.
    public func rootBodyId(ofBody bodyId: Int) -> Int {
        guard bodyId >= 0, bodyId < tables.bodyRootIds.count else { return -1 }
        return tables.bodyRootIds[bodyId]
    }

    /// The body's name; "" out of range.
    public func bodyName(_ bodyId: Int) -> String {
        guard bodyId >= 0, bodyId < tables.bodyNames.count else { return "" }
        return tables.bodyNames[bodyId]
    }

    /// `data.xpos[bodyId]`; zero out of range.
    public func bodyPosition(_ bodyId: Int) -> Vec3 {
        let i = 3 * bodyId
        guard bodyId >= 0, i + 2 < xpos.count else { return Vec3.zero }
        return Vec3(xpos[i], xpos[i + 1], xpos[i + 2])
    }

    /// `data.xmat[bodyId]`, row-major; the identity out of range.
    public func bodyRotation(_ bodyId: Int) -> Mat3 {
        let i = 9 * bodyId
        guard bodyId >= 0, i + 8 < xmat.count else { return Mat3.identity }
        return Mat3(rowMajor: Array(xmat[i ..< i + 9])) ?? Mat3.identity
    }

    /// The injected velocity rule at this physics step.
    public func pointVelocity(ofBody bodyId: Int, atWorldPoint point: Vec3) -> Vec3 {
        velocity.pointVelocity(ofBody: bodyId, atWorldPoint: point, in: physics)
    }
}
