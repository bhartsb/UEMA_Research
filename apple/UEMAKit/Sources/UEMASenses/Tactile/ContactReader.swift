// ContactReader.swift: the first tactile stage of the UEMA Apple port (work item WI-28,
// target B; specification REQ-B-5; contract apple/UEMAKit/Specs/touch_pooler.md section 1).
//
// WHAT THIS FILE IS. One MuJoCo contact, as the physics engine reports it, turned into one
// `SkinContact` record per skin body that the contact touches: the port of
// `BridgeEnv._contacts` (Source/Python/mimo_bridge/env.py:2305-2356) together with the
// parts of MIMo's `TrimeshTouch` it calls (third_party/MIMo/mimo/mimo_infant/touch/
// touch.py:1394-1414 `get_contact_position_relative`, :1444-1465 `get_contact_normal`,
// :1419-1442 `get_raw_force`, :1470-1504 `force_vector_global` and `force_vector`).
// The record is what the two later stages consume: `LinearContactSpreader` (SkinSpread.swift)
// spreads `forceBody` over the skin points, `GoertzelTouchPooler` (TouchPooler.swift) sums
// the records into the patch fields.
//
// INPUT (per physics read). `ContactPhysicsReading`, the five MuJoCo arrays this stage
// needs: `data.ncon` and `data.contact[i]` (geom ids, position `pos`, `frame`, `dist`;
// MuJoCo API reference, types: "mjtNum dist; // distance between nearest points; neg:
// penetration", "mjtNum pos[3]; // position of contact point: midpoint between geoms",
// "mjtNum frame[9]; // normal is in [0-2], points from geom[0] to geom[1]";
// https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html), `mj_contactForce`
// ("Extract 6D force:torque given contact id, in the contact frame";
// https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html), `model.geom_bodyid`
// ("id of geom's body"), `model.body_rootid` ("ancestor that is direct child of world"),
// `data.xpos` ("Cartesian position of body frame (nbody x 3)"), `data.xmat` ("Cartesian
// orientation of body frame (nbody x 9)") and the velocity of a point attached to a body,
// which the physics module forms from `mj_jac` ("computes an end-effector kinematic
// Jacobian ... Each component is a 3-by-nv matrix"; same functions page) times `data.qvel`.
//
// OPERATION. For contact i with geoms g1, g2 of bodies b1, b2 (env.py:2311-2314): skip it
// when neither body carries skin, or when |fn| < 1e-9 N with fn = result[0] of
// mj_contactForce (env.py:2316-2320). f_world = n fn + t1 ft1 + t2 ft2, the force on geom2
// from geom1 (env.py:2322; n, t1, t2 are the three rows of `frame`). For each skin body b
// of the two (env.py:2326-2355): n_into = n when b is geom2's body, else -n; f_on_b =
// f_world when b is geom2's body, else -f_world; tang_world = f_on_b - (f_on_b . n_into)
// n_into; R = xmat[b] as a 3 x 3 matrix; tangential_body = R^T tang_world; normal_body =
// R^T n_into; pos_body = R^T (pos - xpos[b]), moved by normal_body x dist / 2 when dist < 0
// (touch.py:1410-1414: onto the sensing body's surface); force_body = R^T f_on_b
// (touch.py:1486-1504); rel = v(b1, pos) - v(b2, pos), negated when b is geom1's body
// (env.py:2340-2342); closing speed = max(0, rel . n_into) m/s; tangent speed =
// |rel - (rel . n_into) n_into| m/s; friction power = max(0, tang_world . rel) W
// (env.py:2343-2347); penetration = max(0, -dist) m (env.py:2324); the other body's name;
// otherIsSelf = the other body's root body is the infant's root body (env.py:1196-1205).
//
// OUTPUT. `[SkinContact]`, one record per (contact, skin body) pair, in contact index order
// and, within a contact, geom1's body before geom2's body (env.py:2311, :2326). A self-touch
// (both bodies skin) yields two records with the same geom pair (touch_adapter.py:285-290).
//
// MIMo'S IN-PLACE NORMAL NEGATION, REPRODUCED ON REQUEST. MIMo's `get_contact_normal`
// negates MuJoCo's contact frame in place (touch.py:1455-1458: `normal_vector =
// contact.frame[:3]` is a view of the frame, then `normal_vector *= -1` when geom 1 belongs
// to the sensing body), and `get_contact_position_relative` calls it only when dist < 0
// (touch.py:1409-1413). So for a contact whose geom-1 body has skin and whose dist is
// negative, each read of the geom-1 side flips the stored frame, and a record computed
// after an odd number of flips reads -n where MuJoCo wrote n. Which record that is depends
// on the call sequence on one physics state:
//   - MIMo `get_touch_obs` on a frame MuJoCo just wrote (touch.py:1598-1621; the forces of
//     get_contacts are taken before any flip, touch.py:1529-1558): the geom-1 side is
//     spread first and flips the frame, so the geom-2 side is spread at a position
//     corrected along -n (outward by |dist| from the physical one). `FrameNegation.geom2Side`.
//   - env.py `_contacts` on a frame MuJoCo just wrote: `n_into` of the geom-2 side is read
//     after the geom-1 side's `get_contact_position_relative` (env.py:2328, :2334) flipped
//     it. This is the trainer's `_collect` under mechanical_v4 (env.py:3116-3119): the
//     physics-step read before it ran `get_touch_obs` and `_contacts` (env.py:2360-2361),
//     two flips, so the frame is again as MuJoCo wrote it. `FrameNegation.geom2Side`.
//   - env.py `_contacts` right after `get_touch_obs` on the same frame (one flip already
//     done): the geom-1 side reads -n. This is `_collect` without physics-step reads
//     (env.py:3116-3118) and the WI-14 exporter (export_eye_fixtures.py:381-383).
//     `FrameNegation.geom1Side`.
// On the record that reads -n, `normalWorld`, `normalBody`, the surface correction of
// `positionBody` and `closingSpeed` use -n; the tangential force, the tangent speed, the
// friction power and `forceBody` do not depend on the sign (f - (f . n) n is the same for n
// and -n; MIMo's `force_vector` is computed before the flips). `FrameNegation.none` gives
// the physical normal on both records. Only self-touch contacts qualify (both bodies carry
// skin). MIMo also gives the two eye bodies one placeholder point each (has_sensors true);
// a reader built without the eye ids does not count an eye as geom-1 skin.
// TactileFixtureTests.swift undoes the negation of the WI-14 records when it compares the
// spread.
//
// SOLID. Single responsibility: this file reads contacts and converts frames, nothing else.
// Dependency inversion: the reader depends on the `ContactPhysicsReading` protocol; the
// MuJoCo-backed type of work item WI-25 (UEMAPhysics) adopts it and is injected.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain); the "First build on a Mac" checklist is in TouchPooler.swift's header.

import Foundation

// MARK: - Small vector and matrix types

/// A 3-vector of Double: a position in metres, a force in newtons or a unit direction.
/// Plain fields instead of simd so that every operation here is one visible formula.
public struct Vec3: Equatable {
    /// The three components.
    public var x: Double
    public var y: Double
    public var z: Double

    /// Makes a vector from its three components.
    public init(_ x: Double, _ y: Double, _ z: Double) {
        self.x = x
        self.y = y
        self.z = z
    }

    /// Makes a vector from an array of exactly three numbers; any other count gives nil.
    public init?(array: [Double]) {
        guard array.count == 3 else { return nil }
        self.init(array[0], array[1], array[2])
    }

    /// The zero vector.
    public static let zero = Vec3(0, 0, 0)

    /// The components as an array [x, y, z].
    public var array: [Double] { [x, y, z] }

    /// Dot product: x1 x2 + y1 y2 + z1 z2.
    public func dot(_ other: Vec3) -> Double {
        x * other.x + y * other.y + z * other.z
    }

    /// Cross product: (y1 z2 - z1 y2, z1 x2 - x1 z2, x1 y2 - y1 x2).
    public func cross(_ other: Vec3) -> Vec3 {
        Vec3(y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x)
    }

    /// Euclidean length sqrt(x^2 + y^2 + z^2) (numpy.linalg.norm with ord 2).
    public var norm: Double { (x * x + y * y + z * z).squareRoot() }

    public static func + (a: Vec3, b: Vec3) -> Vec3 { Vec3(a.x + b.x, a.y + b.y, a.z + b.z) }
    public static func - (a: Vec3, b: Vec3) -> Vec3 { Vec3(a.x - b.x, a.y - b.y, a.z - b.z) }
    public static prefix func - (a: Vec3) -> Vec3 { Vec3(-a.x, -a.y, -a.z) }
    public static func * (a: Vec3, s: Double) -> Vec3 { Vec3(a.x * s, a.y * s, a.z * s) }
    public static func * (s: Double, a: Vec3) -> Vec3 { Vec3(a.x * s, a.y * s, a.z * s) }
}

/// A 3 x 3 matrix of Double stored row-major as MuJoCo stores `xmat` and `frame`
/// (nine numbers; element (row r, column c) is `elements[3 r + c]`).
public struct Mat3: Equatable {
    /// The nine elements, row-major.
    public let elements: [Double]

    /// Makes a matrix from nine row-major numbers; any other count gives nil.
    public init?(rowMajor elements: [Double]) {
        guard elements.count == 9 else { return nil }
        self.elements = elements
    }

    /// The identity matrix.
    public static let identity = Mat3(unchecked: [1, 0, 0, 0, 1, 0, 0, 0, 1])

    /// Internal constructor for literals already known to hold nine numbers.
    init(unchecked elements: [Double]) {
        self.elements = elements
    }

    /// Row r of the matrix as a vector.
    public func row(_ r: Int) -> Vec3 {
        Vec3(elements[3 * r], elements[3 * r + 1], elements[3 * r + 2])
    }

    /// Column c of the matrix as a vector.
    public func column(_ c: Int) -> Vec3 {
        Vec3(elements[c], elements[3 + c], elements[6 + c])
    }

    /// M v (mimo_utils.rotate_vector, simulation/utils.py:32-47: body frame to world for
    /// an xmat).
    public func apply(_ v: Vec3) -> Vec3 {
        Vec3(row(0).dot(v), row(1).dot(v), row(2).dot(v))
    }

    /// M^T v (mimo_utils.rotate_vector_transpose, simulation/utils.py:50-62: world to body
    /// frame for an xmat; env.py:2331 `R_b.T @ tang_world`).
    public func applyTransposed(_ v: Vec3) -> Vec3 {
        Vec3(column(0).dot(v), column(1).dot(v), column(2).dot(v))
    }
}

// MARK: - What the physics engine must provide

/// One entry of `data.contact` (MuJoCo API reference, types, struct mjContact_):
/// the two geom ids, the midpoint position, the contact frame and the signed distance.
public struct RawContact: Equatable {
    /// `geom[0]` ("geom ids; -1 for flex"); the normal points away from this geom.
    public let geom1: Int
    /// `geom[1]`; the normal points towards this geom.
    public let geom2: Int
    /// `pos[3]`: "position of contact point: midpoint between geoms", metres, world frame.
    public let position: Vec3
    /// `frame[9]`: "normal is in [0-2], points from geom[0] to geom[1]"; rows are the
    /// normal, tangent 1 and tangent 2 directions in the world frame.
    public let frame: Mat3
    /// `dist`: "distance between nearest points; neg: penetration", metres.
    public let distance: Double

    /// Makes one raw contact record.
    public init(geom1: Int, geom2: Int, position: Vec3, frame: Mat3, distance: Double) {
        self.geom1 = geom1
        self.geom2 = geom2
        self.position = position
        self.frame = frame
        self.distance = distance
    }

    /// The contact normal (row 0 of the frame), from geom1 towards geom2.
    public var normal: Vec3 { frame.row(0) }
    /// The first tangent direction (row 1 of the frame).
    public var tangent1: Vec3 { frame.row(1) }
    /// The second tangent direction (row 2 of the frame).
    public var tangent2: Vec3 { frame.row(2) }
}

/// The six numbers `mj_contactForce` writes ("Extract 6D force:torque given contact id, in
/// the contact frame"): force along the normal and the two tangents in newtons, then the
/// three torques in newton metres. Only the three forces are read here (env.py:2317-2321).
public struct ContactForce: Equatable {
    /// `result[0]`: force along the contact normal, N (the force on geom2 from geom1).
    public let normal: Double
    /// `result[1]`: friction force along tangent 1, N.
    public let tangent1: Double
    /// `result[2]`: friction force along tangent 2, N.
    public let tangent2: Double
    /// `result[3...5]`: the torques, N m; carried through unread.
    public let torque: Vec3

    /// Makes one force record.
    public init(normal: Double, tangent1: Double, tangent2: Double, torque: Vec3 = .zero) {
        self.normal = normal
        self.tangent1 = tangent1
        self.tangent2 = tangent2
        self.torque = torque
    }

    /// Makes a force record from the six-number array `mj_contactForce` fills; any other
    /// count gives nil.
    public init?(array: [Double]) {
        guard array.count == 6 else { return nil }
        self.init(normal: array[0], tangent1: array[1], tangent2: array[2],
                  torque: Vec3(array[3], array[4], array[5]))
    }
}

/// The reads of the physics engine that the contact reader needs, each one named after the
/// MuJoCo array or function that answers it. The MuJoCo-backed physics type (work item
/// WI-25) adopts this protocol; tests adopt it with hand-written arrays.
public protocol ContactPhysicsReading {
    /// `data.ncon`: "number of detected contacts".
    var contactCount: Int { get }
    /// `data.contact[index]` for 0 <= index < contactCount.
    func contact(at index: Int) -> RawContact
    /// `mj_contactForce(m, d, index, result)`: the six numbers in the contact frame.
    func contactForce(at index: Int) -> ContactForce
    /// `model.geom_bodyid[geomId]`: "id of geom's body".
    func bodyId(ofGeom geomId: Int) -> Int
    /// `model.body_rootid[bodyId]`: "ancestor that is direct child of world".
    func rootBodyId(ofBody bodyId: Int) -> Int
    /// The body's name from the model (env.py:2354 `m.body(other).name`).
    func bodyName(_ bodyId: Int) -> String
    /// `data.xpos[bodyId]`: the body frame's origin in the world, metres.
    func bodyPosition(_ bodyId: Int) -> Vec3
    /// `data.xmat[bodyId]` as a 3 x 3 row-major matrix that takes body-frame vectors to
    /// the world frame (mimo_utils.get_body_rotation, simulation/utils.py:385-395).
    func bodyRotation(_ bodyId: Int) -> Mat3
    /// The world-frame linear velocity, m/s, of `point` (world frame, metres) taken as
    /// attached to body `bodyId`: the 3 x nv translational Jacobian of `mj_jac(m, d, jacp,
    /// NULL, point, bodyId)` multiplied by `data.qvel` (env.py:2337-2341 `mj_jac` then
    /// `jac @ qvel`).
    func pointVelocity(ofBody bodyId: Int, atWorldPoint point: Vec3) -> Vec3
}

// MARK: - The record

/// One MuJoCo contact as seen from one skin body: the port of `ContactInfo`
/// (touch_adapter.py:109-134) with every number in the unit env.py stores it in.
public struct SkinContact: Equatable {
    /// The skin body that senses this contact (`body_id`).
    public let bodyId: Int
    /// `pair_key[0]`: geom1 of the MuJoCo contact; with `geom2` it identifies the same
    /// contact from one update to the next (touch_adapter.py:113).
    public let geom1: Int
    /// `pair_key[1]`: geom2 of the MuJoCo contact.
    public let geom2: Int
    /// `normal_force`: |fn|, N, at least 0 (env.py:2350).
    public let normalForce: Double
    /// `tangential_body`: the friction force on the sensing body, N, in its body frame.
    public let tangentialBody: Vec3
    /// `pos_body`: the contact point on the sensing body's surface, metres, body frame.
    public let positionBody: Vec3
    /// `pos_world`: the contact midpoint, metres, world frame.
    public let positionWorld: Vec3
    /// `normal_world`: unit direction pointing into the sensing body, world frame.
    public let normalWorld: Vec3
    /// `normal_body`: the same direction in the body frame (R^T normal_world).
    public let normalBody: Vec3
    /// `force_body`: the whole contact force on the sensing body in its body frame, N:
    /// MIMo's `force_vector` (touch.py:1490-1504), the quantity the spreader distributes.
    /// It equals normalForce x normalBody + tangentialBody when fn >= 0.
    public let forceBody: Vec3
    /// `penetration_m`: max(0, -dist), metres.
    public let penetration: Double
    /// `rel_normal_speed`: closing speed along the normal, m/s, at least 0; nil when the
    /// caller has no velocity (touch_adapter.py:280 reads a missing value as 0).
    public let closingSpeed: Double?
    /// `rel_tangent_speed`: sliding speed at the contact point, m/s; nil when unknown, in
    /// which case the pooler matches stored positions instead (touch_adapter.py:300-306).
    public let tangentSpeed: Double?
    /// `friction_power_w`: max(0, tang_world . rel), W; nil when unknown.
    public let frictionPower: Double?
    /// `other_body_name`: the name of the body on the other side of the contact.
    public let otherBodyName: String
    /// `other_is_self`: the other body belongs to the infant (same root body).
    public let otherIsSelf: Bool

    /// Makes one record; every field is stored as given.
    public init(bodyId: Int, geom1: Int, geom2: Int, normalForce: Double, tangentialBody: Vec3,
                positionBody: Vec3, positionWorld: Vec3, normalWorld: Vec3, normalBody: Vec3,
                forceBody: Vec3, penetration: Double, closingSpeed: Double?, tangentSpeed: Double?,
                frictionPower: Double?, otherBodyName: String, otherIsSelf: Bool) {
        self.bodyId = bodyId
        self.geom1 = geom1
        self.geom2 = geom2
        self.normalForce = normalForce
        self.tangentialBody = tangentialBody
        self.positionBody = positionBody
        self.positionWorld = positionWorld
        self.normalWorld = normalWorld
        self.normalBody = normalBody
        self.forceBody = forceBody
        self.penetration = penetration
        self.closingSpeed = closingSpeed
        self.tangentSpeed = tangentSpeed
        self.frictionPower = frictionPower
        self.otherBodyName = otherBodyName
        self.otherIsSelf = otherIsSelf
    }

    /// The key the pooler tracks a contact by between updates: (body, geom1, geom2)
    /// (touch_adapter.py:297 `pair = (c.body_id, c.pair_key)`).
    public var pairKey: ContactPairKey { ContactPairKey(bodyId: bodyId, geom1: geom1, geom2: geom2) }
}

/// (sensing body, geom1, geom2): identifies one contact on one skin body across updates.
public struct ContactPairKey: Hashable {
    /// The sensing body.
    public let bodyId: Int
    /// geom1 of the MuJoCo contact.
    public let geom1: Int
    /// geom2 of the MuJoCo contact.
    public let geom2: Int

    /// Makes a key.
    public init(bodyId: Int, geom1: Int, geom2: Int) {
        self.bodyId = bodyId
        self.geom1 = geom1
        self.geom2 = geom2
    }

    /// The geom pair alone, shared by the two records of one self-touch
    /// (touch_adapter.py:290 `self_pair_keys.add(c.pair_key)`).
    public var geomPair: GeomPair { GeomPair(geom1: geom1, geom2: geom2) }
}

/// (geom1, geom2) of one MuJoCo contact.
public struct GeomPair: Hashable {
    /// geom1 of the MuJoCo contact.
    public let geom1: Int
    /// geom2 of the MuJoCo contact.
    public let geom2: Int

    /// Makes a pair.
    public init(geom1: Int, geom2: Int) {
        self.geom1 = geom1
        self.geom2 = geom2
    }
}

// MARK: - The reader

/// Which record of a qualifying contact (geom-1 body with skin, dist < 0) reads MuJoCo's
/// contact normal negated, reproducing MIMo's in-place frame negation (file header).
public enum FrameNegation: Equatable {
    /// Both records use the frame as MuJoCo wrote it: the physical normal.
    case none
    /// The geom-2-side record reads -n: MIMo `get_touch_obs` on a frame MuJoCo just wrote,
    /// and env.py `_contacts` on such a frame (the trainer's `_collect` under mechanical_v4).
    case geom2Side
    /// The geom-1-side record reads -n: env.py `_contacts` right after `get_touch_obs` on the
    /// same frame (`_collect` without physics-step reads; the WI-14 exporter).
    case geom1Side
}

/// Reads every MuJoCo contact that touches a skin body and converts it into `SkinContact`
/// records: the port of `BridgeEnv._contacts` (env.py:2305-2356).
public struct ContactReader {
    /// The force below which a contact is treated as inactive: |fn| < 1e-9 N
    /// (env.py:2319; touch.py:1549 uses the same line).
    public static let inactiveForceThreshold: Double = 1e-9

    /// The ids of the bodies with skin points (the 53 patch bodies of the shipped skin
    /// table; MIMo also gives the two eye bodies one placeholder point each, and a caller
    /// that wants their records includes them here).
    public let skinBodyIds: Set<Int>
    /// The infant's root body (`model.body_rootid` of the `hip` body, env.py:1201): a
    /// contact's other body is "self" when its root body is this one.
    public let ownRootBodyId: Int

    /// Makes a reader for the given skin bodies and the infant's root body.
    public init(skinBodyIds: Set<Int>, ownRootBodyId: Int) {
        self.skinBodyIds = skinBodyIds
        self.ownRootBodyId = ownRootBodyId
    }

    /// INPUT: the physics engine after a step (contacts, forces and poses current) and the
    /// negation rule (`.none`: the physical normal on every record). OPERATION: the
    /// per-contact arithmetic of the file header. OUTPUT: the records, contact index order,
    /// geom1's body before geom2's body.
    public func read(_ physics: ContactPhysicsReading, negation: FrameNegation = .none) -> [SkinContact] {
        var out: [SkinContact] = []
        let count = physics.contactCount
        guard count > 0 else { return out }
        for index in 0..<count {
            let contact = physics.contact(at: index)
            let body1 = physics.bodyId(ofGeom: contact.geom1)
            let body2 = physics.bodyId(ofGeom: contact.geom2)
            // env.py:2314: [b for b in (b1, b2) if touch.has_sensors(b)]
            let sensing = [body1, body2].filter { skinBodyIds.contains($0) }
            if sensing.isEmpty { continue }
            let force = physics.contactForce(at: index)
            let fn = force.normal
            if abs(fn) < ContactReader.inactiveForceThreshold { continue }
            // env.py:2322: the force on geom2 from geom1, world frame
            let fWorld = contact.normal * fn + contact.tangent1 * force.tangent1
                + contact.tangent2 * force.tangent2
            let posWorld = contact.position
            let penetration = max(0.0, -contact.distance)
            // env.py:2337-2340: velocity at the contact point of each body, world frame
            let velocity1 = physics.pointVelocity(ofBody: body1, atWorldPoint: posWorld)
            let velocity2 = physics.pointVelocity(ofBody: body2, atWorldPoint: posWorld)
            // file header: the frame read negated on one record of a qualifying contact
            let qualifies = contact.distance < 0 && skinBodyIds.contains(body1)
            for body in sensing {
                let isGeom2Side = (body == body2)
                let physicalInto = isGeom2Side ? contact.normal : -contact.normal
                let negated: Bool
                switch negation {
                case .none: negated = false
                case .geom2Side: negated = qualifies && isGeom2Side
                case .geom1Side: negated = qualifies && !isGeom2Side
                }
                let nInto = negated ? -physicalInto : physicalInto
                let fOnBody = isGeom2Side ? fWorld : -fWorld
                let tangWorld = fOnBody - nInto * fOnBody.dot(nInto)
                let rotation = physics.bodyRotation(body)
                let tangBody = rotation.applyTransposed(tangWorld)
                let normalBody = rotation.applyTransposed(nInto)
                let forceBody = rotation.applyTransposed(fOnBody)
                // touch.py:1394-1414: body-frame position, moved to the surface
                var posBody = rotation.applyTransposed(posWorld - physics.bodyPosition(body))
                if contact.distance < 0 {
                    posBody = posBody + normalBody * (contact.distance / 2.0)
                }
                // env.py:2340-2347: relative velocity of the other body against this one
                var rel = velocity1 - velocity2
                if !isGeom2Side { rel = -rel }
                let closing = max(0.0, rel.dot(nInto))
                let tangentVelocity = rel - nInto * rel.dot(nInto)
                let tangentSpeed = tangentVelocity.norm
                let frictionPower = max(0.0, tangWorld.dot(rel))
                let other = isGeom2Side ? body1 : body2
                out.append(SkinContact(
                    bodyId: body, geom1: contact.geom1, geom2: contact.geom2,
                    normalForce: abs(fn), tangentialBody: tangBody,
                    positionBody: posBody, positionWorld: posWorld,
                    normalWorld: nInto, normalBody: normalBody, forceBody: forceBody,
                    penetration: penetration, closingSpeed: closing,
                    tangentSpeed: tangentSpeed, frictionPower: frictionPower,
                    otherBodyName: physics.bodyName(other),
                    otherIsSelf: physics.rootBodyId(ofBody: other) == ownRootBodyId))
            }
        }
        return out
    }
}
