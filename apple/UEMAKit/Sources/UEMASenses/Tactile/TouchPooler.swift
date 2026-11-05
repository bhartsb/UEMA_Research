// TouchPooler.swift: the third and fourth tactile stages of the UEMA Apple port (work item
// WI-28, target B; specification REQ-B-5 and section 5: `TouchPooler`, `PointCloudBuilder`;
// contract apple/UEMAKit/Specs/touch_pooler.md sections 2, 3 and 6).
//
// WHAT THIS FILE IS. `GoertzelTouchPooler`, the port of `TouchPooler` and `Goertzel`
// (Source/Python/mimo_bridge/touch_adapter.py:137-176 and :179-358): per env step (10 ms,
// `dt` 0.01 s) it turns the point forces of SkinSpread.swift and the contact records of
// ContactReader.swift into 10 fields per patch (one patch = one skin body, 53 in the
// released world), the two self-touch columns and the five push columns.
// `TactileFieldNorms`, the port of `tokenize_touch` (Source/Python/tokenizer/modalities/
// touch.py:75-76, :109-117): the ten fields over `_FIELD_NORMS`, padded to 256 slots.
// `PressureSortedPointCloud`, the port of `touch_point_cloud`
// (Source/Python/mimo_bridge/lem_model.py:220-264): the [32, 19] tactile token.
// `TactileReader` joins the four stages for one body.
//
// THE TEN FIELDS (touch_adapter.py:6-16; fixture field_rules): 0 normal pressure Pa = sum of
// normal force on the patch / contact area; 1 shear Pa = |sum of tangential force| / area;
// 2 slip rate mm/s; 3 contact area mm2 = touched points x one point's area (spacing^2);
// 4..7 vibration N^2 = Goertzel power of the patch's normal-force history at 5, 10, 15 and
// 20 Hz over 100 samples (1 s); 8, 9 stretch ratios = 1.0 (no soft skin).
//
// THE TOKEN (lem_model.py:92-96, :221-264). Columns 0-2: the patch body's position relative
// to the root body (hip), metres / 0.5; 3-10: the first 8 normalised fields; 11: the share
// of the patch's force from the infant's own parts (0 to 1); 12: distinct own parts on the
// patch / 4, capped at 1; 13-15: the strongest contact's normal in the body frame; 16: its
// depth, cm / 0.1; 17: its arrival speed, cm/s / 2.0; 18: 1.0 for a filled row. Rows are the
// active patches sorted by column 3 (normalised pressure) descending, 32 at most, the rest 0.
//
// CADENCE. The Python pooler updates once per env step (10 ms; env.py:3120 in `_collect`)
// and the token is built per model step (50 ms = 5 env steps) from the last fields; the
// vibration history therefore holds 100 env steps = 1 s. Under the released contact profile
// mechanical_v4 the point forces are recomputed after every physics step (env.py:2358-2360)
// and `_collect` reuses the last of them (env.py:3116-3117). The Swift `TactileReader` keeps
// both: `readPhysicsStep` ten times and `poolEnvStep` five times per model step (or
// `readEnvStep` five times for a caller without physics-step reads), then `token` once.
//
// NOT PORTED HERE (touch_pooler.md section 4): the `mechanical_v4` virtual skin
// (virtual_skin.py, contact_footprint.py) that overwrites columns 0-3 of the live trainer's
// patches with Hertz-footprint pressures, and the body-function quantisation of columns 0-1.
// This file reproduces `TouchPooler.update` itself, which is what the WI-14 fixture
// tactile_patches.json records ("these are the TouchPooler's own fields, the quantity
// WI-28 reproduces", export_eye_fixtures.py:422-426).
//
// SOLID. One responsibility per type (pooler, normaliser, token builder, reader). Open for
// extension: another band set or window is another `Goertzel`; another ordering rule is
// another `PointCloudBuilder`. Dependency inversion: `TactileReader` takes the four stages
// as protocols at init; nothing here constructs a concrete physics type.
//
// THE TACTILE FILES (Sources/UEMASenses/Tactile/). ContactReader.swift (stage 1, contacts
// into body-frame records); SkinSpread.swift (stage 2, the skin table and the spread; point
// forces stored as float32 like MIMo's); TouchPooler.swift (stages 3 and 4, this file);
// PhysicsContactView.swift (UEMACore's PhysicsReading as the contact reader's input, with
// the injected geom_bodyid / body_rootid / names and the contact-point velocity rule);
// TactileSense.swift (UEMACore's SenseReader: physics-step reads, env-step pools, the
// [n, 10] reading with its [n, 10] side columns, and the [32, 19] token from a reading);
// SkinLayoutAdapter.swift (UEMAScene's SkinLayout as this module's SkinTable).
//
// FIRST BUILD ON A MAC (this work item's acceptance; plan section 8, step 4). After the
// package resolves (Package.swift header, steps 1 to 3):
//  1. `xcodebuild -scheme UEMAKit-Package build -destination 'platform=macOS'` : exit code 0.
//     Constructs to check first if UEMASenses fails to compile: `SkinTable` is declared once,
//     in SkinSpread.swift (UEMAScene names its own protocol `SkinLayoutReading`,
//     SkinLayout.swift:433-457, and SkinLayoutAdapter.swift wraps `SkinLayout` as a
//     `SkinTable`); the plan's WI-26 row still says UEMAScene declares `SkinTable`. The
//     static stored values of non-Sendable types (`RigidPointVelocity.atBodyOrigin`,
//     `Vec3.zero`, `Mat3.identity`, `TactileToken.empty`) compile in the Swift 5 language
//     mode of swift-tools-version 5.9; Swift 6 mode would ask for `Sendable`.
//  2. `xcodebuild -scheme UEMAKit-Package test -destination 'platform=macOS'
//     -only-testing:UEMASensesTests` : exit code 0. The Tactile* test files beside
//     UEMASensesPackageTests.swift must all pass:
//     - TactileSkinSpreadTests: one synthetic contact on a three-point line spreads to the
//       hand-computed weights 4/7, 2/7, 1/7 (scale 0.01 m, distances 0, 0.01, 0.015 m);
//       the all-weights-zero case gives the single nearest point; a vertex unreachable
//       through the neighbour lists is excluded; the 1.5 search margin bridges a vertex
//       at 3.5 scale to one at 2.5 scale; a hand-built submesh with mismatched array
//       counts or an out-of-range neighbour is refused.
//     - TactileTouchPoolerTests: pressure 15,000 Pa, shear 20,000 Pa, area 200 mm2 from
//       the worked example of mimo_bridge/tests/test_touch_adapter.py:4-18; slip 200 mm/s;
//       a 10 Hz sine of amplitude 2 N gives band power 4.0 N^2; the self and push columns.
//     - TactileSenseTests: the contact reader's frame arithmetic on a worked contact; the
//       three negation rules on a worked self-touch with 0.002 m penetration; 0 contacts
//       give 32 zero rows through the full sense cadence (10 physics-step reads, 5 env-step
//       pools, one read), and a reading before any env step gives them too; one 10 N
//       contact gives the hand-computed token row; three active patches sort by pressure
//       with the valid flag last.
//     - TactileFixtureTests: for each of the three WI-14 states, the skin table's digest
//       recomputes to 0581069917fae467; `SkinLayoutTable` over UEMAScene's `SkinLayout`
//       gives the same bodies, scales, live positions and neighbour lists as the JSON
//       decoded by `ShippedSkinTable`; the spreader run on the fixture's contacts (with
//       MIMo's in-place frame negation undone, see that file's header) matches the
//       fixture's point forces; the pooler run on the fixture's point forces and contacts
//       matches its 53 x 10 fields. Each test prints
//       "WI-28 measured difference <name> <state>: <max absolute difference>"; copy the
//       largest of each name into plan section 9 as this item's pass lines (REQ-B-5 names
//       1e-4 per normalised field as the line; the spread line 1e-5 N is provisional).
//       Measured before the first build, November 4, 2026, in numpy on this machine: the
//       same spread rule (search, weights, nearest-point fallback, float32 storage, the
//       negation undone) over the shipped skin table gave maximum point-force difference
//       0.0 N against the fixture on all three states, and the Python `TouchPooler` fed the
//       fixture's point forces and contacts the way TactileFixtureTests feeds the Swift
//       pooler gave maximum normalised field difference 0.0 against `fields`.
//  3. The tactile reading meets UEMATokenizer's `TactileTokenizer` (WI-30) only in the
//     integration target: for one `TactileSense.read` output, `TactileTokenizer().tokenize`
//     and `PressureSortedPointCloud().build(reading:)` must give the same [32, 19] numbers
//     (both read [n, 10] fields and [n, 10] side columns; TactileTokenizer.swift:6-15).
//     Add that comparison to UEMAIntegrationTests when WI-40 wires the readers.
//  4. If TactileFixtureTests fails on the spread while the pooler passes: the Python
//     candidate cache (SkinSpread.swift header, WHY THE RESULT CAN DIFFER) or JSONDecoder's
//     parse of a coordinate is the first suspect; the failure message names the first
//     worst point index and its two forces; read them before changing code.
//  5. If `import CryptoKit` fails on the iOS simulator slice, CryptoKit is available from
//     iOS 13 and macOS 10.15 (https://developer.apple.com/documentation/cryptokit/sha256):
//     check the deployment target in Package.swift (iOS 17, macOS 14).
//  6. Record each command, its exit code and the printed differences in
//     apple/FIRST_BUILD_RECORD.md (WI-41).
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain).

import Foundation

// MARK: - Patches

/// One patch = one skin body (`PatchSpec`, touch_adapter.py:80-90).
public struct PatchSpec: Equatable {
    /// The MuJoCo body id.
    public let bodyId: Int
    /// The body's name.
    public let bodyName: String
    /// The body's live skin points (`n_points`).
    public let pointCount: Int
    /// The sensor spacing, metres (`scale_m`).
    public let scale: Double

    /// Makes a patch specification.
    public init(bodyId: Int, bodyName: String, pointCount: Int, scale: Double) {
        self.bodyId = bodyId
        self.bodyName = bodyName
        self.pointCount = pointCount
        self.scale = scale
    }

    /// One point's area, m2: spacing squared (touch_adapter.py:88-90).
    public var pointAreaM2: Double { scale * scale }
    /// One point's area, mm2 (touch_adapter.py:198).
    public var pointAreaMm2: Double { scale * scale * 1e6 }
}

extension SkinTable {
    /// The patch specifications of the table's bodies in id order, as env.py:1174-1176
    /// builds them from `touch.sensing_bodies()`.
    public var patchSpecs: [PatchSpec] {
        bodies.map { PatchSpec(bodyId: $0.bodyId, bodyName: $0.bodyName, pointCount: $0.sensorCount, scale: $0.scale) }
    }
}

// MARK: - Goertzel

/// Power of chosen frequencies in a window of samples (touch_adapter.py:137-176):
/// power = (2 |X_k| / N)^2, so a full-window sine of amplitude A at a band's frequency
/// gives A^2 (N^2 when the samples are forces in N).
public struct Goertzel: Equatable {
    /// The band frequencies, Hz.
    public let frequenciesHz: [Double]
    /// The sample spacing, s.
    public let dt: Double
    /// The window length in samples.
    public let window: Int
    /// The (possibly fractional) bin index per band: f x window x dt (touch_adapter.py:151).
    public let binIndices: [Double]
    /// 2 pi k / window per band (touch_adapter.py:152).
    public let omegas: [Double]
    /// 2 cos(omega) per band (touch_adapter.py:153).
    public let coefficients: [Double]

    /// Makes the analyser; at dt 0.01 s and window 100 the bands 5, 10, 15 and 20 Hz are
    /// integer bins 5, 10, 15 and 20.
    public init(frequenciesHz: [Double], dt: Double, window: Int) {
        self.frequenciesHz = frequenciesHz
        self.dt = dt
        self.window = window
        let k = frequenciesHz.map { $0 * Double(window) * dt }
        binIndices = k
        omegas = k.map { 2.0 * Double.pi * $0 / Double(window) }
        coefficients = omegas.map { 2.0 * cos($0) }
    }

    /// INPUT: `samples`, oldest first. OPERATION: for each band the recurrence
    /// s = v + coeff s1 - s2 over the samples, then real = s1 - s2 cos(omega),
    /// imag = s2 sin(omega), power = 4 (real^2 + imag^2) / n^2 (touch_adapter.py:155-176).
    /// OUTPUT: one power per band; all zero for an empty window.
    public func power(_ samples: [Double]) -> [Double] {
        let n = samples.count
        var out = [Double](repeating: 0.0, count: frequenciesHz.count)
        if n == 0 { return out }
        for i in 0..<frequenciesHz.count {
            var sPrev = 0.0
            var sPrev2 = 0.0
            let c = coefficients[i]
            for v in samples {
                let s = v + c * sPrev - sPrev2
                sPrev2 = sPrev
                sPrev = s
            }
            let w = omegas[i]
            let real = sPrev - sPrev2 * cos(w)
            let imag = sPrev2 * sin(w)
            out[i] = 4.0 * (real * real + imag * imag) / Double(n * n)
        }
        return out
    }
}

// MARK: - The pooler's output

/// One update's result: the ten fields per patch and the columns kept beside them
/// (`last_fields`, `last_self`, `last_contact`, touch_adapter.py:338-346), as float32 like
/// the Python arrays.
public struct TouchFields: Equatable {
    /// [patchCount][10]: pressure Pa, shear Pa, slip mm/s, area mm2, vibration x4 N^2,
    /// stretch x2 (1.0).
    public let fields: [[Float]]
    /// [patchCount][2]: own-part force share (0 to 1), distinct own parts / 4 (0 to 1).
    public let selfColumns: [[Float]]
    /// [patchCount][5]: the strongest contact's normal in the body frame (3), depth
    /// cm / 0.1, arrival speed cm/s / 2.0.
    public let contactColumns: [[Float]]
    /// [patchCount]: friction power / area, W/m2 (touch_adapter.py:322-323).
    public let frictionPowerDensity: [Double]
    /// Sum over patches of the force from the infant's own parts, N (`last_self_force`).
    public let selfForceTotal: Double
    /// Sum over patches of the normal force, N (`last_total_force`).
    public let totalForce: Double
    /// The number of distinct geom pairs among the self-touch contacts (`last_self_pairs`).
    public let selfPairCount: Int

    /// Makes a result; arrays are stored as given.
    public init(fields: [[Float]], selfColumns: [[Float]], contactColumns: [[Float]],
                frictionPowerDensity: [Double], selfForceTotal: Double, totalForce: Double,
                selfPairCount: Int) {
        self.fields = fields
        self.selfColumns = selfColumns
        self.contactColumns = contactColumns
        self.frictionPowerDensity = frictionPowerDensity
        self.selfForceTotal = selfForceTotal
        self.totalForce = totalForce
        self.selfPairCount = selfPairCount
    }

    /// The number of patches.
    public var patchCount: Int { fields.count }
}

/// Sums point forces and contacts into patch fields, keeping the state the slip and
/// vibration columns need between updates.
public protocol TouchPooler: AnyObject {
    /// The patches in body id order.
    var specs: [PatchSpec] { get }
    /// The number of patches.
    var patchCount: Int { get }
    /// The update spacing, s (0.01 for the released world).
    var dt: Double { get }
    /// The patch row of a body, or nil for a body without a patch.
    func patchRow(ofBody bodyId: Int) -> Int?
    /// One update, in step order; `includeVibration` false skips the Goertzel columns.
    func update(pointForces: SkinPointForces, contacts: [SkinContact], includeVibration: Bool) -> TouchFields
    /// Returns the pooler to its state at construction: zero history, no stored contacts,
    /// update count 0 (a fresh `TouchPooler(specs, dt)` of touch_adapter.py:186-210).
    func reset()
}

/// What can go wrong when a pooler is built.
public enum TouchPoolerError: Error, Equatable {
    /// More skin bodies than the frame holds patch slots (touch_adapter.py:192-193).
    case tooManyPatches(count: Int, maximum: Int)
}

/// The port of `TouchPooler` (touch_adapter.py:179-358). Specification section 5:
/// `GoertzelTouchPooler`.
public final class GoertzelTouchPooler: TouchPooler {
    /// `N_PATCH_FIELDS` (touch_adapter.py:33).
    public static let fieldCount = 10
    /// `VIB_BANDS_HZ` (touch_adapter.py:34).
    public static let vibrationBandsHz: [Double] = [5.0, 10.0, 15.0, 20.0]
    /// `DEFAULT_VIB_WINDOW` samples (touch_adapter.py:35): 1 s at dt 0.01 s.
    public static let defaultVibrationWindow = 100
    /// `ACTIVE_FORCE_EPS`, N (touch_adapter.py:36).
    public static let activeForceThreshold: Double = 1e-6
    /// `SKIN_SCALE_MAX`, m (touch_adapter.py:68): a body at or above it forms no patch.
    public static let skinScaleMax: Double = 0.5
    /// `PUSH_DEPTH_SCALE_CM` (touch_adapter.py:104).
    public static let pushDepthScaleCm: Double = 0.1
    /// `PUSH_ARRIVAL_SCALE_CMS` (touch_adapter.py:105).
    public static let pushArrivalScaleCmPerS: Double = 2.0
    /// `TOUCH_MAX_IMPULSES` (tokenizer/config.py:51).
    public static let maximumPatches = 256
    /// The stored-position slip match gate, m (touch_adapter.py:304: 0.03).
    public static let slipMatchDistance: Double = 0.03

    /// The patches: bodies with scale under `skinScaleMax`, sorted by body id
    /// (touch_adapter.py:190-191).
    public let specs: [PatchSpec]
    /// The update spacing, s.
    public let dt: Double
    /// The vibration window, samples.
    public let vibrationWindow: Int
    /// The band analyser.
    public let goertzel: Goertzel
    /// One point's area per patch, mm2 (touch_adapter.py:198).
    public let pointAreaMm2: [Double]
    /// The last update's result; nil before the first update.
    public private(set) var lastFields: TouchFields?

    private let rowOfBody: [Int: Int]
    /// `_hist`: [patch][window] normal-force sums, oldest first (touch_adapter.py:201).
    private var history: [[Double]]
    /// `_last_contact_pos` (touch_adapter.py:207).
    private var lastContactPositions: [ContactPairKey: [Vec3]] = [:]
    /// `_contact_first_step` (touch_adapter.py:208).
    private var contactFirstStep: [ContactPairKey: Int] = [:]
    /// `_step` (touch_adapter.py:209).
    private var step = 0

    /// The number of patches.
    public var patchCount: Int { specs.count }

    /// touch_adapter.py:186-210. Throws `tooManyPatches` above `maximumPatches`.
    public init(specs: [PatchSpec], dt: Double,
                vibrationWindow: Int = GoertzelTouchPooler.defaultVibrationWindow,
                bandsHz: [Double] = GoertzelTouchPooler.vibrationBandsHz,
                maximumPatches: Int = GoertzelTouchPooler.maximumPatches) throws {
        let kept = specs.filter { $0.scale < GoertzelTouchPooler.skinScaleMax }
            .sorted { $0.bodyId < $1.bodyId }
        if kept.count > maximumPatches {
            throw TouchPoolerError.tooManyPatches(count: kept.count, maximum: maximumPatches)
        }
        self.specs = kept
        self.dt = dt
        self.vibrationWindow = max(0, vibrationWindow)
        goertzel = Goertzel(frequenciesHz: bandsHz, dt: dt, window: self.vibrationWindow)
        pointAreaMm2 = kept.map { $0.pointAreaMm2 }
        var rows: [Int: Int] = [:]
        for (i, spec) in kept.enumerated() { rows[spec.bodyId] = i }
        rowOfBody = rows
        history = Array(repeating: [Double](repeating: 0.0, count: self.vibrationWindow), count: kept.count)
    }

    /// The patch row of a body, or nil.
    public func patchRow(ofBody bodyId: Int) -> Int? {
        rowOfBody[bodyId]
    }

    /// Zero history, no stored contact positions or first steps, update count 0, no last
    /// result: the state `init` leaves.
    public func reset() {
        history = Array(repeating: [Double](repeating: 0.0, count: vibrationWindow), count: specs.count)
        lastContactPositions = [:]
        contactFirstStep = [:]
        step = 0
        lastFields = nil
    }

    /// touch_adapter.py:213-346 for one update. INPUT: the point forces (N, body frame)
    /// and this update's contact records. OUTPUT: the fields and columns (file header).
    public func update(pointForces: SkinPointForces, contacts: [SkinContact],
                       includeVibration: Bool = true) -> TouchFields {
        let n = patchCount
        let columns = GoertzelTouchPooler.fieldCount
        var fields = Array(repeating: [Double](repeating: 0.0, count: columns), count: n)
        for p in 0..<n {
            fields[p][8] = 1.0
            fields[p][9] = 1.0
        }
        // contact area from the active skin points (touch_adapter.py:225-236)
        var activeCounts = [Double](repeating: 0.0, count: n)
        for (bodyId, list) in pointForces.forces {
            guard let p = rowOfBody[bodyId], !list.isEmpty else { continue }
            activeCounts[p] = Double(list.reduce(0) { $0 + ($1.norm > GoertzelTouchPooler.activeForceThreshold ? 1 : 0) })
        }
        let areaMm2 = (0..<n).map { activeCounts[$0] * pointAreaMm2[$0] }
        let areaM2 = areaMm2.map { $0 * 1e-6 }
        // sums per patch from the contacts (touch_adapter.py:239-313)
        var normalSum = [Double](repeating: 0.0, count: n)
        var tangentialSum = [Vec3](repeating: Vec3.zero, count: n)
        var slipMax = [Double](repeating: 0.0, count: n)
        var frictionPower = [Double](repeating: 0.0, count: n)
        var selfForce = [Double](repeating: 0.0, count: n)
        var selfParts = [Set<String>](repeating: [], count: n)
        var strongest = [Double](repeating: 0.0, count: n)
        var contactColumns = Array(repeating: [Double](repeating: 0.0, count: 5), count: n)
        var selfPairKeys = Set<GeomPair>()
        var seenPairs = Set<ContactPairKey>()
        var newPositions: [ContactPairKey: [Vec3]] = [:]
        for c in contacts {
            guard let p = rowOfBody[c.bodyId] else { continue }
            let force = max(0.0, c.normalForce)
            if force >= strongest[p] {
                // the strongest contact's push: direction in the body frame, depth and arrival
                strongest[p] = force
                contactColumns[p][0] = c.normalBody.x
                contactColumns[p][1] = c.normalBody.y
                contactColumns[p][2] = c.normalBody.z
                contactColumns[p][3] = c.penetration * 100.0 / GoertzelTouchPooler.pushDepthScaleCm
                contactColumns[p][4] = (c.closingSpeed ?? 0.0) * 100.0 / GoertzelTouchPooler.pushArrivalScaleCmPerS
            }
            if c.otherIsSelf {
                selfForce[p] += force
                selfParts[p].insert(c.otherBodyName)
                selfPairKeys.insert(c.pairKey.geomPair)
            }
            normalSum[p] += force
            tangentialSum[p] = tangentialSum[p] + c.tangentialBody
            if let tangentSpeed = c.tangentSpeed {
                slipMax[p] = max(slipMax[p], tangentSpeed * 1000.0)
            }
            if let power = c.frictionPower {
                frictionPower[p] += power
            }
            let pair = c.pairKey
            seenPairs.insert(pair)
            let position = c.positionBody
            if c.tangentSpeed == nil, dt > 0, var stored = lastContactPositions[pair], !stored.isEmpty {
                // touch_adapter.py:300-306: the nearest stored point of the same pair
                var best = 0
                var bestDistance = (position - stored[0]).norm
                for (i, q) in stored.enumerated() {
                    let distance = (position - q).norm
                    if distance < bestDistance {
                        bestDistance = distance
                        best = i
                    }
                }
                if bestDistance < GoertzelTouchPooler.slipMatchDistance {
                    slipMax[p] = max(slipMax[p], bestDistance / dt * 1000.0)
                    stored.remove(at: best)
                    lastContactPositions[pair] = stored
                }
            }
            newPositions[pair, default: []].append(position)
            if contactFirstStep[pair] == nil { contactFirstStep[pair] = step }
        }
        lastContactPositions = newPositions
        contactFirstStep = contactFirstStep.filter { seenPairs.contains($0.key) }
        // pressure, shear, slip, area (touch_adapter.py:315-323)
        var frictionPowerDensity = [Double](repeating: 0.0, count: n)
        for p in 0..<n {
            if areaM2[p] > 0 {
                fields[p][0] = normalSum[p] / areaM2[p]
                fields[p][1] = tangentialSum[p].norm / areaM2[p]
                frictionPowerDensity[p] = frictionPower[p] / areaM2[p]
            }
            fields[p][2] = slipMax[p]
            fields[p][3] = areaMm2[p]
        }
        // vibration (touch_adapter.py:326-330)
        if vibrationWindow > 0 {
            for p in 0..<n {
                history[p].removeFirst()
                history[p].append(normalSum[p])
            }
        }
        if includeVibration {
            for p in 0..<n where history[p].contains(where: { $0 != 0.0 }) {
                let power = goertzel.power(history[p])
                for (b, value) in power.enumerated() where 4 + b < 8 {
                    fields[p][4 + b] = value
                }
            }
        }
        step += 1
        // self-touch columns (touch_adapter.py:338-345)
        var selfColumns = Array(repeating: [Double](repeating: 0.0, count: 2), count: n)
        for p in 0..<n {
            selfColumns[p][0] = normalSum[p] > 0 ? selfForce[p] / max(normalSum[p], 1e-12) : 0.0
            selfColumns[p][1] = Double(min(selfParts[p].count, 4)) / 4.0
        }
        let result = TouchFields(
            fields: fields.map { $0.map { Float($0) } },
            selfColumns: selfColumns.map { $0.map { Float($0) } },
            contactColumns: contactColumns.map { $0.map { Float($0) } },
            frictionPowerDensity: frictionPowerDensity,
            selfForceTotal: selfForce.reduce(0.0, +),
            totalForce: normalSum.reduce(0.0, +),
            selfPairCount: selfPairKeys.count)
        lastFields = result
        return result
    }

    /// How long a contact pair has been seen, s (touch_adapter.py:348-352); 0 when unseen.
    public func contactDuration(of key: ContactPairKey) -> Double {
        guard let first = contactFirstStep[key] else { return 0.0 }
        return Double(step - first) * dt
    }

    /// How many updates a contact pair has been seen (touch_adapter.py:354-358).
    public func contactUpdateCount(of key: ContactPairKey) -> Int {
        guard let first = contactFirstStep[key] else { return 0 }
        return step - first
    }
}

// MARK: - Normalisation

/// What can go wrong when the fields become a token.
public enum TactileTokenError: Error, Equatable {
    /// More patch rows than the frame's slots (frame_encoder.py:119-120 refuses them too).
    case tooManyPatches(count: Int, slots: Int)
    /// A row with a column count other than 10.
    case wrongColumnCount(row: Int, count: Int)
}

/// The tokenizer's divisors (`_FIELD_NORMS`, tokenizer/modalities/touch.py:75) and the
/// stretch re-zeroing (`_STRETCH_REST`, :76), applied as `tokenize_touch` does (:109-117).
/// Work item WI-30's tactile tokenizer applies the same rule; this type holds the constants.
public enum TactileFieldNorms {
    /// `_FIELD_NORMS`: pressure 100 kPa, shear 50 kPa, slip 200 mm/s, area 20,000 mm2,
    /// vibration 10 N^2 x4, stretch 0.5 x2.
    public static let divisors: [Float] = [100_000, 50_000, 200, 20_000, 10, 10, 10, 10, 0.5, 0.5]
    /// `_STRETCH_REST`: the raw stretch ratio at rest, subtracted from columns 8 and 9.
    public static let stretchRest: Float = 1.0
    /// `TOUCH_MAX_IMPULSES`: the frame's patch slots (tokenizer/config.py:51).
    public static let slotCount = 256

    /// INPUT: [n][10] raw fields (float32 as the frame stores them). OPERATION: columns 8
    /// and 9 minus 1.0, every column over its divisor, zero rows appended up to `slots`.
    /// OUTPUT: [slots][10]; throws above `slots` rows or on a row of another width.
    public static func normalise(_ fields: [[Float]], slots: Int = TactileFieldNorms.slotCount) throws -> [[Float]] {
        if fields.count > slots {
            throw TactileTokenError.tooManyPatches(count: fields.count, slots: slots)
        }
        var out = Array(repeating: [Float](repeating: 0.0, count: divisors.count), count: slots)
        for (r, row) in fields.enumerated() {
            guard row.count == divisors.count else {
                throw TactileTokenError.wrongColumnCount(row: r, count: row.count)
            }
            for c in 0..<divisors.count {
                let value = c >= 8 ? row[c] - stretchRest : row[c]
                out[r][c] = value / divisors[c]
            }
        }
        return out
    }
}

// MARK: - The token

/// The tactile token of one model step: 32 rows x 19 columns (`TOUCH_TOKENS`,
/// `TOUCH_TOKEN_F`, lem_model.py:93-94), float32.
public struct TactileToken: Equatable {
    /// `TOUCH_TOKENS`: rows.
    public static let rowCount = 32
    /// `TOUCH_TOKEN_F`: columns.
    public static let columnCount = 19
    /// The column of the valid flag (the last).
    public static let validColumn = 18

    /// [32][19].
    public let rows: [[Float]]

    /// Makes a token; rows are stored as given.
    public init(rows: [[Float]]) {
        self.rows = rows
    }

    /// All zero: no active patch.
    public static let empty = TactileToken(
        rows: Array(repeating: [Float](repeating: 0.0, count: TactileToken.columnCount), count: TactileToken.rowCount))

    /// The number of rows whose valid flag is 1.
    public var activeRowCount: Int {
        rows.reduce(0) { $0 + ($1.count > TactileToken.validColumn && $1[TactileToken.validColumn] == 1.0 ? 1 : 0) }
    }

    /// The token as one array of 32 x 19 = 608 numbers, row-major.
    public var flattened: [Float] { rows.flatMap { $0 } }
}

/// Turns the 256 normalised patch slots into the [32, 19] token.
public protocol PointCloudBuilder {
    /// INPUT: `normalisedSlots` [256][10] from `TactileFieldNorms.normalise`;
    /// `patchPositions` one position per patch row (metres, relative to the root body), nil
    /// when unknown; `selfColumns` [n][2] and `contactColumns` [n][5] from the pooler, nil
    /// when absent. OUTPUT: the token.
    func build(normalisedSlots: [[Float]], patchPositions: [Vec3]?,
               selfColumns: [[Float]]?, contactColumns: [[Float]]?) -> TactileToken
}

/// The port of `touch_point_cloud` (lem_model.py:221-264). Specification section 5:
/// `PressureSortedPointCloud`.
public struct PressureSortedPointCloud: PointCloudBuilder {
    /// `TOUCH_POS_SCALE`, m (lem_model.py:96): positions are divided by it.
    public static let positionScale: Float = 0.5

    /// Makes a builder.
    public init() {}

    /// lem_model.py:231-264. Rows with any nonzero column among the 10, sorted by column 0
    /// descending (ties kept in slot order, where torch.argsort gives no fixed order), the
    /// first 32 kept; columns as the file header lists them.
    public func build(normalisedSlots: [[Float]], patchPositions: [Vec3]?,
                      selfColumns: [[Float]]?, contactColumns: [[Float]]?) -> TactileToken {
        var out = TactileToken.empty.rows
        let n: Int
        if let positions = patchPositions {
            n = min(normalisedSlots.count, positions.count)
        } else {
            n = normalisedSlots.count
        }
        var active: [Int] = []
        for i in 0..<n where normalisedSlots[i].reduce(Float(0), { $0 + abs($1) }) > 0 {
            active.append(i)
        }
        if active.isEmpty { return TactileToken(rows: out) }
        let ordered = active.sorted { a, b in
            let pa = normalisedSlots[a].isEmpty ? Float(0) : normalisedSlots[a][0]
            let pb = normalisedSlots[b].isEmpty ? Float(0) : normalisedSlots[b][0]
            if pa != pb { return pa > pb }
            return a < b
        }
        let kept = Array(ordered.prefix(TactileToken.rowCount))
        let highest = kept.max() ?? 0
        for (k, i) in kept.enumerated() {
            if let positions = patchPositions {
                out[k][0] = Float(positions[i].x) / PressureSortedPointCloud.positionScale
                out[k][1] = Float(positions[i].y) / PressureSortedPointCloud.positionScale
                out[k][2] = Float(positions[i].z) / PressureSortedPointCloud.positionScale
            }
            let row = normalisedSlots[i]
            for j in 0..<8 where j < row.count {
                out[k][3 + j] = row[j]
            }
            if let columns = selfColumns, columns.count > highest, columns[i].count >= 2 {
                out[k][11] = columns[i][0]
                out[k][12] = columns[i][1]
            }
            if let columns = contactColumns, columns.count > highest, columns[i].count >= 5 {
                for j in 0..<5 {
                    out[k][13 + j] = columns[i][j]
                }
            }
            out[k][TactileToken.validColumn] = 1.0
        }
        return TactileToken(rows: out)
    }
}

// MARK: - The reader

/// The tactile sense of one body: contacts read, spread, pooled and turned into the token.
/// Specification section 5: `TactileSense` over the four REQ-B-5 stages, each injected.
///
/// With `reproducesFrameNegation` true the contact records and the spread reproduce MIMo's
/// in-place normal negation (ContactReader.swift header) as the Python trainer meets it:
/// the physics-step spread and the pooled records of `poolEnvStep` after a physics-step read
/// use `FrameNegation.geom2Side`; an env step without a physics-step read (`readEnvStep`)
/// spreads with `.geom2Side` and pools records read with `.geom1Side`. With it false every
/// record uses the physical normal (`.none`).
public final class TactileReader {
    /// The skin table the spreader and the patch specifications come from.
    public let table: SkinTable
    /// Stage 1: contacts from the physics engine.
    public let contactReader: ContactReader
    /// Stage 2: forces over skin points.
    public let spreader: ContactSpreader
    /// Stage 3: patch fields.
    public let pooler: TouchPooler
    /// Stage 4: the token.
    public let pointCloud: PointCloudBuilder
    /// The root body whose position the patch positions are measured from (the `hip`,
    /// env.py:1223-1229).
    public let rootBodyId: Int
    /// The last env step's contact records.
    public private(set) var lastContacts: [SkinContact] = []
    /// The last env step's point forces.
    public private(set) var lastPointForces: SkinPointForces?
    /// The last env step's fields.
    public private(set) var lastFields: TouchFields?
    /// True: the records and the spread reproduce MIMo's in-place normal negation (type
    /// comment); false: the physical normal on every record.
    public let reproducesFrameNegation: Bool
    /// True between a `readPhysicsStep` and the `poolEnvStep` that consumes its forces.
    private var pointForcesFromPhysicsStep = false

    /// Makes a reader from its four stages; `reproducesFrameNegation` false gives the
    /// physical normal on every record.
    public init(table: SkinTable, contactReader: ContactReader, spreader: ContactSpreader,
                pooler: TouchPooler, pointCloud: PointCloudBuilder, rootBodyId: Int,
                reproducesFrameNegation: Bool = false) {
        self.table = table
        self.contactReader = contactReader
        self.spreader = spreader
        self.pooler = pooler
        self.pointCloud = pointCloud
        self.rootBodyId = rootBodyId
        self.reproducesFrameNegation = reproducesFrameNegation
    }

    /// The rule for MIMo `get_touch_obs` and for env.py `_contacts` on a frame MuJoCo just
    /// wrote.
    private var freshFrameNegation: FrameNegation { reproducesFrameNegation ? .geom2Side : .none }
    /// The rule for env.py `_contacts` right after `get_touch_obs` on the same frame.
    private var afterTouchObservationNegation: FrameNegation { reproducesFrameNegation ? .geom1Side : .none }

    /// The released configuration over a table: `ContactReader` over the table's bodies
    /// with the root body as the infant's own root, `LinearContactSpreader`,
    /// `GoertzelTouchPooler` at dt 0.01 s with the 100-sample window, and
    /// `PressureSortedPointCloud`. `ownRootBodyId` is `body_rootid` of the hip (env.py:1201);
    /// nil takes `rootBodyId`, which is the same id when the hip is a direct child of the
    /// world. `reproducesFrameNegation` defaults to true, the records the released weights
    /// were trained on. Throws when the table has more than 256 bodies.
    public static func released(table: SkinTable, rootBodyId: Int, ownRootBodyId: Int? = nil,
                                dt: Double = 0.01,
                                reproducesFrameNegation: Bool = true) throws -> TactileReader {
        let ids = Set(table.bodies.map { $0.bodyId })
        let pooler = try GoertzelTouchPooler(specs: table.patchSpecs, dt: dt)
        return TactileReader(
            table: table,
            contactReader: ContactReader(skinBodyIds: ids, ownRootBodyId: ownRootBodyId ?? rootBodyId),
            spreader: LinearContactSpreader(table: table),
            pooler: pooler,
            pointCloud: PressureSortedPointCloud(),
            rootBodyId: rootBodyId,
            reproducesFrameNegation: reproducesFrameNegation)
    }

    /// One physics step (5 ms): read the contacts and spread them into fresh point forces,
    /// the `touch.get_touch_obs()` that `_mechanical_substep` runs after every physics step
    /// (env.py:2358-2360, called from `_single_mujoco_step`, env.py:666-670; touch.py:1612:
    /// the outputs are zeroed per read). Called 10 times per model step. The pooler is not
    /// updated here: the patch fields are formed once per env step (env.py:3117-3120).
    @discardableResult
    public func readPhysicsStep(_ physics: ContactPhysicsReading) -> SkinPointForces {
        let contacts = contactReader.read(physics, negation: freshFrameNegation)
        let forces = spreader.pointForces(for: contacts, table: table)
        lastContacts = contacts
        lastPointForces = forces
        pointForcesFromPhysicsStep = true
        return forces
    }

    /// One env step (10 ms), after its second physics step: `_collect` (env.py:3113-3120)
    /// reuses the point forces of the last physics step (its `get_touch_obs` is skipped once
    /// a physics-step read exists, env.py:3116-3117), reads the contacts again with the
    /// current velocities (`self._contacts()`, on a frame MuJoCo just wrote) and updates the
    /// pooler. Without a previous physics-step read it does what `readEnvStep` does. Called
    /// 5 times per model step.
    @discardableResult
    public func poolEnvStep(_ physics: ContactPhysicsReading, includeVibration: Bool = true) -> TouchFields {
        guard pointForcesFromPhysicsStep, let previous = lastPointForces else {
            return readEnvStep(physics, includeVibration: includeVibration)
        }
        let contacts = contactReader.read(physics, negation: freshFrameNegation)
        return pool(forces: previous, contacts: contacts, includeVibration: includeVibration)
    }

    /// One env step from a single state, for callers that do not observe physics steps:
    /// `_collect` without a physics-step read (env.py:3116-3119) runs `get_touch_obs` (the
    /// spread; records read with the fresh-frame rule) and then `_contacts` on the same
    /// frame (records read with the after-`get_touch_obs` rule).
    @discardableResult
    public func readEnvStep(_ physics: ContactPhysicsReading, includeVibration: Bool = true) -> TouchFields {
        let spreadContacts = contactReader.read(physics, negation: freshFrameNegation)
        let forces = spreader.pointForces(for: spreadContacts, table: table)
        let contacts = contactReader.read(physics, negation: afterTouchObservationNegation)
        return pool(forces: forces, contacts: contacts, includeVibration: includeVibration)
    }

    /// The pooler update shared by the two env-step paths; stores the last reads.
    private func pool(forces: SkinPointForces, contacts: [SkinContact], includeVibration: Bool) -> TouchFields {
        let fields = pooler.update(pointForces: forces, contacts: contacts, includeVibration: includeVibration)
        lastContacts = contacts
        lastPointForces = forces
        lastFields = fields
        pointForcesFromPhysicsStep = false
        return fields
    }

    /// Clears the pooler's state and the last reads (a world reset).
    public func reset() {
        pooler.reset()
        lastContacts = []
        lastPointForces = nil
        lastFields = nil
        pointForcesFromPhysicsStep = false
    }

    /// The patch positions `touch_pos = xpos[patch bodies] - xpos[root]` (env.py:3554),
    /// metres, one per patch row; float32 in Python, so each component is rounded to Float
    /// here before the builder divides it.
    public func patchPositions(_ physics: ContactPhysicsReading) -> [Vec3] {
        let root = physics.bodyPosition(rootBodyId)
        return pooler.specs.map { spec in
            let d = physics.bodyPosition(spec.bodyId) - root
            return Vec3(Double(Float(d.x)), Double(Float(d.y)), Double(Float(d.z)))
        }
    }

    /// The [32, 19] token of the last env step read (lem_model.py:221-264 over the
    /// normalised fields, the patch positions and the pooler's columns); the empty token
    /// before any read. Throws when the pooler has more than 256 patches.
    public func token(_ physics: ContactPhysicsReading) throws -> TactileToken {
        guard let fields = lastFields else { return TactileToken.empty }
        let slots = try TactileFieldNorms.normalise(fields.fields)
        return pointCloud.build(
            normalisedSlots: slots, patchPositions: patchPositions(physics),
            selfColumns: fields.selfColumns, contactColumns: fields.contactColumns)
    }
}
