// Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
// Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
//
// AudioReader.swift: the audio sense's SenseReader and the crib's hearing it renders, a port of
// `CribAudio.render` (Source/Python/mimo_bridge/audio_adapter.py:146-291; contract in
// apple/UEMAKit/Specs/audio_render.md). Written November 4, 2026 for the cross-module consistency
// pass (item 1 of docs/apple_port/HANDOFF_2026-11-04.md): before it, UEMASenses held no audio
// reader and UEMATokenizer's AudioTokenizer accepted three auxiliary layouts.
//
// THE READING (one layout, settled here; like every reader's, raw, before the tokenizer's scales):
//   primary   [256, 2] float32, row = frequency bin (geomspace 40..8000 Hz), column = ear (left,
//             right): the summed linear magnitude of every sounding source (`p_l`, `p_r`,
//             audio_adapter.py:222-269). AudioTokenizer maps it to the encoder's 0..1 scale,
//             clip((10 log10(magnitude + 1e-9) + 80) / 80, 0, 1) (audio_adapter.py:275-277), which
//             `CribAudio.render` also returns as `spectrum` for the fixture check.
//   auxiliary [6] float32, the adapter's raw metadata (audio_adapter.py:285-290): left_rms,
//             right_rms, ild_db (dB), itd_ms (ms), azimuth (radians, folded to the front),
//             front_back. AudioTokenizer divides it into the encoder's spatial pair
//             [ild_db / 20, itd_ms / 1.0] (tokenizer/modalities/audio.py:71-73, sliced to 2 at
//             mimo_bridge/lem_model.py:2476-2477; weight_manifest.json audio.spatial_proj.weight
//             [192, 2]).
//
// TIMING. The trainer renders once per env step (env.py:3542-3546, dt = 0.01 s passed live) and
// the model reads the frame of the last env step of the 5 (audio_render.md section 4). So this
// reader renders in observeEnvStep and read(at:) returns the newest render; before any env step
// (the reset state) read renders once itself. The listener is camera eye_left: head = its
// position, right = column 0 of its rotation matrix, forward = minus column 2 (env.py:3528-3529,
// 3542-3543).
//
// WHAT IS PORTED, AND WHAT IS NOT.
//   Ported exactly: the render (ears, the distance law 0.02 / max(r, 0.2)^1.5, the head shadow,
//   the loudest-source metadata with the azimuth fold, the dB scale), the impact, chime and squeak
//   transients with their gates, decay and spectra, the bee's comb with its wobble walk and 5 Hz
//   tremolo, the dragonfly's flutter pulse. The checks against the Python render are
//   AudioReaderTests (fixture Tests/UEMASensesTests/Fixtures/audio, written by
//   export_audio_fixture.py beside it).
//   Through injected protocols, not ported: what makes a sound happen. The impact trigger (the
//   16 strongest collisions of an env step with frame_count <= 1 and |impulse| > 0.15 N s,
//   env.py:3523-3527, from touch_adapter.py build_collisions), the mobile's chime edge
//   (env.py:3097-3111), the kick panel's squeak (env.py:1614-1639) and the bee and dragonfly
//   visitor state machine (env.py:1548-1593) feed `AudioSourceProviding`; the default
//   `SilentAudioSources` gives none, so a session without them hears silence: primary 0 in every
//   bin (the tokenizer's 10 log10(0 + 1e-9) = -90 dB, below the -80 dB floor, so 0 on its scale)
//   and auxiliary zeros.
//   Not ported: the caregiver's speech (speech.py SpeechStream, clips from caregiver_speech_dir)
//   and the dragonfly's recorded timbre (dragonfly_loop.wav, audio_adapter.py:87-96): the flutter
//   uses the adapter's synthesis fallback shape unless `flutterShape` is given.
//   The wobble's Gaussian draws: numpy's default_rng(7).standard_normal() stream (PCG64 and its
//   ziggurat) is not reimplemented; `GaussianDrawing` takes the draws, and the default
//   `SeededGaussian` is a different deterministic stream, so a bee's f0 differs from the Python
//   one by up to the walk's bound, 5 % of 70 Hz.

import Foundation
import UEMACore

// MARK: - Sources

/// What sounds during one env step: new transients, and the two visitors' positions (metres,
/// world frame; nil when that visitor is absent, or for the bee when it has landed).
public struct AudioSourceFrame: Equatable, Sendable {
    /// A transient started this env step.
    public enum Event: Equatable, Sendable {
        /// A body contact at `position` with impulse magnitude `impulse` N s (CribAudio.impact).
        case impact(position: [Double], impulse: Double)
        /// The mobile struck at `position` (CribAudio.chime).
        case chime(position: [Double])
        /// The kick panel's squeaker at `position` (CribAudio.squeak).
        case squeak(position: [Double])
    }

    /// The transients started this env step, in sequence.
    public var events: [Event]
    /// The bee's position, or nil.
    public var buzzPosition: [Double]?
    /// The dragonfly's position, or nil.
    public var flutterPosition: [Double]?

    /// A frame.
    public init(events: [Event] = [], buzzPosition: [Double]? = nil, flutterPosition: [Double]? = nil) {
        self.events = events
        self.buzzPosition = buzzPosition
        self.flutterPosition = flutterPosition
    }

    /// Nothing sounds.
    public static let silent = AudioSourceFrame()
}

/// The source of each env step's sounding events and visitors (this file's header, WHAT IS PORTED).
public protocol AudioSourceProviding: AnyObject {
    /// The frame of the env step that just ran, read from the state after it.
    func sources(after physics: any PhysicsReading) throws -> AudioSourceFrame
}

/// No events and no visitors.
public final class SilentAudioSources: AudioSourceProviding {
    public init() {}
    public func sources(after physics: any PhysicsReading) throws -> AudioSourceFrame { .silent }
}

/// Standard normal draws for the bee's wobble walk.
public protocol GaussianDrawing: AnyObject {
    /// The next draw.
    func next() -> Double
}

/// A deterministic standard normal stream (SplitMix64 into Box-Muller), seeded 7 like the
/// adapter's generator but not the same sequence (this file's header).
public final class SeededGaussian: GaussianDrawing {
    private var state: UInt64
    private var spare: Double?

    public init(seed: UInt64 = 7) {
        state = seed
    }

    private func uniform() -> Double {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        z ^= z >> 31
        return (Double(z >> 11) + 0.5) / 9_007_199_254_740_992.0
    }

    public func next() -> Double {
        if let s = spare {
            spare = nil
            return s
        }
        let r = (-2 * log(uniform())).squareRoot()
        let theta = 2 * Double.pi * uniform()
        spare = r * sin(theta)
        return r * cos(theta)
    }
}

/// Draws given in advance, then zeros; for tests that replay the Python generator's draws.
public final class RecordedGaussian: GaussianDrawing {
    private var draws: [Double]
    private var index = 0

    public init(_ draws: [Double]) {
        self.draws = draws
    }

    public func next() -> Double {
        defer { index += 1 }
        return index < draws.count ? draws[index] : 0
    }
}

// MARK: - The render

/// The crib's hearing (audio_adapter.py CribAudio), one render per env step.
public final class CribAudio {
    /// Frequency bins: 256 (audio_adapter.py:34).
    public static let binCount = 256
    /// Half the distance between the ears: 0.035 m (:36).
    public static let earHalfMetres = 0.035
    /// The speed of sound: 343 m/s (:37).
    public static let speedOfSoundMetresPerSecond = 343.0
    /// The dB floor and span of the 0..1 scale: -80, 80 (:38).
    public static let decibelFloor = -80.0
    public static let decibelSpan = 80.0
    /// Body-contact loudness relative to a raw strike: 0.15 (:44).
    public static let bodyContactGain = 0.15

    /// The bin centres, geomspace(40, 8000, 256) Hz (:78).
    public let frequencies: [Double]
    /// exp(-f / 1500) (:79).
    public let impactShape: [Double]
    /// The flutter's spectral shape: the synthesis fallback (:98-99) unless given.
    public let flutterShape: [Double]

    private let gaussian: GaussianDrawing
    private var wobble = 0.0
    private var transients: [Transient] = []
    /// The adapter's clock, seconds: + dt per render (:159).
    public private(set) var time = 0.0

    private enum Kind { case impact, chime, squeak }

    private struct Transient {
        let kind: Kind
        let position: [Double]
        let start: Double
        let tau: Double
        let gain: Double
    }

    /// numpy.geomspace(start, stop, count): 10 ** linspace(log10(start), log10(stop)) with the
    /// two ends set exactly.
    public static func geomspace(_ start: Double, _ stop: Double, _ count: Int) -> [Double] {
        let a = log10(start), b = log10(stop)
        let step = (b - a) / Double(count - 1)
        var out = (0 ..< count).map { pow(10.0, Double($0) * step + a) }
        out[0] = start
        out[count - 1] = stop
        return out
    }

    /// A render with `gaussian` for the wobble and an optional recorded flutter shape (256 numbers).
    public init(gaussian: GaussianDrawing = SeededGaussian(), flutterShape: [Double]? = nil) {
        let f = CribAudio.geomspace(40, 8000, CribAudio.binCount)
        frequencies = f
        impactShape = f.map { exp(-$0 / 1500) }
        self.flutterShape = flutterShape ?? f.map {
            0.8 * exp(-0.5 * pow(($0 - 180) / 120, 2)) + 0.25 * exp(-0.5 * pow(($0 - 700) / 350, 2))
        }
        self.gaussian = gaussian
    }

    /// A contact sound (audio_adapter.py:111-125): gain clip(impulse / 0.5, 0, 1), nothing below
    /// 0.05, nothing while 3 or more non-squeak transients are younger than 0.2 s.
    public func impact(position: [Double], impulse: Double) {
        let g = Swift.min(Swift.max(impulse / 0.5, 0), 1)
        guard g >= 0.05 else { return }
        let young = transients.filter { $0.kind != .squeak && (time - $0.start) < 0.2 }.count
        guard young < 3 else { return }
        transients.append(Transient(kind: .impact, position: position, start: time, tau: 0.12,
                                    gain: g * CribAudio.bodyContactGain))
    }

    /// The mobile's strike (:127-135).
    public func chime(position: [Double]) {
        transients.append(Transient(kind: .chime, position: position, start: time, tau: 0.22, gain: 1.6))
    }

    /// The kick panel's squeak (:137-143).
    public func squeak(position: [Double]) {
        transients.append(Transient(kind: .squeak, position: position, start: time, tau: 0.28, gain: 3.0))
    }

    /// Applies one frame's events.
    public func apply(_ frame: AudioSourceFrame) {
        for event in frame.events {
            switch event {
            case let .impact(position, impulse): impact(position: position, impulse: impulse)
            case let .chime(position): chime(position: position)
            case let .squeak(position): squeak(position: position)
            }
        }
    }

    /// One frame of hearing (audio_adapter.py:146-291): `magnitudes` [256 x 2] row-major (bin, ear),
    /// linear; `spectrum`, the same on the 0..1 scale as float32 (the Python render's `spec01`);
    /// and the 6 metadata numbers of this file's header.
    public func render(head: [Double], right: [Double], forward: [Double], buzz: [Double]?, flutter: [Double]?,
                       dt: Double = ModelStep.envStepSeconds)
        -> (magnitudes: [Double], spectrum: [Float], metadata: [Double])
    {
        time += dt
        let n = CribAudio.binCount
        let h = CribAudio.earHalfMetres
        let earL = (0 ..< 3).map { head[$0] - h * right[$0] }
        let earR = (0 ..< 3).map { head[$0] + h * right[$0] }
        var pL = [Double](repeating: 0, count: n)
        var pR = [Double](repeating: 0, count: n)
        var sources: [(shape: [Double], position: [Double], gain: Double)] = []

        var kept: [Transient] = []
        for tr in transients {
            let age = time - tr.start
            let a = exp(-age / tr.tau)
            if a < 0.02 { continue }
            kept.append(tr)
            switch tr.kind {
            case .chime:
                let shape = frequencies.map { f in
                    exp(-0.5 * pow((f - 880) / 55, 2)) + 0.45 * a * exp(-0.5 * pow((f - 880 * 2.4) / 55, 2))
                }
                sources.append((shape, tr.position, tr.gain * a))
            case .squeak:
                let c = 500 + 700 * sin(Double.pi * Swift.min(Swift.max(age / 0.25, 0), 1))
                let shape = frequencies.map { f -> Double in
                    var s = 0.0
                    for k in 1 ... 4 {
                        let ck = c * Double(k)
                        s += (1 / pow(Double(k), 0.7)) * exp(-0.5 * pow((f - ck) / (60 + 0.06 * ck), 2))
                    }
                    return s
                }
                sources.append((shape, tr.position, tr.gain * a))
            case .impact:
                sources.append((impactShape, tr.position, tr.gain * a))
            }
        }
        transients = kept

        if let buzz {
            wobble = Swift.min(Swift.max(wobble * 0.995 + 0.006 * gaussian.next(), -0.05), 0.05)
            let f0 = 70 * (1 + wobble)
            let comb = frequencies.map { f -> Double in
                var s = 0.0
                for k in 1 ... 10 {
                    let roll = k <= 4 ? pow(Double(k), 0.9) : pow(Double(k), 2)
                    let fk = f0 * Double(k)
                    s += (1 / roll) * exp(-0.5 * pow((f - fk) / (18 + 0.02 * fk), 2))
                }
                return s * (1 + 1.5 * exp(-0.5 * pow((f - 350) / 220, 2)))
            }
            let am = 1 + 0.25 * sin(2 * Double.pi * 5 * time)
            sources.append((comb, buzz, 0.8 * am))
        }
        if let flutter {
            let pulse = 0.8 + 0.2 * sin(2 * Double.pi * 18 * time)
            sources.append((flutterShape, flutter, 0.3 * pulse))
        }

        func norm(_ v: [Double]) -> Double { (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).squareRoot() }
        func dot(_ a: [Double], _ b: [Double]) -> Double { a[0] * b[0] + a[1] * b[1] + a[2] * b[2] }

        var best: (loud: Double, itd: Double, ild: Double, azimuth: Double, frontBack: Double)?
        for source in sources {
            let v = (0 ..< 3).map { source.position[$0] - head[$0] }
            let r = norm(v)
            let u = v.map { $0 / Swift.max(r, 1e-6) }
            let lateral = dot(u, right)
            let rL = Swift.max(norm((0 ..< 3).map { source.position[$0] - earL[$0] }), 0.05)
            let rR = Swift.max(norm((0 ..< 3).map { source.position[$0] - earR[$0] }), 0.05)
            let attL = 0.02 / pow(Swift.max(rL, 0.2), 1.5)
            let attR = 0.02 / pow(Swift.max(rR, 0.2), 1.5)
            let gl = source.gain * attL * (1 - 0.35 * Swift.max(0, lateral))
            let gr = source.gain * attR * (1 - 0.35 * Swift.max(0, -lateral))
            var loudSum = 0.0
            for i in 0 ..< n {
                pL[i] += gl * source.shape[i]
                pR[i] += gr * source.shape[i]
                loudSum += (gl + gr) * 0.5 * source.shape[i]
            }
            let loud = loudSum / Double(n)
            if best == nil || loud > best!.loud {
                let itd = (rR - rL) / CribAudio.speedOfSoundMetresPerSecond * 1000
                let ild = 20 * log10(Swift.max(gl, 1e-9) / Swift.max(gr, 1e-9))
                let along = dot(u, forward)
                var az = atan2(lateral, Swift.max(along, -1))
                if abs(az) > Double.pi / 2 {
                    az = (az > 0 ? 1.0 : -1.0) * (Double.pi - abs(az))
                }
                best = (loud, itd, ild, az, abs(along))
            }
        }

        var magnitudes = [Double](repeating: 0, count: 2 * n)
        var spectrum = [Float](repeating: 0, count: 2 * n)
        for i in 0 ..< n {
            for (ear, p) in [pL[i], pR[i]].enumerated() {
                magnitudes[2 * i + ear] = p
                let db = 10 * log10(p + 1e-9)
                let scaled = Swift.min(Swift.max((db - CribAudio.decibelFloor) / CribAudio.decibelSpan, 0), 1)
                spectrum[2 * i + ear] = Float(scaled)
            }
        }
        func rms(_ p: [Double]) -> Double {
            var sum = 0.0
            for value in p { sum += value * value }
            return (sum / Double(n)).squareRoot()
        }
        var metadata: [Double] = [rms(pL), rms(pR), 0, 0, 0, 0]
        if let best {
            metadata[2] = best.ild
            metadata[3] = best.itd
            metadata[4] = best.azimuth
            metadata[5] = best.frontBack
        }
        return (magnitudes, spectrum, metadata)
    }
}

// MARK: - The reader

/// The audio SenseReader of this file's header.
public final class AudioReader: SenseReader {
    /// The sense produced: audio.
    public let sense = SenseName.audio
    /// The metadata width of the reading's auxiliary: 6.
    public static let metadataCount = 6

    /// The render.
    public private(set) var audio: CribAudio
    /// The scene the listener camera's pose is read from.
    public let scene: any SceneGeomReading
    /// The sounding events and visitors.
    public let sources: any AudioSourceProviding
    private let makeAudio: () -> CribAudio
    private var last: SenseReading?

    /// A reader listening at camera eye_left of `scene`. `makeAudio` builds the render, again on
    /// reset.
    public init(scene: any SceneGeomReading, sources: any AudioSourceProviding = SilentAudioSources(),
                makeAudio: @escaping () -> CribAudio = { CribAudio() }) {
        self.scene = scene
        self.sources = sources
        self.makeAudio = makeAudio
        self.audio = makeAudio()
    }

    /// One render of the current state, packed.
    private func render(_ physics: any PhysicsReading) throws -> SenseReading {
        let frame = try sources.sources(after: physics)
        audio.apply(frame)
        let pose = try scene.cameraPose(.eyeLeft)
        let m = pose.rotation
        let rendered = audio.render(
            head: pose.position, right: [m[0], m[3], m[6]], forward: [-m[2], -m[5], -m[8]],
            buzz: frame.buzzPosition, flutter: frame.flutterPosition)
        let metadata = rendered.metadata
        return SenseReading(
            sense: .audio,
            primary: try ShapedArray(shape: [CribAudio.binCount, 2], values: rendered.magnitudes.map { Float($0) }),
            auxiliary: try ShapedArray(shape: [AudioReader.metadataCount], values: metadata.map { Float($0) }))
    }

    /// Renders after every env step (env.py:3542).
    public func observeEnvStep(_ physics: any PhysicsReading) throws {
        last = try render(physics)
    }

    /// The render of the last env step; before any, a render of the current state.
    public func read(_ physics: any PhysicsReading, at step: ModelStep) throws -> SenseReading {
        if let last { return last }
        let first = try render(physics)
        last = first
        return first
    }

    /// A fresh render state (clock 0, no transients, wobble 0) and no reading.
    public func reset() {
        audio = makeAudio()
        last = nil
    }
}
