// UEMASenses.swift: the placeholder source of the UEMASenses module (work item WI-17).
//
// It keeps the target non-empty until the work item named below adds the module's
// types; it stays as the module's name constant afterwards. Nothing in this file
// was compiled on the machine that wrote it (no Swift toolchain on Windows).

/// The six sense readers over Physics (work items WI-27, WI-28, WI-29):
/// proprioception 26 rows x 12 numbers, motor 26 rows x 10 numbers, vestibular
/// 29 numbers, tactile (contact forces spread over the 4,683 skin points and
/// pooled to 256 patches x 10 fields), vision (five cameras rendered every 2nd
/// model step through SceneKit) and audio (per apple/UEMAKit/Specs/audio_render.md,
/// WI-16). Each reader implements UEMACore's SenseReader protocol.
public enum UEMASensesModule {
    /// The module's name, equal to its SwiftPM target name; the placeholder test
    /// of this module checks it.
    public static let name = "UEMASenses"
}
