// UEMACommand.swift: the placeholder source of the UEMACommand module (work item WI-17).
//
// It keeps the target non-empty until the work item named below adds the module's
// types; it stays as the module's name constant afterwards. Nothing in this file
// was compiled on the machine that wrote it (no Swift toolchain on Windows).

/// The command decoder (work item WI-37): one command (26 handle rows x 9 numbers)
/// into 92 motor controls, each clipped to -1..1, per
/// apple/UEMAKit/Specs/command_decoder.md (WI-16; action_decoder.py, 527 lines),
/// plus the body-function modulation of actuator gain and force range
/// (env.py:2840-2845); written to Physics.
public enum UEMACommandModule {
    /// The module's name, equal to its SwiftPM target name; the placeholder test
    /// of this module checks it.
    public static let name = "UEMACommand"
}
