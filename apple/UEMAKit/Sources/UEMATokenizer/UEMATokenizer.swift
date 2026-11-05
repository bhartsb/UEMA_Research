// UEMATokenizer.swift: the placeholder source of the UEMATokenizer module (work item WI-17).
//
// It keeps the target non-empty until the work item named below adds the module's
// types; it stays as the module's name constant afterwards. Nothing in this file
// was compiled on the machine that wrote it (no Swift toolchain on Windows).

/// The six tokenizers (work item WI-30): fixed divisors that turn each sense's
/// raw arrays into the encoder inputs (positions / 200 cm, velocities / 500 cm/s,
/// energy / 1000, tactile fields / _FIELD_NORMS; vision cut into 16 x 16 x 3
/// patches of 768 numbers plus 4 angle numbers each; plan section 2, stage 3).
public enum UEMATokenizerModule {
    /// The module's name, equal to its SwiftPM target name; the placeholder test
    /// of this module checks it.
    public static let name = "UEMATokenizer"
}
