// UEMACore.swift: the placeholder source of the UEMACore module (work item WI-17).
//
// It keeps the target non-empty until the work item named below adds the module's
// types; it stays as the module's name constant afterwards. Nothing in this file
// was compiled on the machine that wrote it (no Swift toolchain on Windows).

/// The value types and protocols every other UEMAKit module depends on
/// (work item WI-20 fills it): Command (26 body handles x 9 numbers: target
/// position x y z in cm, orientation as a quaternion of 4 numbers, radius,
/// strength 0 to 1; README.md:19), Token (192 Float; encoder_registry.py:33),
/// SenseName (the six active senses: vision, tactile, audio, proprioception,
/// motor, vestibular), ModelStep (a count of 50 ms model updates), and the
/// protocols Physics, SceneLoading, SenseReader, Tokenizer, WeightsLoading,
/// ForwardModeling, Planning, CommandDecoding and EyeRendering.
public enum UEMACoreModule {
    /// The module's name, equal to its SwiftPM target name; the placeholder test
    /// of this module checks it.
    public static let name = "UEMACore"
}
