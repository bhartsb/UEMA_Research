// UEMAModel.swift: the placeholder source of the UEMAModel module (work item WI-17).
//
// It keeps the target non-empty until the work item named below adds the module's
// types; it stays as the module's name constant afterwards. Nothing in this file
// was compiled on the machine that wrote it (no Swift toolchain on Windows).

import Foundation

/// The released model in MLX Swift (work items WI-31, WI-35, WI-36): the
/// safetensors loader checked against Resources/weight_manifest.json (written by
/// apple/scripts/dump_weight_manifest.py, WI-12; 18,131,745 parameters), the six
/// encoders (one 192-number token each), the forward model (4 transformer blocks,
/// 8 heads, width 192, 768 tokens), the prediction heads, the motor controller
/// and the wellness sum.
public enum UEMAModelModule {
    /// The module's name, equal to its SwiftPM target name; the placeholder test
    /// of this module checks it.
    public static let name = "UEMAModel"

    /// The bundle that holds this module's resources, copied as is from
    /// Sources/UEMAModel/Resources; read the manifest with
    /// `resourceBundle.url(forResource: "weight_manifest", withExtension: "json",
    /// subdirectory: "Resources")`.
    public static var resourceBundle: Bundle { Bundle.module }
}
