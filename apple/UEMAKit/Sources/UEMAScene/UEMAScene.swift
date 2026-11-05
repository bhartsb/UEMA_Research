// UEMAScene.swift: the placeholder source of the UEMAScene module (work item WI-17).
//
// It keeps the target non-empty until the work item named below adds the module's
// types; it stays as the module's name constant afterwards. Nothing in this file
// was compiled on the machine that wrote it (no Swift toolchain on Windows).

import Foundation

/// The exported scene files in this module's bundle (work item WI-26):
/// Resources/skin_layout_age24_trunk_v1.json (4,683 skin points, written by
/// apple/scripts/export_skin_layout.py, WI-13) and Resources/crib_age24_trunk_v1/
/// (scene.xml and its files, written by apple/scripts/export_scene_bundle.py,
/// WI-15). SceneBundle hands their URLs to Physics and the tactile reader;
/// SkinLayout parses the JSON into arrays.
public enum UEMASceneModule {
    /// The module's name, equal to its SwiftPM target name; the placeholder test
    /// of this module checks it.
    public static let name = "UEMAScene"

    /// The bundle that holds this module's resources, copied as is from
    /// Sources/UEMAScene/Resources; read a file with
    /// `resourceBundle.url(forResource:withExtension:subdirectory: "Resources")`.
    public static var resourceBundle: Bundle { Bundle.module }
}
