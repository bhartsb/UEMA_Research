// CMuJoCo.c: the placeholder compiled source of the CMuJoCo target (work item WI-17).
//
// CMuJoCo is a C-family SwiftPM target over the binary target "mujoco" (the MuJoCo
// 3.3.0 XCFramework of work item WI-18). Its public headers folder is the target folder
// itself (publicHeadersPath "." in Package.swift), so work item WI-19 places
// module.modulemap and shim.h beside this file; shim.h includes <mujoco/mujoco.h> from
// the XCFramework's headers and the module map names the headers Swift imports.
//
// This file includes no MuJoCo header on purpose: it compiles whether or not the
// XCFramework's headers are on the include path, and it exists only because SwiftPM
// needs at least one compiled source in a C target. No .swift file may be added to
// this folder: at swift-tools-version 5.9 SwiftPM stops with "contains mixed language
// source files; mixed language targets require a tools version of 6.5 or later"
// (the check in TargetSourcesBuilder.swift, the message in PackageBuilder.swift, both of
// swift-package-manager, read November 4, 2026). Until WI-19's module.modulemap exists,
// SwiftPM writes an umbrella-directory module map over this folder (ModuleMapGenerator.swift,
// read the same day: a public headers folder without a custom module map gets
// `.umbrellaDirectory(publicHeadersDir)` whether or not it holds files other than
// headers), and that umbrella covers zero headers, so `import CMuJoCo` compiles to an
// empty module; once module.modulemap exists it is used as is.

/// Returns the target's name. Not declared in any public header, so Swift does not see
/// it; it keeps the target non-empty.
const char *cmujoco_placeholder_name(void) {
    return "CMuJoCo";
}
