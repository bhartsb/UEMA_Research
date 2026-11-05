// shim.h: the one public header of the CMuJoCo module (apple/UEMAKit, work item WI-19
// of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html).
//
// WHAT THIS MODULE IS. CMuJoCo is the C module through which the Swift modules of
// UEMAKit reach MuJoCo 3.3.0, the physics program written in C, which the binary target
// "mujoco" (apple/UEMAKit/Binaries/mujoco.xcframework, written by
// apple/scripts/build_mujoco_xcframework.sh, work item WI-18) holds as one static
// library per platform with its 14 public headers. This header includes MuJoCo's
// umbrella header; module.modulemap beside it names this header as the module's one
// header. A Swift file writes `import CMuJoCo` and calls the C functions and reads the
// C structs directly, with the types Swift's C importer gives them: mjtNum is double
// (include/mujoco/mjtnum.h at tag 3.3.0: "typedef double mjtNum;" unless mjUSESINGLE,
// which no slice defines), so Double; int is Int32; mjtNum* is
// UnsafeMutablePointer<Double>; a fixed-size C array is a tuple; a C enum such as
// mjtCatBit is a struct with a UInt32 rawValue.
//
// Nothing in this file was compiled by a Swift toolchain: the machine that wrote it
// has none (plan section 3, fact 1). The check that ran here is stated in
// Tests/CMuJoCoTests/CMuJoCoSymbolTests.swift: clang 20.1.8 (the one installed with
// Visual Studio) parsed this header and the module map against the 14 headers of tag
// 3.3.0 for the target arm64-apple-macosx14.0, textually and as a clang module.
//
// THE SYMBOLS THE PORT USES. Prototypes copied from include/mujoco/mujoco.h at tag
// 3.3.0; the quoted one-line descriptions are from the API reference,
// https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html (both read
// November 4, 2026). In every line m is the mjModel* and d the mjData*.
//
//   mjModel* mj_loadXML(const char* filename, const mjVFS* vfs, char* error, int error_sz)
//       "Parse XML file in MJCF or URDF format, compile it; return low-level model."
//       INPUT filename: the path of the scene XML (UTF-8 bytes, NUL-terminated);
//       vfs: NULL, so files are read from disk (mjVFS is defined in mjmodel.h);
//       error, error_sz: a byte buffer and its size, filled with the message when
//       the return value is NULL. OUTPUT the model, freed by mj_deleteModel(m).
//   mjData* mj_makeData(const mjModel* m)
//       "Allocate mjData corresponding to given model." Freed by mj_deleteData(d).
//       d->qpos starts at the model's qpos0 (the joint positions written in the XML);
//       the body frames d->xpos and d->xquat hold zeros until mj_forward(m, d)
//       ("Forward dynamics: same as mj_step but do not integrate in time") or the first
//       mj_step (both measured with Python mujoco 3.3.0 on the test scene of
//       Tests/CMuJoCoTests/CMuJoCoSymbolTests.swift, November 4, 2026).
//   void mj_step(const mjModel* m, mjData* d)
//       "Advance simulation, use control callback to obtain external force and
//       control." INPUT d->ctrl (nu doubles; 92 in the crib scene, each -1 to 1).
//       OPERATION one physics step of m->opt.timestep seconds (0.005 s in the crib
//       scene; a model step is 10 of them: 5 environment frames of 10 ms,
//       Source/Python/mimo_bridge/env.py:742, each frame_skip 2 physics steps,
//       env.py:1157 and mimo_env.py:616-619). OUTPUT d->qpos,
//       d->qvel, d->xpos, d->xquat, d->sensordata, d->actuator_force, d->ncon,
//       d->contact, d->time.
//   void mj_rnePostConstraint(const mjModel* m, mjData* d)
//       "Recursive Newton Euler with final computed forces and accelerations."
//       OUTPUT d->cacc, d->cfrc_int, d->cfrc_ext (nbody x 6 doubles each: 3 torque
//       then 3 force). MIMo calls it after every mj_step
//       (third_party/MIMo/mimo/mimo_infant/simulation/envs/mimo_env.py:574-575); it
//       changes no position or velocity.
//   void mj_contactForce(const mjModel* m, const mjData* d, int id, mjtNum result[6])
//       "Extract 6D force:torque given contact id, in the contact frame." INPUT id
//       from 0 to d->ncon - 1. OUTPUT result[0..2] force and result[3..5] torque in
//       the contact frame (d->contact[id].frame; "normal is in [0-2]"), newtons and
//       newton-metres; result[0] is the normal force. The tactile sense calls it once
//       per contact (third_party/MIMo/mimo/mimo_infant/touch/touch.py:1434).
//   void mjv_defaultScene(mjvScene* scn); void mjv_makeScene(const mjModel* m, mjvScene* scn, int maxgeom)
//       "Set default abstract scene." then "Allocate resources in abstract scene."
//       INPUT maxgeom: the geom buffer size. mjv_freeScene(scn) releases the buffer.
//   void mjv_updateScene(const mjModel* m, mjData* d, const mjvOption* opt,
//                        const mjvPerturb* pert, mjvCamera* cam, int catmask, mjvScene* scn)
//       "Update entire scene given model state." INPUT opt after mjv_defaultOption,
//       pert NULL, cam after mjv_defaultCamera, catmask mjCAT_ALL (7: static,
//       dynamic and decorative geoms, mjvisualize.h). OUTPUT scn->ngeom geoms in
//       scn->geoms, each with type, size[3], pos[3], mat[9] and rgba[4]; the
//       SceneKit renderer of work item WI-29 mirrors them. No OpenGL call is made:
//       mjvisualize.h holds data types only.
//   const char* mj_versionString(void)
//       "Return the current version of MuJoCo as a null-terminated string." The
//       acceptance test of this work item expects "3.3.0"; int mj_version(void)
//       returns 330 and the macro mjVERSION_HEADER is 330 (mujoco.h at tag 3.3.0).
//
// THE mjData FIELDS THE SENSES READ (include/mujoco/mjdata.h at tag 3.3.0, the
// comments copied): qpos "position (nq x 1)"; qvel "velocity (nv x 1)"; ctrl
// "control (nu x 1)"; sensordata "sensor data array (nsensordata x 1)"; xpos
// "Cartesian position of body frame (nbody x 3)"; xquat "Cartesian orientation of
// body frame (nbody x 4)"; actuator_force "actuator force in actuation space
// (nu x 1)"; ncon "number of detected contacts"; contact "array of all detected
// contacts (ncon x 1)" of mjContact (geom[2] "geom ids; -1 for flex", pos[3], frame[9],
// dist "distance between nearest points; neg: penetration"). The counts nq, nv, nu,
// nbody, nsensordata and ngeom are fields of mjModel (mjmodel.h). In the crib scene
// nq is 146 and nu is 92 (plan rows WI-25 and WI-37).
//
// WHAT IS NOT USED. The umbrella header also declares the classic OpenGL renderer:
// the 26 functions whose names start with "mjr" in the "OpenGL rendering" section of
// mujoco.h (the count is a grep of the tag 3.3.0 header). No file of this package
// calls one. The specification (REQ-B-1) quotes MuJoCo's own statement that the
// library "can be used on systems without OpenGL support, as long as" the functions
// with that prefix "are not called", and states that the renderer "needs OpenGL
// 1.5, iOS has OpenGL ES through 12.0 only". Tests/CMuJoCoTests/CMuJoCoRendererFreeTests.swift reads every
// source file of the package and fails if that prefix appears in code (the part of a
// line before its first "//").
//
// WHY NO SWIFT FILE IN THIS FOLDER. The plan row names a CMuJoCo.swift here. At
// swift-tools-version 5.9 SwiftPM refuses a target folder holding both .c and .swift
// files (PackageBuilder.swift: "contains mixed language source files; mixed language
// targets require a tools version of 6.5 or later", read November 4, 2026), and this
// folder holds CMuJoCo.c (SwiftPM needs one compiled source in a C target below tools
// version 6.5: "hasHeaderOnlyCModule = self.manifest.toolsVersion >= .v6_5 &&
// hasPublicHeaders"). The Swift-side layer over these symbols therefore belongs in
// Sources/UEMAPhysics (work item WI-25), and Tests/CMuJoCoTests holds the Swift calls
// of every symbol above as its executable example.
//
// FIRST BUILD ON A MAC (plan section 8; this work item's acceptance is step 4)
//  [ ] 1. bash apple/scripts/build_mujoco_xcframework.sh exited 0, so
//         apple/UEMAKit/Binaries/mujoco.xcframework/<slice>/Headers holds
//         module.modulemap and mujoco/*.h (14 files). The include below resolves
//         against that folder only; it is not in git (apple/.gitignore).
//  [ ] 2. cd apple/UEMAKit && swift package resolve : exit 0 (WI-17).
//  [ ] 3. xcodebuild -scheme UEMAKit-Package build -destination 'platform=macOS' :
//         exit 0. If it stops with "'mujoco/mujoco.h' file not found", step 1 did not
//         run or its stage_headers step wrote no mujoco/ subfolder. If it stops with
//         "contains mixed language source files", a .swift file was put in
//         Sources/CMuJoCo: move it to Sources/UEMAPhysics. If it stops at link time on
//         "Undefined symbol: _mj_", the slice's libmujoco.a was not linked: check that
//         Package.swift still lists "mujoco" under CMuJoCo's dependencies and that
//         the XCFramework's Info.plist lists the slice for the destination.
//  [ ] 4. xcodebuild -scheme UEMAKit-Package test -destination 'platform=macOS'
//         -only-testing:CMuJoCoTests : every test passes. The acceptance line is
//         testVersionStringIs330 in CMuJoCoSymbolTests.swift: mj_versionString()
//         equals "3.3.0". The physics tests compare against numbers measured with
//         Python mujoco 3.3.0 on November 4, 2026 (the file states each).
//  [ ] 5. Write the commands, exit codes and clock times (AM/PM Pacific) into
//         apple/FIRST_BUILD_RECORD.md (WI-41); the plan page's WI-19 row moves to
//         VERIFIED or FAILED with the commit hash.

#ifndef UEMAKIT_CMUJOCO_SHIM_H_
#define UEMAKIT_CMUJOCO_SHIM_H_

// The umbrella header of MuJoCo 3.3.0 (include/mujoco/mujoco.h), from the Headers
// folder of the XCFramework slice: it includes mjmodel.h, mjdata.h, mjvisualize.h and
// the other public headers and declares every mj_ and mjv_ function above.
#include <mujoco/mujoco.h>

#endif  // UEMAKIT_CMUJOCO_SHIM_H_
