// UEMAPhysics.swift: the placeholder source of the UEMAPhysics module (work item WI-17).
//
// It keeps the target non-empty until the work item named below adds the module's
// types; it stays as the module's name constant afterwards. Nothing in this file
// was compiled on the machine that wrote it (no Swift toolchain on Windows).

/// MuJoCoPhysics over the CMuJoCo module (work item WI-25): loads the scene
/// bundle with mj_loadXML, writes the 90 motor controls clipped to the
/// actuator bounds (-1..1 narrowed to each motor's force range), runs
/// one model step as 5 env steps of 2 x (mj_step + mj_rnePostConstraint) at
/// the scene's 0.005 s timestep (10 physics steps of 5 ms; mimo_env.py:573-575,
/// env.py:1157), and reads qpos, qvel, sensordata, xpos, xquat, actuator_force
/// and the contacts (ncon, mj_contactForce).
public enum UEMAPhysicsModule {
    /// The module's name, equal to its SwiftPM target name; the placeholder test
    /// of this module checks it.
    public static let name = "UEMAPhysics"
}
