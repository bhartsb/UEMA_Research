// UEMAPlanner.swift: the placeholder source of the UEMAPlanner module (work item WI-17).
//
// It keeps the target non-empty until the work item named below adds the module's
// types; it stays as the module's name constant afterwards. Nothing in this file
// was compiled on the machine that wrote it (no Swift toolchain on Windows).

/// The goal ring, the planner and the hand descent (work item WI-39): the goal
/// ring keeps the last 200 model steps' tokens with their wellness; every 10 model
/// steps once it holds 20 entries the planner runs Adam at learning rate 0.1 for
/// 20 updates on one command through the frozen forward model (21 evaluations)
/// and keeps the best-scored command; the hand descent runs 20 steps over 250 ms
/// on the finger joints (intention_generator.py:271, 298; lem_model.py:4252).
public enum UEMAPlannerModule {
    /// The module's name, equal to its SwiftPM target name; the placeholder test
    /// of this module checks it.
    public static let name = "UEMAPlanner"
}
