// ModelStepTests.swift: the tests of ModelStep.swift (work item WI-20): one model step is
// 50 ms of 5 env steps x 2 physics steps of 0.005 s; the count, the simulated time and the
// planning cadence, and the decode check on the count.
// Not compiled on the machine that wrote it (no Swift toolchain on Windows).

import Foundation
import XCTest
import UEMACore

final class ModelStepTests: XCTestCase {
    /// One model step is 50 ms: 5 env steps of 2 physics steps of 0.005 s.
    func testOneModelStepIs50Milliseconds() {
        XCTAssertEqual(ModelStep.milliseconds, 50)
        XCTAssertEqual(ModelStep.seconds, 0.05, accuracy: 1e-12)
        XCTAssertEqual(ModelStep.envStepsPerModelStep, 5)
        XCTAssertEqual(ModelStep.physicsStepsPerEnvStep, 2)
        XCTAssertEqual(ModelStep.physicsStepsPerModelStep, 10)
        XCTAssertEqual(ModelStep.physicsTimestepSeconds, 0.005, accuracy: 1e-12)
        XCTAssertEqual(ModelStep.envStepSeconds, 0.010, accuracy: 1e-12)
        XCTAssertEqual(
            ModelStep.physicsTimestepSeconds * Double(ModelStep.physicsStepsPerModelStep),
            ModelStep.seconds, accuracy: 1e-12)
    }

    /// The count advances by one per model step; 20 model steps span 1.0 s (the WI-25 pass line).
    func testCountAndSimulatedTime() {
        let twenty = ModelStep(count: 20)
        XCTAssertEqual(twenty.simulatedSeconds, 1.0, accuracy: 1e-12)
        XCTAssertEqual(twenty.simulatedMilliseconds, 1000)
        XCTAssertEqual(ModelStep.start.count, 0)
        XCTAssertEqual(ModelStep.start.simulatedSeconds, 0, accuracy: 1e-12)
        XCTAssertEqual(ModelStep.start.advanced(by: 3), ModelStep(count: 3))
        XCTAssertEqual(ModelStep(count: 1).advanced(), ModelStep(count: 2))
        XCTAssertLessThan(ModelStep(count: 1), ModelStep(count: 2))
        XCTAssertEqual(ModelStep(count: 7), ModelStep(count: 7))
    }

    /// The planner's cadence is every 10th model step over a goal ring of 200 with 20 needed.
    func testPlanningCadenceAndContext() {
        XCTAssertEqual(ModelStep.planningInterval, 10)
        XCTAssertEqual(ModelStep.goalRingLength, 200)
        XCTAssertEqual(ModelStep.goalRingMinimumForPlanning, 20)
        XCTAssertEqual(ModelStep.contextLength, 128)
        XCTAssertTrue(ModelStep(count: 0).isPlanningCandidate)
        XCTAssertTrue(ModelStep(count: 10).isPlanningCandidate)
        XCTAssertTrue(ModelStep(count: 200).isPlanningCandidate)
        XCTAssertFalse(ModelStep(count: 5).isPlanningCandidate)
        XCTAssertFalse(ModelStep(count: 11).isPlanningCandidate)
    }

    /// A model step count survives an encode and decode; a negative count in a file throws
    /// a DecodingError instead of stopping the process.
    func testDecodingRefusesANegativeCount() throws {
        let step = ModelStep(count: 492_800)
        let data = try JSONEncoder().encode(step)
        XCTAssertEqual(try JSONDecoder().decode(ModelStep.self, from: data), step)
        XCTAssertEqual(try JSONDecoder().decode(ModelStep.self, from: Data(#"{"count":0}"#.utf8)), .start)
        XCTAssertThrowsError(try JSONDecoder().decode(ModelStep.self, from: Data(#"{"count":-1}"#.utf8))) { error in
            XCTAssertTrue(error is DecodingError)
        }
    }
}
