// GoalRingTests.swift: the goal ring and the highest-wellness goal rule (work item WI-39).
// Nothing in this file was compiled on the machine that wrote it (no Swift toolchain on
// Windows). Pass lines from apple/UEMAKit/Specs/planner_loss.md section 1 (ring of 200,
// argmax of wellness) and torch.argmax's first-maximum rule.

import XCTest
import UEMACore
@testable import UEMAPlanner

final class GoalRingTests: XCTestCase {
    private func entry(_ step: Int, _ wellness: Float) -> GoalRingEntry {
        GoalRingEntry(step: ModelStep(count: step), tokens: [.tactile: Token(repeating: Float(step))], wellness: wellness)
    }

    /// The released ring keeps 200 entries (lem_model.py:998).
    func testDefaultCapacityIsTwoHundred() {
        XCTAssertEqual(GoalRing().capacity, 200)
    }

    /// Appending to a full ring drops the oldest entry, as deque(maxlen=200) does.
    func testFullRingDropsTheOldestEntry() {
        let ring = GoalRing(capacity: 3)
        for step in 1 ... 5 {
            ring.append(entry(step, Float(step)))
        }
        XCTAssertEqual(ring.count, 3)
        XCTAssertTrue(ring.isFull)
        XCTAssertEqual(ring.entries.map { $0.step.count }, [3, 4, 5])
    }

    /// The goal is the entry with the highest wellness.
    func testGoalIsTheHighestWellnessEntry() throws {
        let ring = GoalRing()
        for (step, wellness) in [(1, 0.2), (2, 0.9), (3, -0.4), (4, 0.5)] as [(Int, Float)] {
            ring.append(entry(step, wellness))
        }
        let goal = try XCTUnwrap(ring.goal(by: HighestWellnessGoalSelection()))
        XCTAssertEqual(goal.index, 1)
        XCTAssertEqual(goal.entry.step.count, 2)
        XCTAssertEqual(goal.entry.tokens[.tactile], Token(repeating: 2))
    }

    /// Among equal maxima the oldest entry wins (torch.argmax returns the first maximum).
    func testTieGoesToTheOldestEntry() {
        let entries = [entry(1, 0.7), entry(2, 0.7), entry(3, 0.1)]
        XCTAssertEqual(HighestWellnessGoalSelection().goalIndex(in: entries), 0)
    }

    /// Non-finite wellness never becomes the goal; an all-NaN ring has no goal.
    func testNonFiniteWellnessIsSkipped() {
        let rule = HighestWellnessGoalSelection()
        XCTAssertEqual(rule.goalIndex(in: [entry(1, .nan), entry(2, -3), entry(3, .infinity)]), 1)
        XCTAssertNil(rule.goalIndex(in: [entry(1, .nan)]))
        XCTAssertNil(GoalRing().goal(by: rule))
    }

    /// Rank 1 is the highest wellness; equal values share a rank.
    func testWellnessRank() {
        let ring = GoalRing()
        for (step, wellness) in [(1, 0.2), (2, 0.9), (3, 0.9), (4, -1)] as [(Int, Float)] {
            ring.append(entry(step, wellness))
        }
        XCTAssertEqual(ring.wellnessRank(ofEntryAt: 1), 1)
        XCTAssertEqual(ring.wellnessRank(ofEntryAt: 2), 1)
        XCTAssertEqual(ring.wellnessRank(ofEntryAt: 0), 3)
        XCTAssertEqual(ring.wellnessRank(ofEntryAt: 3), 4)
        XCTAssertNil(ring.wellnessRank(ofEntryAt: 4))
    }

    /// removeAll empties the ring.
    func testRemoveAllEmptiesTheRing() {
        let ring = GoalRing()
        ring.append(entry(1, 1))
        ring.removeAll()
        XCTAssertEqual(ring.count, 0)
    }
}
