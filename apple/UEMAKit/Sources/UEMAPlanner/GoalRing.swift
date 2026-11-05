// GoalRing.swift: the goal ring (the memory of the last 200 model steps the planner picks
// its goal from) and the rule that picks the goal entry. Work item WI-39, target B of
// docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html (specification REQ-B-12; the
// specification's section 5 names the concrete type WellnessGoalRing).
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). Plain Swift over UEMACore: no MLX symbol, so the ring
// and its tests build without the Metal library. The "First build on a Mac" checklist of
// this work item is in Planner.swift's header.
//
// SOURCES READ BEFORE WRITING (November 4, 2026), each with the fact it gave (paths under
// D:/UEMA_Research):
// - apple/UEMAKit/Specs/planner_loss.md section 1: "The ring is a `deque(maxlen=200)` of
//   `(tokens {sense: [B, 192]}, wball [B])` appended every model step (lem_model.py:998,
//   :4271-4272)"; "A planning event runs when the goal ring holds at least 20 entries and
//   `step % 10 == 0` (lem_model.py:4283)"; the goal index per body is "`best_t = argmax over
//   T of ranked[T, B]`" with `goal_state[m] = ring tokens[best_t[b]][m]` (lem_model.py:4314-4317).
// - apple/UEMAKit/Sources/UEMACore/ModelStep.swift: `ModelStep.goalRingLength` 200 and
//   `ModelStep.goalRingMinimumForPlanning` 20.
// - apple/UEMAKit/Sources/UEMACore/Protocols.swift: `Planning.record(tokens:wellness:at:)`
//   hands one scalar wellness per model step (the wellness channel sum of WI-36).
// - PyTorch's documentation of torch.argmax: "If there are multiple maximal values then the
//   indices of the first maximal value are returned." (the tie rule below).
//
// WHAT IS NOT PORTED HERE. The released run ranks ring entries by `wellness_rank
// normalized` (planner_loss.md section 1: each wellness channel centred on its running mean,
// divided by its spread, weighted and summed, with entries whose `harm` row exceeds 0.92 set
// to minus infinity). The Planning protocol hands this ring one number per model step, so the
// ranking here is the argmax of that number (the plan row's "the highest-wellness entry is
// the goal"). A ranking over the 14 channels is another `GoalSelecting` type that the session
// injects; it needs the channel record, which the Planning protocol does not carry.

import UEMACore

// MARK: - One entry

/// One goal ring entry: the six encoder tokens of one model step (192 numbers each) and the
/// wellness score of that model step (the wellness channel sum).
public struct GoalRingEntry: Equatable, Sendable {
    /// The model step the tokens were encoded at.
    public let step: ModelStep

    /// The encoder tokens of that model step, one per active sense.
    public let tokens: TokenSet

    /// The wellness channel sum of that model step.
    public let wellness: Float

    /// An entry as given.
    public init(step: ModelStep, tokens: TokenSet, wellness: Float) {
        self.step = step
        self.tokens = tokens
        self.wellness = wellness
    }
}

// MARK: - The goal rule

/// Picks the goal entry from the goal ring's entries (oldest first). Returns nil when no
/// entry qualifies.
public protocol GoalSelecting {
    /// The index into `entries` of the goal entry, or nil.
    func goalIndex(in entries: [GoalRingEntry]) -> Int?
}

/// The goal is the entry with the highest wellness. Entries whose wellness is not finite
/// (NaN or plus or minus infinity) are skipped; among equal maxima the oldest entry wins
/// (torch.argmax returns the first maximal index).
public struct HighestWellnessGoalSelection: GoalSelecting {
    /// The rule has no settings.
    public init() {}

    public func goalIndex(in entries: [GoalRingEntry]) -> Int? {
        var best: Int?
        var bestValue = -Float.infinity
        for (index, entry) in entries.enumerated() where entry.wellness.isFinite {
            if best == nil || entry.wellness > bestValue {
                best = index
                bestValue = entry.wellness
            }
        }
        return best
    }
}

// MARK: - The ring

/// The goal ring: the last `capacity` model steps' tokens with their wellness, oldest first.
/// Appending to a full ring drops the oldest entry, as `deque(maxlen=200)` does
/// (lem_model.py:998). The memory it holds at 200 entries: 200 x 6 x 192 x 4 bytes =
/// 921,600 bytes of tokens plus 200 wellness numbers (specification REQ-B-15).
public final class GoalRing {
    /// Entries the ring keeps: 200 by default (`ModelStep.goalRingLength`, lem_model.py:998).
    public let capacity: Int

    /// The entries, oldest first; at most `capacity`.
    public private(set) var entries: [GoalRingEntry] = []

    /// An empty ring keeping `capacity` entries (1 or more; 200 in the released run).
    public init(capacity: Int = ModelStep.goalRingLength) {
        precondition(capacity >= 1, "a goal ring keeps 1 or more entries; got \(capacity)")
        self.capacity = capacity
        entries.reserveCapacity(capacity)
    }

    /// The entries held now, 0 to `capacity`.
    public var count: Int { entries.count }

    /// True when the ring holds `capacity` entries.
    public var isFull: Bool { entries.count == capacity }

    /// Appends one model step's entry; at capacity the oldest entry is dropped first.
    public func append(_ entry: GoalRingEntry) {
        if entries.count == capacity {
            entries.removeFirst()
        }
        entries.append(entry)
    }

    /// Appends one model step's tokens and wellness.
    public func append(tokens: TokenSet, wellness: Float, at step: ModelStep) {
        append(GoalRingEntry(step: step, tokens: tokens, wellness: wellness))
    }

    /// The goal entry under `rule`, with its index among `entries`; nil for an empty ring or
    /// when no entry qualifies.
    public func goal(by rule: any GoalSelecting) -> (index: Int, entry: GoalRingEntry)? {
        guard let index = rule.goalIndex(in: entries), entries.indices.contains(index) else {
            return nil
        }
        return (index, entries[index])
    }

    /// The rank of the entry at `index` by wellness, 1 for the highest; nil for an index
    /// outside the ring or a non-finite wellness. The plan panel shows it beside the goal's
    /// wellness (specification REQ-B-14).
    public func wellnessRank(ofEntryAt index: Int) -> Int? {
        guard entries.indices.contains(index), entries[index].wellness.isFinite else { return nil }
        let value = entries[index].wellness
        return 1 + entries.filter { $0.wellness.isFinite && $0.wellness > value }.count
    }

    /// Empties the ring.
    public func removeAll() {
        entries.removeAll(keepingCapacity: true)
    }
}
