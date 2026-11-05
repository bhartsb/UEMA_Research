// ViewsTests.swift: XCTest cases for work item WI-38 (UEMAMindApp.swift and Views/), target
// B of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html. They test the views'
// arithmetic and wording types (RunCounterText, EyeImageRaster, TokenHeatScale,
// SensePanelText, CommandTable, PlannerScoreBars, MemoryReadout, WeightsFileCheck), the
// kept weights copy (ApplicationSupportWeightsStore over a temporary folder: the SHA-256
// of "abc" against FIPS 180-2's published digest, a matching file kept, a mismatching file
// refused and deleted, MindAppModel making the session from the kept copy), the
// released-stack builder's refusal, and one and 21 model steps on the DEBUG fake stack
// (the acceptance line "one model step runs on fake physics in a preview", run here
// without the canvas). 19 tests.
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). Compiled by the test targets UEMAMindTests-macOS and
// UEMAMindTests-iOS of apple/project.yml, which compile apple/UEMAMind/Tests/. The app
// module is imported as UEMAMind_macOS or UEMAMind_iOS (PRODUCT_NAME = the target name,
// "-" made "_"); the "First build on a Mac" checklist is in UEMAMindApp.swift's header.

import CoreGraphics
import Foundation
import UEMACore
import XCTest

#if os(macOS)
@testable import UEMAMind_macOS
#elseif os(iOS)
@testable import UEMAMind_iOS
#endif

/// A memory reading with fixed numbers for the tests.
private final class ViewsTestsMemoryMonitor: MemoryStatusReading {
    var availableMemoryBytes: Int? = 2_000 * 1_048_576
    var thermalState: ProcessInfo.ThermalState = .nominal
}

/// A session builder that records the weights URL it was given and then throws
/// `SessionBuildingError.noWeightsFile`, so no session is made.
@MainActor
private final class RecordingSessionBuilder: SessionBuilding {
    var name: String { "recording builder" }
    private(set) var receivedURLs: [URL?] = []

    func makeRunViewModel(
        weightsURL: URL?,
        planningPermit: any PlanningPermitting,
        observers: [any ModelStepObserving]
    ) throws -> RunViewModel {
        receivedURLs.append(weightsURL)
        throw SessionBuildingError.noWeightsFile
    }
}

/// The SHA-256 of the 3 bytes "abc": FIPS 180-2, appendix B.1.
private let abcSHA256 = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

/// A new empty folder under the temporary directory.
private func makeTemporaryFolder() throws -> URL {
    let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("uema_mind_views_tests_" + UUID().uuidString, isDirectory: true)
    try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    return url
}

final class ViewsTests: XCTestCase {
    // MARK: RunControlView

    func testCounterLineNamesModelStepsAndSimulatedSeconds() {
        XCTAssertEqual(
            RunCounterText.line(for: ModelStep(count: 1)),
            "1 model step, 0.05 s of simulated time (50 ms each)")
        XCTAssertEqual(
            RunCounterText.line(for: ModelStep(count: 10)),
            "10 model steps, 0.50 s of simulated time (50 ms each)")
    }

    func testNextPlanningCandidateIsTheNextMultipleOfTenAtOrAboveTheNextCount() {
        // After N completed model steps the next one has count N (the first has count 0).
        XCTAssertEqual(RunCounterText.nextPlanningCandidate(after: ModelStep(count: 0)), 0)
        XCTAssertEqual(RunCounterText.nextPlanningCandidate(after: ModelStep(count: 1)), 10)
        XCTAssertEqual(RunCounterText.nextPlanningCandidate(after: ModelStep(count: 10)), 10)
        XCTAssertEqual(RunCounterText.nextPlanningCandidate(after: ModelStep(count: 11)), 20)
        XCTAssertEqual(RunCounterText.nextPlanningCandidate(after: ModelStep(count: 20)), 20)
    }

    // MARK: EyesView

    func testEyeRasterKeepsWidthHeightAndTwentyFourBitsPerPixel() throws {
        let image = try EyeImage(camera: .eyeLeft, width: 2, height: 1, rgb: [255, 0, 0, 0, 0, 255])
        let cgImage = try XCTUnwrap(EyeImageRaster.cgImage(from: image))
        XCTAssertEqual(cgImage.width, 2)
        XCTAssertEqual(cgImage.height, 1)
        XCTAssertEqual(cgImage.bitsPerPixel, 24)
        XCTAssertEqual(cgImage.bytesPerRow, 6)
    }

    func testEyeCaptionAndHeading() {
        XCTAssertEqual(EyeImageRaster.caption(for: .foveaLeft), "fovea_left, 288 px, 24 degrees")
        XCTAssertEqual(EyeImageRaster.caption(for: .ringLeft), "ring_left, 256 px, 150 degrees")
        XCTAssertEqual(
            EyeImageRaster.heading(renderedAt: ModelStep(count: 12), hasImages: true),
            "Eye images rendered at model step 12")
        XCTAssertEqual(EyeImageRaster.heading(renderedAt: nil, hasImages: true), "Eye images of the latest render")
        XCTAssertEqual(EyeImageRaster.heading(renderedAt: nil, hasImages: false), "No eye render yet")
    }

    // MARK: SensesView

    func testHeatScaleEndsMiddleClipAndNonFinite() {
        func assertComponents(_ value: Float, _ expected: (Double, Double, Double), line: UInt = #line) {
            let c = TokenHeatScale.components(of: value)
            XCTAssertEqual(c.red, expected.0, accuracy: 1e-9, line: line)
            XCTAssertEqual(c.green, expected.1, accuracy: 1e-9, line: line)
            XCTAssertEqual(c.blue, expected.2, accuracy: 1e-9, line: line)
        }
        assertComponents(0, (1, 1, 1))
        assertComponents(3, (1, 0, 0))
        assertComponents(-3, (0, 0, 1))
        assertComponents(6, (1, 0, 0))
        assertComponents(1.5, (1, 0.5, 0.5))
        assertComponents(.nan, (0, 0, 0))
    }

    func testSenseBlocksAreSixInTheActiveSequence() throws {
        let token = Token(repeating: 1)
        let motorRow = SensePanelRow(
            sense: .motor, inputShape: [26, 19], auxiliaryShape: nil, inputMinimum: -2, inputMaximum: 4,
            isFresh: false, token: token, heatMap: SensePanelRow.heatMap(of: token), predictionError: 1)
        let blocks = SensePanelText.blocks([motorRow])
        XCTAssertEqual(blocks.map(\.sense), SenseName.active)
        XCTAssertEqual(blocks.compactMap(\.row).map(\.sense), [.motor])
        XCTAssertNil(try XCTUnwrap(blocks.first { $0.sense == .vision }).row)
        XCTAssertEqual(SensePanelRow.heatMapRows * SensePanelRow.heatMapColumns, Token.count)
        XCTAssertEqual(
            SensePanelText.shapeLine(motorRow),
            "input shape [26, 19], second array none, not refreshed this model step")
    }

    func testShapeAndNumberText() {
        XCTAssertEqual(SensePanelText.shapeText([26, 19]), "[26, 19]")
        XCTAssertEqual(SensePanelText.shapeText(nil), "none")
        XCTAssertEqual(SensePanelText.numberText(nil), "none")
        XCTAssertEqual(SensePanelText.numberText(0.5), "0.5")
    }

    // MARK: CommandView

    @MainActor
    func testCommandTableIsTwentySixRowsOfNineWithThreeDecimals() {
        var command = Command.zero(scale: .centimetres)
        command[.head, .positionX] = 0.12345
        let rows = CommandViewModel.rows(of: command)
        let cells = CommandTable.cells(rows)
        XCTAssertEqual(CommandTable.columnHeaders, CommandViewModel.columnNames)
        XCTAssertEqual(cells.count, Command.handleCount)
        XCTAssertTrue(cells.allSatisfy { $0.count == Command.numbersPerHandle })
        XCTAssertEqual(cells[BodyHandle.head.rawValue][CommandColumn.positionX.rawValue], "0.123")
        XCTAssertEqual(rows.first?.name, "wrist_L")
        XCTAssertEqual(rows[23].name, "gaze")
        XCTAssertEqual(rows.last?.name, "chest")
    }

    // MARK: PlannerView

    func testPlannerBarsFractionsAndSelection() {
        let bars = PlannerScoreBars.make(scores: [2, 1, 1.5], selectedIteration: 1)
        XCTAssertEqual(bars.map(\.iteration), [0, 1, 2])
        XCTAssertEqual(bars[0].fraction, 1, accuracy: 1e-9)
        XCTAssertEqual(bars[1].fraction, 0, accuracy: 1e-9)
        XCTAssertEqual(bars[2].fraction, 0.5, accuracy: 1e-9)
        XCTAssertEqual(bars.map(\.isSelected), [false, true, false])
    }

    func testPlannerBarsEqualAndNonFiniteScoresHaveZeroLength() {
        let bars = PlannerScoreBars.make(scores: [1, 1, .nan], selectedIteration: nil)
        XCTAssertEqual(bars.map(\.fraction), [0, 0, 0])
        XCTAssertFalse(bars.contains { $0.isSelected })
    }

    func testPlannerSummaryNamesTheSelectedEvaluation() {
        let scores = (0 ..< PlanResult.evaluationsPerEvent).map { Float(1.0 - 0.02 * Double($0)) }
        XCTAssertTrue(
            PlannerScoreBars.summary(scores: scores, selectedIteration: 20)
                .hasPrefix("selected update 20 of 20, score "))
        XCTAssertTrue(PlannerScoreBars.summary(scores: scores, selectedIteration: 0).hasPrefix("selected the start"))
        XCTAssertEqual(PlannerScoreBars.summary(scores: [], selectedIteration: nil), "no scores")
    }

    // MARK: MemoryView

    func testMemoryReadoutText() {
        XCTAssertEqual(MemoryReadout.text(megabytes: nil), "not reported on this platform")
        XCTAssertEqual(MemoryReadout.text(megabytes: 2_500.4), "2500 MB")
        XCTAssertEqual(MemoryReadout.text(thermalState: .serious), "serious (planning events skipped)")
        XCTAssertEqual(MemoryReadout.text(thermalState: .nominal), "nominal (planning events run)")
    }

    // MARK: WeightsPickerView

    func testWeightsSizeTextAgainstTheReleasedSize() {
        XCTAssertEqual(WeightsFileCheck.expectedBytes, 73_119_988)
        XCTAssertEqual(WeightsFileCheck.sizeText(bytes: 73_119_988), "73,119,988 bytes, the released size")
        XCTAssertEqual(WeightsFileCheck.sizeText(bytes: 1_024), "1,024 bytes; the released file has 73,119,988")
        XCTAssertEqual(WeightsFileCheck.sizeText(bytes: nil), "size unread")
        XCTAssertEqual(WeightsFileCheck.groupedDigits(999), "999")
        XCTAssertEqual(WeightsFileCheck.groupedDigits(1_000), "1,000")
        XCTAssertEqual(WeightsFileCheck.expectedSHA256.count, 64)
    }

    func testSHA256OfABCMatchesThePublishedDigestAtEveryChunkSize() throws {
        let folder = try makeTemporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let file = folder.appendingPathComponent("abc.bin")
        try Data("abc".utf8).write(to: file)
        XCTAssertEqual(try WeightsFileCheck.sha256Hex(of: file), abcSHA256)
        XCTAssertEqual(try WeightsFileCheck.sha256Hex(of: file, chunkBytes: 1), abcSHA256)
        XCTAssertEqual(try WeightsFileCheck.sha256Hex(of: file, chunkBytes: 2), abcSHA256)
    }

    func testStoreKeepsAMatchingFileAndRefusesAMismatch() throws {
        let folder = try makeTemporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let source = folder.appendingPathComponent("picked.safetensors")
        try Data("abc".utf8).write(to: source)
        let storeFolder = folder.appendingPathComponent("store", isDirectory: true)

        let refusing = ApplicationSupportWeightsStore(folder: storeFolder, expectedSHA256: String(repeating: "0", count: 64))
        XCTAssertThrowsError(try refusing.importWeights(from: source)) { error in
            XCTAssertEqual(
                error as? WeightsFileError,
                .sha256Mismatch(expected: String(repeating: "0", count: 64), found: abcSHA256))
        }
        XCTAssertNil(refusing.storedWeightsURL())
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: storeFolder.appendingPathComponent("model.safetensors.partial").path))

        let accepting = ApplicationSupportWeightsStore(folder: storeFolder, expectedSHA256: abcSHA256.uppercased())
        XCTAssertNil(accepting.storedWeightsURL())
        let kept = try accepting.importWeights(from: source)
        XCTAssertEqual(kept.lastPathComponent, ApplicationSupportWeightsStore.fileName)
        XCTAssertEqual(accepting.storedWeightsURL(), kept)
        XCTAssertEqual(try Data(contentsOf: kept), Data("abc".utf8))
        // A second pick replaces the kept copy.
        XCTAssertEqual(try accepting.importWeights(from: source), kept)
    }

    @MainActor
    func testAppModelBuildsFromTheKeptCopyOnPickAndAtLaunch() throws {
        let folder = try makeTemporaryFolder()
        defer { try? FileManager.default.removeItem(at: folder) }
        let source = folder.appendingPathComponent("picked.safetensors")
        try Data("abc".utf8).write(to: source)
        let store = ApplicationSupportWeightsStore(
            folder: folder.appendingPathComponent("store", isDirectory: true), expectedSHA256: abcSHA256)
        let memory = MemoryViewModel(monitor: ViewsTestsMemoryMonitor())

        let builder = RecordingSessionBuilder()
        let app = MindAppModel(builder: builder, memory: memory, weightsStore: store)
        app.loadKeptWeights()
        XCTAssertTrue(builder.receivedURLs.isEmpty, "no kept copy yet, so no session is attempted")
        app.pickWeights(source)
        let kept = try XCTUnwrap(store.storedWeightsURL())
        XCTAssertEqual(builder.receivedURLs, [kept])

        // A later launch: a new app model over the same store makes the session from the copy.
        let relaunchBuilder = RecordingSessionBuilder()
        let relaunched = MindAppModel(builder: relaunchBuilder, memory: memory, weightsStore: store)
        relaunched.loadKeptWeights()
        XCTAssertEqual(relaunchBuilder.receivedURLs, [kept])

        // A refused file reaches no builder and names both digests.
        let refusing = ApplicationSupportWeightsStore(
            folder: folder.appendingPathComponent("other", isDirectory: true),
            expectedSHA256: String(repeating: "0", count: 64))
        let refusingBuilder = RecordingSessionBuilder()
        let refusingApp = MindAppModel(builder: refusingBuilder, memory: memory, weightsStore: refusing)
        refusingApp.pickWeights(source)
        XCTAssertTrue(refusingBuilder.receivedURLs.isEmpty)
        XCTAssertTrue(refusingApp.loadMessage?.contains(abcSHA256) ?? false)
    }

    // MARK: Composition

    @MainActor
    func testReleasedStackBuilderNamesTheMissingParts() {
        let memory = MemoryViewModel(monitor: ViewsTestsMemoryMonitor())
        let builder = RealStackSessionBuilder()
        XCTAssertThrowsError(
            try builder.makeRunViewModel(
                weightsURL: URL(fileURLWithPath: "/nonexistent/model.safetensors"),
                planningPermit: memory, observers: [])
        ) { error in
            XCTAssertEqual(error as? SessionBuildingError, .partsNotBuilt(RealStackSessionBuilder.missingParts))
        }
        XCTAssertThrowsError(
            try builder.makeRunViewModel(weightsURL: nil, planningPermit: memory, observers: [])
        ) { error in
            XCTAssertEqual(error as? SessionBuildingError, .noWeightsFile)
        }
    }

    #if DEBUG
    @MainActor
    func testOneModelStepRunsOnTheFakeStack() throws {
        let app = MindAppModel(builder: FakeStackSessionBuilder(), memory: MemoryViewModel(monitor: ViewsTestsMemoryMonitor()))
        app.makeSession(weightsURL: nil)
        let run = try XCTUnwrap(app.run)
        XCTAssertTrue(run.stepOnce())
        XCTAssertEqual(run.completedModelSteps, 1)
        XCTAssertEqual(run.eyeImages.count, EyeCamera.allCases.count)
        XCTAssertEqual(run.command?.scale, .centimetres)
        XCTAssertEqual(app.senses.rows.map(\.sense), SenseName.active)
        XCTAssertTrue(app.senses.rows.allSatisfy { $0.predictionError == nil })
        XCTAssertEqual(app.command.sentRows.count, Command.handleCount)
        XCTAssertTrue(run.stepOnce())
        XCTAssertTrue(app.senses.rows.allSatisfy { $0.predictionError != nil })
    }

    @MainActor
    func testFirstFakePlanningEventIsAtModelStepTwenty() throws {
        let app = MindAppModel(builder: FakeStackSessionBuilder(), memory: MemoryViewModel(monitor: ViewsTestsMemoryMonitor()))
        app.makeSession(weightsURL: nil)
        let run = try XCTUnwrap(app.run)
        for _ in 0 ..< 20 {
            XCTAssertTrue(run.stepOnce())
        }
        XCTAssertNil(app.planner.lastEventStep)
        XCTAssertTrue(run.stepOnce())
        XCTAssertEqual(app.planner.lastEventStep, ModelStep(count: 20))
        XCTAssertEqual(app.planner.scores.count, PlanResult.evaluationsPerEvent)
        XCTAssertEqual(app.planner.selectedIteration, PlanResult.evaluationsPerEvent - 1)
        XCTAssertEqual(app.planner.goalRingCount, 21)
    }
    #endif
}
