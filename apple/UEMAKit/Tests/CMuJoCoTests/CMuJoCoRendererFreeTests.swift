// CMuJoCoRendererFreeTests.swift: the rule of work item WI-19 that no function of
// MuJoCo's classic OpenGL renderer is named anywhere in the package
// (docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html, row WI-19 forbids any
// function of the OpenGL renderer, whose names start with the prefix this test assembles
// below, anywhere in the package; specification REQ-B-1: the library can be
// used on systems without OpenGL support as long as those functions are not called).
//
// WHAT THIS TEST DOES. INPUT: every file with the extension swift, h, c or modulemap
// under apple/UEMAKit/Sources and apple/UEMAKit/Tests, plus apple/UEMAKit/Package.swift,
// found from this file's own path (#filePath is Tests/CMuJoCoTests/<this file>, so the
// package root is three folders up). OPERATION: read each file as UTF-8 text, cut
// every line at its first "//" (a line comment in Swift, C and module maps), and search
// the code that remains for the renderer prefix, the three letters m, j, r followed by
// an underscore; the prefix is assembled from two string parts so that this file does
// not contain it. Comments may name the renderer to say it is not used (for example
// the header comment of Sources/UEMASenses/Vision/GeomMirror.swift, work item WI-29,
// names it on two lines on November 4, 2026); a call or declaration
// cannot hide in a line comment. A "//" inside a string literal also cuts its line,
// so a prefix after such a literal on the same line is not seen. OUTPUT: the list of
// file:line entries holding the prefix in code, which must be empty, and the count of
// files read, which must be above 1. The same check ran on Windows on November 4, 2026
// as a grep over the same folders that dropped lines starting with "//" and printed
// no line. The test reads the source tree through #filePath, so it runs on the Mac
// that holds the checkout (the macOS destination of the first build).
//
// Nothing here was compiled: no Swift toolchain on the machine that wrote it.
//
// Pages read for the symbols used (November 4, 2026):
// - FileManager.enumerator(at:includingPropertiesForKeys:) and
//   FileManager.DirectoryEnumerator: https://developer.apple.com/documentation/foundation/filemanager/enumerator(at:includingpropertiesforkeys:)
// - String.init(contentsOf:encoding:): https://developer.apple.com/documentation/swift/string/init(contentsof:encoding:)
// - XCTAssertEqual, XCTAssertGreaterThan, XCTAssertFalse:
//   https://developer.apple.com/documentation/xctest ; declarations read in
//   swift-corelibs-xctest, Sources/XCTest/Public/XCTAssert.swift.

import Foundation
import XCTest

/// Reads every source file of the package and fails on the renderer prefix.
final class CMuJoCoRendererFreeTests: XCTestCase {
    /// The prefix of the classic renderer's functions, in two parts.
    static let rendererPrefix = "mjr" + "_"

    /// The file extensions scanned.
    static let scannedExtensions: Set<String> = ["swift", "h", "c", "modulemap"]

    /// apple/UEMAKit, three folders above this file.
    static var packageRoot: URL {
        URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()  // CMuJoCoTests
            .deletingLastPathComponent()  // Tests
            .deletingLastPathComponent()  // UEMAKit
    }

    /// The files under `folder` (recursively) whose extension is scanned, sorted by path.
    func sourceFiles(under folder: URL) throws -> [URL] {
        let enumerator = try XCTUnwrap(
            FileManager.default.enumerator(at: folder, includingPropertiesForKeys: [.isRegularFileKey]),
            "cannot enumerate \(folder.path)")
        var files: [URL] = []
        for case let url as URL in enumerator where Self.scannedExtensions.contains(url.pathExtension) {
            let isRegular = try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile ?? false
            if isRegular {
                files.append(url)
            }
        }
        return files.sorted { $0.path < $1.path }
    }

    /// The part of `line` before its first "//", or the whole line when it has none.
    static func codePart(of line: Substring) -> Substring {
        guard let commentStart = line.range(of: "//") else {
            return line
        }
        return line[line.startIndex..<commentStart.lowerBound]
    }

    /// The 1-based numbers of the lines of `text` whose code part holds the prefix.
    static func offendingLines(in text: String) -> [Int] {
        var numbers: [Int] = []
        for (index, line) in text.split(separator: "\n", omittingEmptySubsequences: false).enumerated()
        where codePart(of: line).contains(rendererPrefix) {
            numbers.append(index + 1)
        }
        return numbers
    }

    /// The comment cut keeps code and drops comments: a prefix after "//" is not
    /// counted, a prefix in code is.
    func testCommentCutFindsOnlyCode() {
        let commentOnly = "let x = 1 // the " + Self.rendererPrefix + "render call is not used"
        let inCode = "    " + Self.rendererPrefix + "render(viewport, scene, context)"
        XCTAssertEqual(Self.offendingLines(in: commentOnly), [])
        XCTAssertEqual(Self.offendingLines(in: "first line\n" + inCode), [2])
    }

    /// No file under Sources, Tests or the manifest holds the renderer prefix in code.
    func testNoRendererSymbolInThePackage() throws {
        let root = Self.packageRoot
        var files = try sourceFiles(under: root.appendingPathComponent("Sources"))
        files += try sourceFiles(under: root.appendingPathComponent("Tests"))
        files.append(root.appendingPathComponent("Package.swift"))
        var offenders: [String] = []
        for file in files {
            let text = try String(contentsOf: file, encoding: .utf8)
            for line in Self.offendingLines(in: text) {
                offenders.append("\(file.path):\(line)")
            }
        }
        XCTAssertGreaterThan(files.count, 1, "no source file found under \(root.path)")
        XCTAssertEqual(offenders, [], "files naming the renderer: \(offenders)")
    }

    /// The module map and the shim of Sources/CMuJoCo are among the scanned files, so
    /// the scan above covers the C module itself.
    func testScanCoversTheCModule() throws {
        let files = try sourceFiles(under: Self.packageRoot.appendingPathComponent("Sources/CMuJoCo"))
        let names = Set(files.map { $0.lastPathComponent })
        XCTAssertTrue(names.contains("module.modulemap"), "scanned: \(names)")
        XCTAssertTrue(names.contains("shim.h"), "scanned: \(names)")
        XCTAssertFalse(names.contains("CMuJoCo.swift"),
                       "a .swift file in Sources/CMuJoCo makes a mixed-language target, refused at swift-tools-version 5.9")
    }
}
