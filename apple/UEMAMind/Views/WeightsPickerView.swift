// WeightsPickerView.swift: the file importer for model.safetensors, the released weights
// (73,119,988 bytes; weights/b1_continue_B1K_skip13_492800/MANIFEST.json:121 through the
// plan's section 1; not in git, fetched by
// `python fetch_weights.py --accept-license`, plan section 8). Work item WI-38, target B
// of docs/UEMA_APPLE_PORT_IMPLEMENTATION_PLAN_2026-11-04.html (row WI-38:
// "WeightsPickerView: a file importer for model.safetensors (not in git) that copies the
// picked file into the app's Application Support folder, checks its SHA-256 against
// weight_manifest.json, and loads from there on later launches").
//
// Nothing in this file was compiled on the machine that wrote it (Windows 11, no Swift
// toolchain; plan section 3, fact 1). SwiftUI's `fileImporter` modifier (macOS 11 and
// iOS 14 or newer), UniformTypeIdentifiers, and CryptoKit's `SHA256` (macOS 10.15 and
// iOS 13 or newer) for the file check; no UIKit or AppKit type.
//
// INPUT: the file the person picks in the system file dialog. OPERATION: the URL from the
// dialog is security-scoped on iOS (and on a sandboxed macOS app); the view calls
// `startAccessingSecurityScopedResource()`, reads the file size, calls `onPick` with the
// URL while access is held, and calls `stopAccessingSecurityScopedResource()` after
// `onPick` returns. `onPick` must therefore read the file (UEMAModel's SafetensorsLoader,
// WI-31, reads it synchronously in `load(from:)`, Protocols.swift:623-634) before it
// returns. OUTPUT: the `onPick` call; the view shows the picked file's name, its byte
// count against 73,119,988, and the last load message it is given.
//
// THE KEPT COPY (`ApplicationSupportWeightsStore`, injected into MindAppModel through the
// `WeightsFileStoring` protocol, so the view holds no file logic): `onPick` copies the
// picked file to <Application Support>/UEMAMind/model.safetensors.partial, computes the
// SHA-256 of the copy in 1,048,576-byte chunks, compares the 64 hex digits with
// weight_manifest.json's source.sha256 (apple/UEMAKit/Sources/UEMAModel/Resources/
// weight_manifest.json:7, "8f99a3ce...a315a"), and on a match moves the copy to
// model.safetensors, replacing an earlier one; on a mismatch it deletes the copy and
// throws `WeightsFileError.sha256Mismatch` naming both digests. On a later launch
// MindAppModel.loadKeptWeights() makes the session from the kept copy, so the picker is
// needed once per device. The copy and the hash run on the main actor inside `onPick`,
// while the file dialog's security scope is held.
//
// The "First build on a Mac" checklist of this work item is in UEMAMindApp.swift's header.

import CryptoKit
import Foundation
import SwiftUI
import UniformTypeIdentifiers

/// The weights file check. One responsibility: the expected size and the file type.
enum WeightsFileCheck {
    /// The released model.safetensors in bytes: 73,119,988
    /// (weights/b1_continue_B1K_skip13_492800/MANIFEST.json:121, plan section 1;
    /// Protocols.swift:623; weight_manifest.json source.bytes).
    static let expectedBytes = 73_119_988

    /// The released model.safetensors' SHA-256, 64 lowercase hex digits
    /// (apple/UEMAKit/Sources/UEMAModel/Resources/weight_manifest.json:7, source.sha256).
    static let expectedSHA256 = "8f99a3ce379303eb01c8364c9f40007a13e811c1b77f59b1b89c4058f44a315a"

    /// Bytes read per chunk while hashing: 1,048,576 (1 MB), so the 73,119,988-byte file
    /// is never held in memory whole.
    static let hashChunkBytes = 1_048_576

    /// The SHA-256 of the file at `url` as 64 lowercase hex digits, read in chunks of
    /// `chunkBytes` (at least 1); throws when the file cannot be opened or read.
    static func sha256Hex(of url: URL, chunkBytes: Int = hashChunkBytes) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let chunk = try handle.read(upToCount: max(chunkBytes, 1)), !chunk.isEmpty {
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// The file types the dialog offers: the "safetensors" extension when the system can
    /// make a type of it, else any data file.
    static var allowedTypes: [UTType] {
        if let type = UTType(filenameExtension: "safetensors", conformingTo: .data) {
            return [type]
        }
        return [.data]
    }

    /// "73,119,988 bytes, the released size" or "1,024 bytes; the released file has
    /// 73,119,988"; "size unread" for nil.
    static func sizeText(bytes: Int?) -> String {
        guard let bytes else { return "size unread" }
        if bytes == expectedBytes {
            return "\(groupedDigits(bytes)) bytes, the released size"
        }
        return "\(groupedDigits(bytes)) bytes; the released file has \(groupedDigits(expectedBytes))"
    }

    /// `value` with a comma between each group of three digits, such as "73,119,988",
    /// the same in every locale (a NumberFormatter would follow the device's locale).
    static func groupedDigits(_ value: Int) -> String {
        let digits = String(value.magnitude)
        var out = ""
        for (index, character) in digits.enumerated() {
            if index > 0, (digits.count - index) % 3 == 0 {
                out.append(",")
            }
            out.append(character)
        }
        return value < 0 ? "-" + out : out
    }
}

/// Why a picked weights file was not kept.
enum WeightsFileError: Error, Equatable, CustomStringConvertible {
    /// The copy's SHA-256 is not the released file's.
    case sha256Mismatch(expected: String, found: String)

    /// The Application Support folder could not be found or made.
    case noStoreFolder(String)

    var description: String {
        switch self {
        case .sha256Mismatch(let expected, let found):
            return "The picked file's SHA-256 is \(found); the released model.safetensors has \(expected). The file was not kept."
        case .noStoreFolder(let reason):
            return "The Application Support folder is not available: \(reason)"
        }
    }
}

/// Keeps one checked copy of the weights file for later launches.
protocol WeightsFileStoring {
    /// The kept copy's URL when one exists; nil otherwise.
    func storedWeightsURL() -> URL?

    /// Copies `source` into the store, checks the copy's SHA-256, and returns the kept
    /// copy's URL; a mismatch deletes the copy and throws `WeightsFileError.sha256Mismatch`.
    func importWeights(from source: URL) throws -> URL
}

/// WeightsFileStoring over a folder, by default <Application Support>/UEMAMind.
struct ApplicationSupportWeightsStore: WeightsFileStoring {
    /// The kept copy's file name: model.safetensors.
    static let fileName = "model.safetensors"

    /// The folder name under Application Support: UEMAMind.
    static let folderName = "UEMAMind"

    private let folder: URL?
    private let expectedSHA256: String
    private let fileManager: FileManager

    /// A store in `folder` (nil: <Application Support>/UEMAMind, found when first used),
    /// accepting only a file whose SHA-256 is `expectedSHA256` (hex, either case).
    init(folder: URL? = nil, expectedSHA256: String = WeightsFileCheck.expectedSHA256, fileManager: FileManager = .default) {
        self.folder = folder
        self.expectedSHA256 = expectedSHA256.lowercased()
        self.fileManager = fileManager
    }

    func storedWeightsURL() -> URL? {
        guard let directory = try? storeFolder() else { return nil }
        let url = directory.appendingPathComponent(ApplicationSupportWeightsStore.fileName)
        return fileManager.fileExists(atPath: url.path) ? url : nil
    }

    func importWeights(from source: URL) throws -> URL {
        let directory = try storeFolder()
        try fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        let destination = directory.appendingPathComponent(ApplicationSupportWeightsStore.fileName)
        let partial = directory.appendingPathComponent(ApplicationSupportWeightsStore.fileName + ".partial")
        if fileManager.fileExists(atPath: partial.path) {
            try fileManager.removeItem(at: partial)
        }
        try fileManager.copyItem(at: source, to: partial)
        let found: String
        do {
            found = try WeightsFileCheck.sha256Hex(of: partial)
        } catch {
            try? fileManager.removeItem(at: partial)
            throw error
        }
        guard found == expectedSHA256 else {
            try? fileManager.removeItem(at: partial)
            throw WeightsFileError.sha256Mismatch(expected: expectedSHA256, found: found)
        }
        if fileManager.fileExists(atPath: destination.path) {
            try fileManager.removeItem(at: destination)
        }
        try fileManager.moveItem(at: partial, to: destination)
        return destination
    }

    /// The folder given at init, or <Application Support>/UEMAMind.
    private func storeFolder() throws -> URL {
        if let folder {
            return folder
        }
        do {
            let support = try fileManager.url(
                for: .applicationSupportDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            return support.appendingPathComponent(ApplicationSupportWeightsStore.folderName, isDirectory: true)
        } catch {
            throw WeightsFileError.noStoreFolder(String(describing: error))
        }
    }
}

/// A button that opens the system file dialog for model.safetensors, and the picked
/// file's name, size and load message.
struct WeightsPickerView: View {
    /// The last load's message from the session, such as "802 tensors loaded" or an
    /// error naming a missing key; nil before a pick.
    let loadMessage: String?

    /// Called with the picked file's URL while its security scope is held; must read the
    /// file before returning.
    let onPick: (URL) -> Void

    @State private var isImporterPresented = false
    @State private var pickedName: String?
    @State private var pickedBytes: Int?
    @State private var pickError: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Button("Pick model.safetensors") {
                isImporterPresented = true
            }
            if let pickedName {
                Text("\(pickedName): \(WeightsFileCheck.sizeText(bytes: pickedBytes))")
                    .font(.caption)
            }
            if let loadMessage {
                Text(loadMessage)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            if let pickError {
                Text("File dialog error: \(pickError)")
                    .font(.caption)
                    .foregroundStyle(.red)
            }
        }
        .fileImporter(
            isPresented: $isImporterPresented,
            allowedContentTypes: WeightsFileCheck.allowedTypes
        ) { result in
            switch result {
            case .success(let url):
                pickError = nil
                let scoped = url.startAccessingSecurityScopedResource()
                defer {
                    if scoped {
                        url.stopAccessingSecurityScopedResource()
                    }
                }
                pickedName = url.lastPathComponent
                pickedBytes = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize
                onPick(url)
            case .failure(let error):
                pickError = error.localizedDescription
            }
        }
    }
}

#Preview("Weights picker") {
    WeightsPickerView(loadMessage: nil, onPick: { _ in })
        .padding()
}
