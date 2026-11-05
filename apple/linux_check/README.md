# Linux check of the Apple port

`bash run.sh` compiles and tests every part of `apple/UEMAKit` and `apple/UEMAMind` that needs no Apple
framework and no MLX, on Linux. It downloads Swift 6.2 for Ubuntu 24.04 and the MuJoCo 3.3.0 Linux
release into `$UEMA_LINUX_CACHE` (default `~/.cache/uema_linux_check`) when they are absent. It needs
curl, tar, python3 with numpy and Pillow, and network access to download.swift.org and github.com. It
exits 0 when every test passes and no code line calls the MuJoCo renderer (`mjr_`).

`Package.swift` here is a second manifest over the same files: `Sources`, `Tests` and `UEMAMind` are
links to `../UEMAKit/Sources`, `../UEMAKit/Tests` and `../UEMAMind`, so it checks the files the Mac
build compiles. `Shims/` stands in for four Apple modules: CryptoKit (swift-crypto), Combine
(OpenCombine), and the one SwiftUI and Foundation name each that the app's view models use. The
manifest's header lists what is left out and why. It does not replace the Mac build
(`apple/FIRST_BUILD_CHECKLIST.md`).
