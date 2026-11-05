// Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
// Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
// Linux check only: the one SwiftUI name the app's view models use.
@_exported import Combine

public enum ScenePhase: Equatable, Hashable, Sendable { case active, inactive, background }
