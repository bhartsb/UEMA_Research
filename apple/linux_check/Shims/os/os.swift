// Copyright (c) 2026 Mind Machine Learning, Inc. All rights reserved.
// Licensed under the Mind Machine Learning Noncommercial License 1.0; see the LICENSE file.
// Linux check only: ProcessInfo.ThermalState exists only in Apple's Foundation.
import Foundation

extension ProcessInfo {
    public enum ThermalState: Int, Sendable { case nominal, fair, serious, critical }
    public var thermalState: ThermalState { .nominal }
}
