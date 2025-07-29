// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "FastVLMKit",
    platforms: [
        .iOS(.v17),
        .macOS(.v14)
    ],
    products: [
        .library(
            name: "FastVLMKit",
            targets: ["FastVLMKit"]
        ),
    ],
    dependencies: [
        // Add dependencies here as needed
    ],
    targets: [
        .target(
            name: "FastVLMKit",
            dependencies: []
        ),
        .testTarget(
            name: "FastVLMKitTests",
            dependencies: ["FastVLMKit"]
        ),
    ]
)