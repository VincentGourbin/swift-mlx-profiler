// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "swift-mlx-profiler",
    platforms: [
        .macOS(.v15)
    ],
    products: [
        .library(name: "MLXProfiler", targets: ["MLXProfiler"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
    ],
    targets: [
        .target(
            name: "MLXProfiler",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
            ]
        ),
        .testTarget(
            name: "MLXProfilerTests",
            dependencies: ["MLXProfiler"]
        ),
    ]
)
