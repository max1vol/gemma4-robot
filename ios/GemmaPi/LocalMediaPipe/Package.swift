// swift-tools-version: 5.9
import PackageDescription

let package = Package(
  name: "LocalMediaPipeTasks",
  platforms: [
    .iOS(.v17)
  ],
  products: [
    .library(
      name: "MediaPipeTasksCommon",
      targets: ["MediaPipeTasksCommon"]
    ),
    .library(
      name: "MediaPipeTasksVision",
      targets: ["MediaPipeTasksCommon", "MediaPipeTasksVision"]
    )
  ],
  targets: [
    .binaryTarget(
      name: "MediaPipeTasksCommon",
      path: "Artifacts/MediaPipeTasksCommon.xcframework"
    ),
    .binaryTarget(
      name: "MediaPipeTasksVision",
      path: "Artifacts/MediaPipeTasksVision.xcframework"
    )
  ]
)
