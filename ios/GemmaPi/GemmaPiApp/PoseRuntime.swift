import CoreGraphics
import Compression
import Foundation
import MediaPipeTasksVision
import UIKit

enum PoseBackend: String, Sendable {
  case cpu
  case gpu

  var displayName: String {
    switch self {
    case .cpu: "MediaPipe CPU"
    case .gpu: "MediaPipe GPU"
    }
  }

  var delegate: Delegate {
    switch self {
    case .cpu: .CPU
    case .gpu: .GPU
    }
  }
}

enum PoseModelQuality: String, Sendable, CaseIterable {
  case lite
  case full
  case heavy

  var displayName: String {
    switch self {
    case .lite: "Lite"
    case .full: "Full"
    case .heavy: "Heavy"
    }
  }

  var fileName: String {
    "pose_landmarker_\(rawValue).task"
  }

  var cdnURL: URL {
    URL(string: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_\(rawValue)/float16/latest/pose_landmarker_\(rawValue).task")!
  }
}

struct PoseBridgeInput: Sendable {
  let format: String
  let width: Int
  let height: Int
  let data: Data
  let backend: PoseBackend
  let modelQuality: PoseModelQuality
}

struct PoseBridgeOutput: @unchecked Sendable {
  let payload: [String: Any]
}

final class PoseRuntime: @unchecked Sendable {
  private let queue = DispatchQueue(label: "com.gemma4robot.pose-runtime")
  private var landmarkers: [LandmarkerKey: PoseLandmarker] = [:]

  func detect(_ input: PoseBridgeInput) async throws -> PoseBridgeOutput {
    try await withCheckedThrowingContinuation { continuation in
      queue.async {
        do {
          let output = try self.detectSync(input)
          continuation.resume(returning: output)
        } catch {
          continuation.resume(throwing: error)
        }
      }
    }
  }

  private func detectSync(_ input: PoseBridgeInput) throws -> PoseBridgeOutput {
    let decodeStart = Date()
    let image = try makeMPImage(input)
    let decodeSeconds = -decodeStart.timeIntervalSinceNow

    let loadStart = Date()
    let landmarker = try landmarker(for: input.backend, modelQuality: input.modelQuality)
    let loadSeconds = -loadStart.timeIntervalSinceNow

    let inferenceStart = Date()
    let result = try landmarker.detect(image: image)
    let inferenceSeconds = -inferenceStart.timeIntervalSinceNow

    return PoseBridgeOutput(payload: Self.payload(
      from: result,
      backend: input.backend,
      modelQuality: input.modelQuality,
      inputFormat: input.format,
      width: input.width,
      height: input.height,
      inputBytes: input.data.count,
      decodeSeconds: decodeSeconds,
      loadSeconds: loadSeconds,
      inferenceSeconds: inferenceSeconds
    ))
  }

  private func landmarker(for backend: PoseBackend, modelQuality: PoseModelQuality) throws -> PoseLandmarker {
    let key = LandmarkerKey(backend: backend, modelQuality: modelQuality)
    if let landmarker = landmarkers[key] {
      return landmarker
    }

    let modelURL = try Self.ensureModel(modelQuality)
    let options = PoseLandmarkerOptions()
    options.baseOptions.modelAssetPath = modelURL.path
    options.baseOptions.delegate = backend.delegate
    options.runningMode = .image
    options.numPoses = 1
    options.minPoseDetectionConfidence = 0.5
    options.minPosePresenceConfidence = 0.5
    options.minTrackingConfidence = 0.5
    options.shouldOutputSegmentationMasks = false

    AppLog.info("Pose runtime load starting: backend=\(backend.rawValue), pose_model=\(modelQuality.rawValue), model=\(modelURL.path)")
    let landmarker = try PoseLandmarker(options: options)
    landmarkers[key] = landmarker
    AppLog.info("Pose runtime load finished: backend=\(backend.rawValue), pose_model=\(modelQuality.rawValue)")
    return landmarker
  }

  private func makeMPImage(_ input: PoseBridgeInput) throws -> MPImage {
    let rgba: Data
    switch input.format {
    case "rgba32":
      guard input.data.count == input.width * input.height * 4 else {
        throw PoseRuntimeError.invalidFrame("rgba32 byte count does not match width*height*4")
      }
      rgba = input.data
    case "rgb24":
      rgba = try Self.rgbaFromRGB24(input.data, width: input.width, height: input.height)
    case "yuv420":
      rgba = try Self.rgbaFromYUV420(input.data, width: input.width, height: input.height)
    case "deflate_rgb24", "zlib_rgb24":
      let rgb = try Self.decompressZlib(input.data, expectedSize: input.width * input.height * 3)
      rgba = try Self.rgbaFromRGB24(rgb, width: input.width, height: input.height)
    case "deflate_yuv420", "zlib_yuv420":
      let yuv = try Self.decompressZlib(input.data, expectedSize: input.width * input.height * 3 / 2)
      rgba = try Self.rgbaFromYUV420(yuv, width: input.width, height: input.height)
    case "jpeg", "jpg":
      guard let image = UIImage(data: input.data) else {
        throw PoseRuntimeError.invalidFrame("could not decode jpeg")
      }
      return try MPImage(uiImage: image, orientation: .up)
    default:
      throw PoseRuntimeError.invalidFrame("unsupported pose frame format: \(input.format)")
    }

    guard let cgImage = Self.cgImageFromRGBA(rgba, width: input.width, height: input.height) else {
      throw PoseRuntimeError.invalidFrame("could not create CGImage")
    }
    return try MPImage(uiImage: UIImage(cgImage: cgImage), orientation: .up)
  }

  private static func ensureModel(_ modelQuality: PoseModelQuality) throws -> URL {
    let manager = FileManager.default
    let directory = try manager.url(
      for: .applicationSupportDirectory,
      in: .userDomainMask,
      appropriateFor: nil,
      create: true
    ).appendingPathComponent("Models", isDirectory: true)
    try manager.createDirectory(at: directory, withIntermediateDirectories: true)
    let target = directory.appendingPathComponent(modelQuality.fileName)
    if manager.fileExists(atPath: target.path) {
      return target
    }

    let source = modelQuality.cdnURL
    AppLog.info("Pose model download starting: \(source.absoluteString)")
    let data = try Data(contentsOf: source)
    try data.write(to: target, options: .atomic)
    AppLog.info("Pose model download finished: path=\(target.path), size=\(data.count)")
    return target
  }

  private static func rgbaFromRGB24(_ rgb: Data, width: Int, height: Int) throws -> Data {
    guard rgb.count == width * height * 3 else {
      throw PoseRuntimeError.invalidFrame("rgb24 byte count does not match width*height*3")
    }
    var rgba = Data(count: width * height * 4)
    rgb.withUnsafeBytes { rgbBuffer in
      rgba.withUnsafeMutableBytes { rgbaBuffer in
        let src = rgbBuffer.bindMemory(to: UInt8.self)
        let dst = rgbaBuffer.bindMemory(to: UInt8.self)
        var si = 0
        var di = 0
        while si < src.count {
          dst[di] = src[si]
          dst[di + 1] = src[si + 1]
          dst[di + 2] = src[si + 2]
          dst[di + 3] = 255
          si += 3
          di += 4
        }
      }
    }
    return rgba
  }

  private static func decompressZlib(_ compressed: Data, expectedSize: Int) throws -> Data {
    guard expectedSize > 0 else {
      throw PoseRuntimeError.invalidFrame("zlib frame has invalid expected size")
    }
    var output = Data(count: expectedSize)
    let decodedSize: Int = compressed.withUnsafeBytes { sourceRaw in
      output.withUnsafeMutableBytes { targetRaw in
        guard
          let source = sourceRaw.bindMemory(to: UInt8.self).baseAddress,
          let target = targetRaw.bindMemory(to: UInt8.self).baseAddress
        else {
          return 0
        }
        return compression_decode_buffer(
          target,
          expectedSize,
          source,
          compressed.count,
          nil,
          COMPRESSION_ZLIB
        )
      }
    }
    guard decodedSize == expectedSize else {
      throw PoseRuntimeError.invalidFrame("deflate decode produced \(decodedSize) bytes, expected \(expectedSize)")
    }
    return output
  }

  private static func rgbaFromYUV420(_ yuv: Data, width: Int, height: Int) throws -> Data {
    let frame = width * height
    guard yuv.count == frame * 3 / 2 else {
      throw PoseRuntimeError.invalidFrame("yuv420 byte count does not match width*height*3/2")
    }

    var rgba = Data(count: width * height * 4)
    yuv.withUnsafeBytes { yuvBuffer in
      rgba.withUnsafeMutableBytes { rgbaBuffer in
        let src = yuvBuffer.bindMemory(to: UInt8.self)
        let dst = rgbaBuffer.bindMemory(to: UInt8.self)
        let uOffset = frame
        let vOffset = frame + frame / 4
        let chromaWidth = width / 2
        for row in 0..<height {
          for col in 0..<width {
            let y = Int(src[row * width + col])
            let uv = (row / 2) * chromaWidth + (col / 2)
            let u = Int(src[uOffset + uv])
            let v = Int(src[vOffset + uv])
            let c = y - 16
            let d = u - 128
            let e = v - 128
            let r = clamp((298 * c + 409 * e + 128) >> 8)
            let g = clamp((298 * c - 100 * d - 208 * e + 128) >> 8)
            let b = clamp((298 * c + 516 * d + 128) >> 8)
            let out = (row * width + col) * 4
            dst[out] = UInt8(r)
            dst[out + 1] = UInt8(g)
            dst[out + 2] = UInt8(b)
            dst[out + 3] = 255
          }
        }
      }
    }
    return rgba
  }

  private static func cgImageFromRGBA(_ rgba: Data, width: Int, height: Int) -> CGImage? {
    let provider = CGDataProvider(data: rgba as CFData)
    let colorSpace = CGColorSpaceCreateDeviceRGB()
    return CGImage(
      width: width,
      height: height,
      bitsPerComponent: 8,
      bitsPerPixel: 32,
      bytesPerRow: width * 4,
      space: colorSpace,
      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
      provider: provider!,
      decode: nil,
      shouldInterpolate: false,
      intent: .defaultIntent
    )
  }

  private static func payload(
    from result: PoseLandmarkerResult,
    backend: PoseBackend,
    modelQuality: PoseModelQuality,
    inputFormat: String,
    width: Int,
    height: Int,
    inputBytes: Int,
    decodeSeconds: TimeInterval,
    loadSeconds: TimeInterval,
    inferenceSeconds: TimeInterval
  ) -> [String: Any] {
    let landmarks = result.landmarks.first ?? []
    let worldLandmarks = result.worldLandmarks.first ?? []
    let posePresence = landmarks.map { landmark in
      landmark.presence?.doubleValue ?? landmark.visibility?.doubleValue ?? 0
    }.max() ?? 0

    return [
      "pose_count": result.landmarks.isEmpty ? 0 : 1,
      "pose_presence": posePresence,
      "pose_landmarks": [landmarks.map(normalizedPayload)],
      "pose_world_landmarks": [worldLandmarks.map(worldPayload)],
      "backend": backend.rawValue,
      "backend_name": backend.displayName,
      "pose_model": modelQuality.rawValue,
      "pose_model_name": modelQuality.displayName,
      "input_format": inputFormat,
      "frame_width": width,
      "frame_height": height,
      "input_bytes": inputBytes,
      "decode_seconds": decodeSeconds,
      "load_seconds": loadSeconds,
      "inference_seconds": inferenceSeconds,
      "total_seconds": decodeSeconds + loadSeconds + inferenceSeconds
    ]
  }

  private static func normalizedPayload(_ landmark: NormalizedLandmark) -> [String: Any] {
    [
      "x": Double(landmark.x),
      "y": Double(landmark.y),
      "z": Double(landmark.z),
      "visibility": landmark.visibility?.doubleValue ?? 0,
      "presence": landmark.presence?.doubleValue ?? 0
    ]
  }

  private static func worldPayload(_ landmark: Landmark) -> [String: Any] {
    [
      "x": Double(landmark.x),
      "y": Double(landmark.y),
      "z": Double(landmark.z),
      "visibility": landmark.visibility?.doubleValue ?? 0,
      "presence": landmark.presence?.doubleValue ?? 0
    ]
  }

  private static func clamp(_ value: Int) -> Int {
    if value < 0 { return 0 }
    if value > 255 { return 255 }
    return value
  }
}

private struct LandmarkerKey: Hashable {
  let backend: PoseBackend
  let modelQuality: PoseModelQuality
}

enum PoseRuntimeError: LocalizedError {
  case invalidFrame(String)

  var errorDescription: String? {
    switch self {
    case .invalidFrame(let message):
      message
    }
  }
}
