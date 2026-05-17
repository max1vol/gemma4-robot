import Foundation

@MainActor
final class ModelStore: ObservableObject {
  static let defaultModelURL =
    "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"
  static let defaultProjectorURL =
    "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F16.gguf"

  @Published var modelURLString = defaultModelURL
  @Published var projectorURLString = defaultProjectorURL
  @Published private(set) var localModelURL: URL?
  @Published private(set) var localProjectorURL: URL?
  @Published private(set) var isDownloading = false
  @Published private(set) var isDownloadingProjector = false
  @Published private(set) var status = "Model not checked."
  @Published private(set) var projectorStatus = "Projector not checked."
  @Published private(set) var downloadedBytes: Int64 = 0
  @Published private(set) var projectorDownloadedBytes: Int64 = 0
  @Published private(set) var totalBytesExpected: Int64?
  @Published private(set) var projectorTotalBytesExpected: Int64?
  @Published private(set) var downloadSpeedBytesPerSecond: Double = 0
  @Published private(set) var projectorDownloadSpeedBytesPerSecond: Double = 0

  init() {
    refresh()
  }

  var modelFileName: String {
    URL(string: modelURLString)?.lastPathComponent.nonEmpty ?? "gemma-4-E2B-it-Q4_K_M.gguf"
  }

  var projectorFileName: String {
    URL(string: projectorURLString)?.lastPathComponent.nonEmpty ?? "mmproj-F16.gguf"
  }

  var hasModel: Bool {
    modelURLForLoading != nil
  }

  var hasProjector: Bool {
    projectorURLForLoading != nil
  }

  var modelURLForLoading: URL? {
    let candidateURL = localModelURL ?? modelsDirectory().appendingPathComponent(modelFileName)
    guard FileManager.default.fileExists(atPath: candidateURL.path) else {
      return nil
    }
    return candidateURL
  }

  var projectorURLForLoading: URL? {
    let candidateURL = localProjectorURL ?? modelsDirectory().appendingPathComponent(projectorFileName)
    guard FileManager.default.fileExists(atPath: candidateURL.path) else {
      return nil
    }
    return candidateURL
  }

  var modelSizeForLoading: Int64? {
    guard let modelURLForLoading else { return nil }
    return fileSize(modelURLForLoading)
  }

  var downloadProgressFraction: Double? {
    guard
      let totalBytesExpected,
      totalBytesExpected > 0,
      isDownloading
    else {
      return nil
    }

    return min(1, max(0, Double(downloadedBytes) / Double(totalBytesExpected)))
  }

  var projectorDownloadProgressFraction: Double? {
    guard
      let projectorTotalBytesExpected,
      projectorTotalBytesExpected > 0,
      isDownloadingProjector
    else {
      return nil
    }

    return min(1, max(0, Double(projectorDownloadedBytes) / Double(projectorTotalBytesExpected)))
  }

  var downloadDetail: String {
    let currentSize = Self.formatBytes(downloadedBytes)
    let speed = Self.formatSpeed(downloadSpeedBytesPerSecond)

    if let totalBytesExpected, totalBytesExpected > 0 {
      let totalSize = Self.formatBytes(totalBytesExpected)
      let percent = 100 * Double(downloadedBytes) / Double(totalBytesExpected)
      return String(format: "%.1f%% - %@ of %@ - %@", percent, currentSize, totalSize, speed)
    }

    return "\(currentSize) downloaded - \(speed)"
  }

  var projectorDownloadDetail: String {
    let currentSize = Self.formatBytes(projectorDownloadedBytes)
    let speed = Self.formatSpeed(projectorDownloadSpeedBytesPerSecond)

    if let projectorTotalBytesExpected, projectorTotalBytesExpected > 0 {
      let totalSize = Self.formatBytes(projectorTotalBytesExpected)
      let percent = 100 * Double(projectorDownloadedBytes) / Double(projectorTotalBytesExpected)
      return String(format: "%.1f%% - %@ of %@ - %@", percent, currentSize, totalSize, speed)
    }

    return "\(currentSize) downloaded - \(speed)"
  }

  func refresh() {
    let localURL = modelsDirectory().appendingPathComponent(modelFileName)
    localModelURL = localURL
    if let size = fileSize(localURL) {
      downloadedBytes = size
      totalBytesExpected = size
      downloadSpeedBytesPerSecond = 0
      status = "Model ready: \(Self.formatBytes(size))"
    } else {
      downloadedBytes = 0
      totalBytesExpected = nil
      downloadSpeedBytesPerSecond = 0
      status = "Model will download to app storage."
    }

    let localProjector = modelsDirectory().appendingPathComponent(projectorFileName)
    localProjectorURL = localProjector
    if let size = fileSize(localProjector) {
      projectorDownloadedBytes = size
      projectorTotalBytesExpected = size
      projectorDownloadSpeedBytesPerSecond = 0
      projectorStatus = "Projector ready: \(Self.formatBytes(size))"
    } else {
      projectorDownloadedBytes = 0
      projectorTotalBytesExpected = nil
      projectorDownloadSpeedBytesPerSecond = 0
      projectorStatus = "Projector will download to app storage."
    }
  }

  func downloadModel() async {
    guard !isDownloading else { return }
    guard let remoteURL = URL(string: modelURLString) else {
      status = "Invalid model URL."
      print("Gemma model download failed: invalid URL \(modelURLString)")
      return
    }

    let targetURL = modelsDirectory().appendingPathComponent(remoteURL.lastPathComponent)
    localModelURL = targetURL
    isDownloading = true
    downloadedBytes = 0
    totalBytesExpected = nil
    downloadSpeedBytesPerSecond = 0
    status = "Downloading model..."
    let startedAt = Date()

    do {
      try FileManager.default.createDirectory(
        at: modelsDirectory(),
        withIntermediateDirectories: true
      )
      let (tempURL, response) = try await download(
        from: remoteURL,
        suggestedFileName: targetURL.lastPathComponent
      ) { [weak self] bytesWritten, totalBytes in
        guard let self else { return }
        Task { @MainActor in
          self.downloadedBytes = bytesWritten
          self.totalBytesExpected = totalBytes
          let elapsed = max(0.1, Date().timeIntervalSince(startedAt))
          self.downloadSpeedBytesPerSecond = Double(bytesWritten) / elapsed
          self.status = "Downloading model..."
        }
      }
      if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
        throw DownloadError.badStatus(http.statusCode)
      }
      if FileManager.default.fileExists(atPath: targetURL.path) {
        try FileManager.default.removeItem(at: targetURL)
      }
      try FileManager.default.moveItem(at: tempURL, to: targetURL)
      var values = URLResourceValues()
      values.isExcludedFromBackup = true
      var mutableTarget = targetURL
      try? mutableTarget.setResourceValues(values)
      downloadedBytes = fileSize(targetURL) ?? response.expectedContentLength
      totalBytesExpected = downloadedBytes
      downloadSpeedBytesPerSecond = 0
      status = "Model ready: \(Self.formatBytes(downloadedBytes))"
    } catch {
      status = "Download failed: \(error.localizedDescription)"
      downloadSpeedBytesPerSecond = 0
      print("Gemma model download failed: \(error.localizedDescription)")
    }

    isDownloading = false
  }

  func downloadProjector() async {
    guard !isDownloadingProjector else { return }
    guard let remoteURL = URL(string: projectorURLString) else {
      projectorStatus = "Invalid projector URL."
      print("Gemma projector download failed: invalid URL \(projectorURLString)")
      return
    }

    let targetURL = modelsDirectory().appendingPathComponent(remoteURL.lastPathComponent)
    localProjectorURL = targetURL
    isDownloadingProjector = true
    projectorDownloadedBytes = 0
    projectorTotalBytesExpected = nil
    projectorDownloadSpeedBytesPerSecond = 0
    projectorStatus = "Downloading projector..."
    let startedAt = Date()

    do {
      try FileManager.default.createDirectory(
        at: modelsDirectory(),
        withIntermediateDirectories: true
      )
      let (tempURL, response) = try await download(
        from: remoteURL,
        suggestedFileName: targetURL.lastPathComponent
      ) { [weak self] bytesWritten, totalBytes in
        guard let self else { return }
        Task { @MainActor in
          self.projectorDownloadedBytes = bytesWritten
          self.projectorTotalBytesExpected = totalBytes
          let elapsed = max(0.1, Date().timeIntervalSince(startedAt))
          self.projectorDownloadSpeedBytesPerSecond = Double(bytesWritten) / elapsed
          self.projectorStatus = "Downloading projector..."
        }
      }
      if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
        throw DownloadError.badStatus(http.statusCode)
      }
      if FileManager.default.fileExists(atPath: targetURL.path) {
        try FileManager.default.removeItem(at: targetURL)
      }
      try FileManager.default.moveItem(at: tempURL, to: targetURL)
      var values = URLResourceValues()
      values.isExcludedFromBackup = true
      var mutableTarget = targetURL
      try? mutableTarget.setResourceValues(values)
      projectorDownloadedBytes = fileSize(targetURL) ?? response.expectedContentLength
      projectorTotalBytesExpected = projectorDownloadedBytes
      projectorDownloadSpeedBytesPerSecond = 0
      projectorStatus = "Projector ready: \(Self.formatBytes(projectorDownloadedBytes))"
    } catch {
      projectorStatus = "Projector download failed: \(error.localizedDescription)"
      projectorDownloadSpeedBytesPerSecond = 0
      print("Gemma projector download failed: \(error.localizedDescription)")
    }

    isDownloadingProjector = false
  }

  func deleteModel() {
    guard let modelURL = localModelURL ?? modelURLForLoading else { return }
    try? FileManager.default.removeItem(at: modelURL)
    refresh()
  }

  func deleteProjector() {
    guard let projectorURL = localProjectorURL ?? projectorURLForLoading else { return }
    try? FileManager.default.removeItem(at: projectorURL)
    refresh()
  }

  private func modelsDirectory() -> URL {
    let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
    let directory = base.appendingPathComponent("Models", isDirectory: true)
    var isDirectory: ObjCBool = false
    if FileManager.default.fileExists(atPath: directory.path, isDirectory: &isDirectory),
       !isDirectory.boolValue
    {
      AppLog.error("Models path was a file, removing it so the model directory can be recreated: \(directory.path)")
      try? FileManager.default.removeItem(at: directory)
    }
    try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    return directory
  }

  private func fileSize(_ url: URL) -> Int64? {
    guard
      let values = try? url.resourceValues(forKeys: [.fileSizeKey]),
      let size = values.fileSize
    else {
      return nil
    }
    return Int64(size)
  }

  private nonisolated func download(
    from remoteURL: URL,
    suggestedFileName: String,
    progress: @escaping @Sendable (Int64, Int64?) -> Void
  ) async throws -> (URL, URLResponse) {
    let delegate = ModelDownloadDelegate(
      suggestedFileName: suggestedFileName,
      progress: progress
    )
    let session = URLSession(
      configuration: .default,
      delegate: delegate,
      delegateQueue: nil
    )
    defer {
      session.finishTasksAndInvalidate()
    }

    return try await withTaskCancellationHandler {
      try await withCheckedThrowingContinuation { continuation in
        delegate.continuation = continuation
        session.downloadTask(with: remoteURL).resume()
      }
    } onCancel: {
      session.invalidateAndCancel()
    }
  }

  static func formatBytes(_ bytes: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
  }

  static func formatSpeed(_ bytesPerSecond: Double) -> String {
    let bytes = Int64(max(0, bytesPerSecond))
    return "\(ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file))/s"
  }
}

private final class ModelDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
  var continuation: CheckedContinuation<(URL, URLResponse), Error>?

  private let suggestedFileName: String
  private let progress: @Sendable (Int64, Int64?) -> Void
  private let lock = NSLock()

  init(
    suggestedFileName: String,
    progress: @escaping @Sendable (Int64, Int64?) -> Void
  ) {
    self.suggestedFileName = suggestedFileName
    self.progress = progress
  }

  func urlSession(
    _ session: URLSession,
    downloadTask: URLSessionDownloadTask,
    didWriteData bytesWritten: Int64,
    totalBytesWritten: Int64,
    totalBytesExpectedToWrite: Int64
  ) {
    let expected = totalBytesExpectedToWrite > 0 ? totalBytesExpectedToWrite : nil
    progress(totalBytesWritten, expected)
  }

  func urlSession(
    _ session: URLSession,
    downloadTask: URLSessionDownloadTask,
    didFinishDownloadingTo location: URL
  ) {
    do {
      let tempURL = FileManager.default.temporaryDirectory
        .appendingPathComponent(UUID().uuidString)
        .appendingPathExtension(suggestedFileName.pathExtensionOrDefault("litertlm"))
      if FileManager.default.fileExists(atPath: tempURL.path) {
        try FileManager.default.removeItem(at: tempURL)
      }
      try FileManager.default.moveItem(at: location, to: tempURL)
      let fallbackResponse = URLResponse(
        url: downloadTask.originalRequest?.url ?? tempURL,
        mimeType: nil,
        expectedContentLength: -1,
        textEncodingName: nil
      )
      resume(.success((tempURL, downloadTask.response ?? fallbackResponse)))
    } catch {
      resume(.failure(error))
    }
  }

  func urlSession(
    _ session: URLSession,
    task: URLSessionTask,
    didCompleteWithError error: Error?
  ) {
    if let error {
      resume(.failure(error))
    }
  }

  private func resume(_ result: Result<(URL, URLResponse), Error>) {
    lock.lock()
    let continuation = continuation
    self.continuation = nil
    lock.unlock()

    switch result {
    case .success(let value):
      continuation?.resume(returning: value)
    case .failure(let error):
      continuation?.resume(throwing: error)
    }
  }
}

private enum DownloadError: LocalizedError {
  case badStatus(Int)

  var errorDescription: String? {
    switch self {
    case .badStatus(let status):
      return "HTTP \(status)"
    }
  }
}

private extension String {
  var nonEmpty: String? {
    isEmpty ? nil : self
  }

  func pathExtensionOrDefault(_ fallback: String) -> String {
    let ext = (self as NSString).pathExtension
    return ext.isEmpty ? fallback : ext
  }
}
