import AVFoundation
import Foundation
import SWCompression

enum PhoneTTSBackend: String, CaseIterable, Sendable {
  case piperRyanHigh = "piper-ryan-high"

  static let selectableCases: [PhoneTTSBackend] = [.piperRyanHigh]

  static func parse(_ text: String?) -> PhoneTTSBackend {
    guard let text, !text.isEmpty else { return .piperRyanHigh }
    return PhoneTTSBackend(rawValue: text.lowercased()) ?? .piperRyanHigh
  }

  var displayName: String {
    switch self {
    case .piperRyanHigh:
      return "Piper"
    }
  }

  var defaultVoice: String {
    switch self {
    case .piperRyanHigh:
      return "en_US-ryan-high"
    }
  }

  var availableVoices: [String] {
    switch self {
    case .piperRyanHigh:
      return ["en_US-ryan-high"]
    }
  }
}

struct PhoneTTSResult: Sendable {
  let backend: PhoneTTSBackend
  let voice: String
  let sampleRate: Int
  let audioSeconds: Double
  let elapsedSeconds: Double
  let firstAudioSeconds: Double
  let chunks: Int
  let audioBytes: Int

  var payload: [String: Any] {
    [
      "backend": backend.rawValue,
      "backend_name": backend.displayName,
      "voice": voice,
      "sample_rate": sampleRate,
      "audio_seconds": audioSeconds,
      "elapsed_seconds": elapsedSeconds,
      "first_audio_seconds": firstAudioSeconds,
      "chunks": chunks,
      "audio_bytes": audioBytes,
      "realtime_factor": elapsedSeconds > 0 ? audioSeconds / elapsedSeconds : 0.0
    ]
  }
}

struct PhoneTTSDownloadProgress: Equatable, Sendable {
  let title: String
  let detail: String
  let fractionCompleted: Double?

  static let loading = PhoneTTSDownloadProgress(
    title: "Loading TTS assets",
    detail: "Preparing Piper Ryan high...",
    fractionCompleted: nil
  )

  static func complete(detail: String) -> PhoneTTSDownloadProgress {
    PhoneTTSDownloadProgress(title: "TTS assets ready", detail: detail, fractionCompleted: 1)
  }
}

struct PhoneTTSBenchmarkRow: Sendable {
  let backend: PhoneTTSBackend
  let voice: String
  let result: PhoneTTSResult?
  let error: String?

  var payload: [String: Any] {
    if let result {
      return result.payload.merging(["ok": true]) { _, new in new }
    }
    return [
      "ok": false,
      "backend": backend.rawValue,
      "backend_name": backend.displayName,
      "voice": voice,
      "error": error ?? "unknown error"
    ]
  }
}

enum PhoneTTSError: LocalizedError {
  case unavailable(String)
  case emptyText
  case badDownload(URL)
  case badArchive(String)
  case missingAssets(String)
  case extractionBlocked(String)
  case synthesisFailed(String)

  var errorDescription: String? {
    switch self {
    case .unavailable(let detail):
      return detail
    case .emptyText:
      return "TTS text is empty."
    case .badDownload(let url):
      return "Failed to download Piper assets from \(url.absoluteString)."
    case .badArchive(let detail):
      return "Piper asset archive is invalid: \(detail)"
    case .missingAssets(let detail):
      return "Piper assets are missing: \(detail)"
    case .extractionBlocked(let detail):
      return "Piper asset extraction was blocked: \(detail)"
    case .synthesisFailed(let detail):
      return "Piper synthesis failed: \(detail)"
    }
  }
}

actor PhoneTTSRuntime {
  private static let piperArchiveURL = URL(
    string: "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/vits-piper-en_US-ryan-high.tar.bz2"
  )!
  private static let piperArchiveName = "vits-piper-en_US-ryan-high.tar.bz2"
  private static let piperRootName = "vits-piper-en_US-ryan-high"
  private static let piperModelName = "en_US-ryan-high.onnx"
  private static let piperTokensName = "tokens.txt"
  private static let piperReadyName = ".ready"

  private var synthesisBusy = false
  private var synthesisWaiters: [CheckedContinuation<Void, Never>] = []
  private var piper: PiperEngine?

  func releaseCachedModels(reason: String) async {
    guard !synthesisBusy else {
      AppLog.info("Piper TTS release skipped because synthesis is active: reason=\(reason)")
      return
    }

    if piper != nil {
      AppLog.info("Piper TTS releasing cached engine: reason=\(reason)")
      piper = nil
    } else {
      AppLog.info("Piper TTS release requested with no cached engine: reason=\(reason)")
    }
  }

  func synthesizeStreaming(
    text rawText: String,
    backend backendName: String?,
    voice requestedVoice: String?,
    onDownloadProgress: @escaping @MainActor @Sendable (PhoneTTSDownloadProgress?) -> Void = { _ in },
    onAudioChunk: @escaping @Sendable (_ chunk: Data, _ sampleRate: Int) async -> Void
  ) async throws -> PhoneTTSResult {
    let text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty else { throw PhoneTTSError.emptyText }

    let backend = PhoneTTSBackend.parse(backendName)
    let voice = normalizedVoice(requestedVoice, backend: backend)
    try Task.checkCancellation()
    await enterSynthesisQueue(backend: backend, voice: voice, chars: text.count)
    defer { leaveSynthesisQueue() }
    try Task.checkCancellation()

    switch backend {
    case .piperRyanHigh:
      return try await synthesizePiper(
        text: text,
        voice: voice,
        onDownloadProgress: onDownloadProgress,
        onAudioChunk: onAudioChunk
      )
    }
  }

  private func enterSynthesisQueue(backend: PhoneTTSBackend, voice: String, chars: Int) async {
    if !synthesisBusy {
      synthesisBusy = true
      AppLog.info("TTS synth queue acquired: backend=\(backend.rawValue), voice=\(voice), chars=\(chars)")
      return
    }

    AppLog.info("TTS synth queued behind active request: backend=\(backend.rawValue), voice=\(voice), chars=\(chars), waiters=\(synthesisWaiters.count + 1)")
    await withCheckedContinuation { continuation in
      synthesisWaiters.append(continuation)
    }
    AppLog.info("TTS synth queue acquired after wait: backend=\(backend.rawValue), voice=\(voice), chars=\(chars)")
  }

  private func leaveSynthesisQueue() {
    if synthesisWaiters.isEmpty {
      synthesisBusy = false
      AppLog.info("TTS synth queue released: waiters=0")
    } else {
      let continuation = synthesisWaiters.removeFirst()
      AppLog.info("TTS synth queue passed to next waiter: remaining_waiters=\(synthesisWaiters.count)")
      continuation.resume()
    }
  }

  func benchmark(text: String) async -> [PhoneTTSBenchmarkRow] {
    var rows: [PhoneTTSBenchmarkRow] = []
    for backend in PhoneTTSBackend.selectableCases {
      do {
        let result = try await synthesizeStreaming(
          text: text,
          backend: backend.rawValue,
          voice: backend.defaultVoice
        ) { _, _ in }
        rows.append(PhoneTTSBenchmarkRow(backend: backend, voice: backend.defaultVoice, result: result, error: nil))
      } catch {
        rows.append(PhoneTTSBenchmarkRow(backend: backend, voice: backend.defaultVoice, result: nil, error: AppLog.describe(error)))
      }
    }
    return rows
  }

  private func normalizedVoice(_ requestedVoice: String?, backend: PhoneTTSBackend) -> String {
    guard let requestedVoice else { return backend.defaultVoice }
    let trimmed = requestedVoice.trimmingCharacters(in: .whitespacesAndNewlines)
    return backend.availableVoices.contains(trimmed) ? trimmed : backend.defaultVoice
  }

  private func synthesizePiper(
    text: String,
    voice: String,
    onDownloadProgress: @escaping @MainActor @Sendable (PhoneTTSDownloadProgress?) -> Void,
    onAudioChunk: @escaping @Sendable (_ chunk: Data, _ sampleRate: Int) async -> Void
  ) async throws -> PhoneTTSResult {
    let engine = try await ensurePiper(onDownloadProgress: onDownloadProgress)
    try Task.checkCancellation()

    let started = Date()
    AppLog.info("Piper TTS synth starting: voice=\(voice), chars=\(text.count), sample_rate=\(engine.sampleRate)")

    let state = PiperStreamState(started: started)
    let stream = AsyncThrowingStream<Data, Error> { continuation in
      state.install(continuation)
    }

    let generationTask = Task.detached(priority: .userInitiated) {
      try await engine.generate(text: text, state: state)
    }

    do {
      for try await chunk in stream {
        try Task.checkCancellation()
        await onAudioChunk(chunk, engine.sampleRate)
      }
      try await generationTask.value
    } catch {
      state.cancel()
      generationTask.cancel()
      throw error
    }

    let snapshot = state.snapshot()
    let elapsed = Date().timeIntervalSince(started)
    let audioSeconds = Double(snapshot.sampleCount) / Double(engine.sampleRate)
    AppLog.info(String(format: "Piper TTS synth complete: %.2fs audio, %.2fs wall, chunks=%d", audioSeconds, elapsed, snapshot.chunks))
    return PhoneTTSResult(
      backend: .piperRyanHigh,
      voice: voice,
      sampleRate: engine.sampleRate,
      audioSeconds: audioSeconds,
      elapsedSeconds: elapsed,
      firstAudioSeconds: snapshot.firstAudioSeconds,
      chunks: snapshot.chunks,
      audioBytes: snapshot.audioBytes
    )
  }

  private func ensurePiper(
    onDownloadProgress: @escaping @MainActor @Sendable (PhoneTTSDownloadProgress?) -> Void
  ) async throws -> PiperEngine {
    if let piper {
      await onDownloadProgress(nil)
      return piper
    }

    let started = Date()
    AppLog.info("Piper TTS load starting")
    let assets = try await ensurePiperAssets(onDownloadProgress: onDownloadProgress)
    await onDownloadProgress(.loading)
    let engine = try PiperEngine(paths: assets)
    piper = engine
    await onDownloadProgress(nil)
    AppLog.info(String(format: "Piper TTS load complete in %.2fs, sample_rate=%d", Date().timeIntervalSince(started), engine.sampleRate))
    return engine
  }

  private func ensurePiperAssets(
    onDownloadProgress: @escaping @MainActor @Sendable (PhoneTTSDownloadProgress?) -> Void
  ) async throws -> PiperAssetPaths {
    let root = try Self.piperAssetsRoot()
    let ready = root.appendingPathComponent(Self.piperReadyName)
    let paths = PiperAssetPaths(root: root)
    if FileManager.default.fileExists(atPath: ready.path), paths.areUsable {
      await onDownloadProgress(nil)
      return paths
    }

    if FileManager.default.fileExists(atPath: root.path) {
      try FileManager.default.removeItem(at: root)
    }
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)

    let downloadStarted = Date()
    await onDownloadProgress(
      PhoneTTSDownloadProgress(
        title: "Downloading TTS assets",
        detail: "Piper Ryan high: starting download...",
        fractionCompleted: nil
      )
    )
    let archiveURL = try await download(
      from: Self.piperArchiveURL,
      suggestedFileName: Self.piperArchiveName
    ) { bytesWritten, totalBytes in
      let elapsed = max(0.1, Date().timeIntervalSince(downloadStarted))
      let speed = Double(bytesWritten) / elapsed
      let progress = Self.downloadProgress(bytesWritten: bytesWritten, totalBytes: totalBytes, speed: speed)
      Task { @MainActor in
        onDownloadProgress(progress)
      }
    }
    defer {
      try? FileManager.default.removeItem(at: archiveURL)
    }

    try Task.checkCancellation()
    await onDownloadProgress(
      PhoneTTSDownloadProgress(
        title: "Extracting TTS assets",
        detail: "Piper Ryan high: decompressing archive...",
        fractionCompleted: nil
      )
    )
    try extractPiperArchive(archiveURL: archiveURL, root: root, onDownloadProgress: onDownloadProgress)
    try validatePiperAssets(paths)
    FileManager.default.createFile(atPath: ready.path, contents: Data(), attributes: nil)
    await onDownloadProgress(PhoneTTSDownloadProgress.complete(detail: "Piper Ryan high is ready."))
    return paths
  }

  private func extractPiperArchive(
    archiveURL: URL,
    root: URL,
    onDownloadProgress: @escaping @MainActor @Sendable (PhoneTTSDownloadProgress?) -> Void
  ) throws {
    let archiveData = try Data(contentsOf: archiveURL, options: [.mappedIfSafe])
    let tarData = try BZip2.decompress(data: archiveData)
    let entries = try TarContainer.open(container: tarData)
    let expectedPrefix = Self.piperRootName + "/"
    let fileEntries = entries.filter { $0.data != nil }
    let total = max(1, fileEntries.count)
    var completed = 0

    for entry in entries {
      let name = entry.info.name
      guard name.hasPrefix(expectedPrefix) else { continue }
      let relative = String(name.dropFirst(expectedPrefix.count))
      guard !relative.isEmpty else { continue }
      guard !relative.contains(".."), !relative.hasPrefix("/") else {
        throw PhoneTTSError.extractionBlocked(relative)
      }

      let destination = root.appendingPathComponent(relative)
      if let data = entry.data {
        try FileManager.default.createDirectory(
          at: destination.deletingLastPathComponent(),
          withIntermediateDirectories: true
        )
        try data.write(to: destination, options: [.atomic])
        completed += 1
        if completed == total || completed % 25 == 0 {
          let fraction = min(1, max(0, Double(completed) / Double(total)))
          let percent = Int((fraction * 100).rounded())
          let completedFiles = completed
          let totalFiles = total
          Task { @MainActor in
            onDownloadProgress(
              PhoneTTSDownloadProgress(
                title: "Extracting TTS assets",
                detail: "Piper Ryan high: \(percent)% (\(completedFiles)/\(totalFiles) files)",
                fractionCompleted: fraction
              )
            )
          }
        }
      } else {
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
      }
    }
  }

  private func validatePiperAssets(_ paths: PiperAssetPaths) throws {
    guard FileManager.default.fileExists(atPath: paths.model.path) else {
      throw PhoneTTSError.missingAssets(paths.model.lastPathComponent)
    }
    guard FileManager.default.fileExists(atPath: paths.tokens.path) else {
      throw PhoneTTSError.missingAssets(paths.tokens.lastPathComponent)
    }
    guard FileManager.default.fileExists(atPath: paths.espeakData.path) else {
      throw PhoneTTSError.missingAssets(paths.espeakData.lastPathComponent)
    }
  }

  private func download(
    from remoteURL: URL,
    suggestedFileName: String,
    progress: @escaping @Sendable (Int64, Int64?) -> Void
  ) async throws -> URL {
    let delegate = TTSAssetDownloadDelegate(
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

  private static func piperAssetsRoot() throws -> URL {
    let appSupport = try FileManager.default.url(
      for: .applicationSupportDirectory,
      in: .userDomainMask,
      appropriateFor: nil,
      create: true
    )
    let root = appSupport
      .appendingPathComponent("TTS", isDirectory: true)
      .appendingPathComponent("Piper", isDirectory: true)
      .appendingPathComponent(piperRootName, isDirectory: true)
    return root
  }

  private static func downloadProgress(bytesWritten: Int64, totalBytes: Int64?, speed: Double) -> PhoneTTSDownloadProgress {
    let size = formatBytes(bytesWritten)
    let speedText = formatSpeed(speed)
    if let totalBytes, totalBytes > 0 {
      let fraction = min(1, max(0, Double(bytesWritten) / Double(totalBytes)))
      let percent = Int((fraction * 100).rounded())
      let total = formatBytes(totalBytes)
      return PhoneTTSDownloadProgress(
        title: "Downloading TTS assets",
        detail: "Piper Ryan high: \(percent)% (\(size) / \(total), \(speedText))",
        fractionCompleted: fraction
      )
    }
    return PhoneTTSDownloadProgress(
      title: "Downloading TTS assets",
      detail: "Piper Ryan high: \(size), \(speedText)",
      fractionCompleted: nil
    )
  }

  nonisolated static func pcmS16LEData(samples: [Float]) -> Data {
    samples.withUnsafeBufferPointer { buffer in
      guard let base = buffer.baseAddress else { return Data() }
      return pcmS16LEData(buffer: base, count: buffer.count)
    }
  }

  nonisolated static func pcmS16LEData(buffer: UnsafePointer<Float>, count: Int) -> Data {
    var data = Data()
    data.reserveCapacity(count * 2)
    for index in 0..<count {
      let clamped = max(-1.0, min(1.0, buffer[index]))
      var value = Int16((clamped * 32767.0).rounded()).littleEndian
      withUnsafeBytes(of: &value) { bytes in
        data.append(contentsOf: bytes)
      }
    }
    return data
  }

  private static func formatBytes(_ bytes: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
  }

  private static func formatSpeed(_ bytesPerSecond: Double) -> String {
    let bytes = Int64(max(0, bytesPerSecond))
    return "\(formatBytes(bytes))/s"
  }
}

private struct PiperAssetPaths: Sendable {
  let root: URL

  var model: URL { root.appendingPathComponent("en_US-ryan-high.onnx") }
  var tokens: URL { root.appendingPathComponent("tokens.txt") }
  var espeakData: URL { root.appendingPathComponent("espeak-ng-data", isDirectory: true) }

  var areUsable: Bool {
    FileManager.default.fileExists(atPath: model.path)
      && FileManager.default.fileExists(atPath: tokens.path)
      && FileManager.default.fileExists(atPath: espeakData.path)
  }
}

private final class PiperEngine: @unchecked Sendable {
  let tts: OpaquePointer
  let sampleRate: Int

  init(paths: PiperAssetPaths) throws {
    let strings = CStringArena([
      paths.model.path,
      "",
      paths.tokens.path,
      paths.espeakData.path,
      "",
      "cpu",
      "",
      ""
    ])
    defer { strings.release() }

    guard strings.count == 8 else {
      throw PhoneTTSError.synthesisFailed("failed to prepare Piper config strings")
    }

    var config = SherpaOnnxOfflineTtsConfig()
    config.model.vits.model = strings[0]
    config.model.vits.lexicon = strings[1]
    config.model.vits.tokens = strings[2]
    config.model.vits.data_dir = strings[3]
    config.model.vits.noise_scale = 0.667
    config.model.vits.noise_scale_w = 0.8
    config.model.vits.length_scale = 1.0
    config.model.vits.dict_dir = strings[4]
    config.model.num_threads = 2
    config.model.debug = 0
    config.model.provider = strings[5]
    config.rule_fsts = strings[6]
    config.rule_fars = strings[7]
    config.max_num_sentences = 1
    config.silence_scale = 0.2

    guard let created = withUnsafePointer(to: &config, { SherpaOnnxCreateOfflineTts($0) }) else {
      throw PhoneTTSError.synthesisFailed("SherpaOnnxCreateOfflineTts returned NULL")
    }

    self.tts = created
    self.sampleRate = Int(SherpaOnnxOfflineTtsSampleRate(created))
    guard sampleRate > 0 else {
      SherpaOnnxDestroyOfflineTts(created)
      throw PhoneTTSError.synthesisFailed("Piper engine reported invalid sample rate")
    }
  }

  deinit {
    SherpaOnnxDestroyOfflineTts(tts)
  }

  func generate(text: String, state: PiperStreamState) async throws {
    if Task.isCancelled {
      state.finish(throwing: CancellationError())
      throw CancellationError()
    }

    var config = SherpaOnnxGenerationConfig()
    config.silence_scale = 0.2
    config.speed = 1.0
    config.sid = 0

    let audio = text.withCString { cText in
      withUnsafePointer(to: &config) { configPointer in
        SherpaOnnxOfflineTtsGenerateWithConfig(
          tts,
          cText,
          configPointer,
          piperProgressCallback,
          Unmanaged.passUnretained(state).toOpaque()
        )
      }
    }

    guard let audio else {
      state.finish(throwing: PhoneTTSError.synthesisFailed("SherpaOnnxOfflineTtsGenerateWithConfig returned NULL"))
      throw PhoneTTSError.synthesisFailed("SherpaOnnxOfflineTtsGenerateWithConfig returned NULL")
    }

    if !state.hasChunks, let samples = audio.pointee.samples, audio.pointee.n > 0 {
      state.yield(samples: samples, count: Int(audio.pointee.n))
    }
    SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio)
    state.finish()
  }
}

private typealias PiperProgressCallback = @convention(c) (
  UnsafePointer<Float>?,
  Int32,
  Float,
  UnsafeMutableRawPointer?
) -> Int32

private let piperProgressCallback: PiperProgressCallback = { samples, count, _, rawState in
  guard let rawState, let samples, count > 0 else { return 1 }
  let state = Unmanaged<PiperStreamState>.fromOpaque(rawState).takeUnretainedValue()
  guard !state.isCancelled else { return 0 }
  state.yield(samples: samples, count: Int(count))
  return 1
}

private struct PiperStreamSnapshot: Sendable {
  let chunks: Int
  let audioBytes: Int
  let sampleCount: Int
  let firstAudioSeconds: Double
}

private final class PiperStreamState: @unchecked Sendable {
  private let lock = NSLock()
  private let started: Date
  private var continuation: AsyncThrowingStream<Data, Error>.Continuation?
  private var chunks = 0
  private var audioBytes = 0
  private var sampleCount = 0
  private var firstAudioSeconds = 0.0
  private var cancelled = false
  private var finished = false

  init(started: Date) {
    self.started = started
  }

  var isCancelled: Bool {
    lock.lock()
    defer { lock.unlock() }
    return cancelled
  }

  var hasChunks: Bool {
    lock.lock()
    defer { lock.unlock() }
    return chunks > 0
  }

  func install(_ continuation: AsyncThrowingStream<Data, Error>.Continuation) {
    lock.lock()
    self.continuation = continuation
    lock.unlock()
  }

  func yield(samples: UnsafePointer<Float>, count: Int) {
    let chunk = PhoneTTSRuntime.pcmS16LEData(buffer: samples, count: count)
    let continuation: AsyncThrowingStream<Data, Error>.Continuation?
    lock.lock()
    guard !finished else {
      lock.unlock()
      return
    }
    if chunks == 0 {
      firstAudioSeconds = Date().timeIntervalSince(started)
    }
    chunks += 1
    audioBytes += chunk.count
    sampleCount += count
    continuation = self.continuation
    lock.unlock()
    continuation?.yield(chunk)
  }

  func snapshot() -> PiperStreamSnapshot {
    lock.lock()
    defer { lock.unlock() }
    return PiperStreamSnapshot(
      chunks: chunks,
      audioBytes: audioBytes,
      sampleCount: sampleCount,
      firstAudioSeconds: firstAudioSeconds
    )
  }

  func cancel() {
    lock.lock()
    cancelled = true
    let continuation = self.continuation
    lock.unlock()
    continuation?.finish(throwing: CancellationError())
  }

  func finish(throwing error: Error? = nil) {
    let continuation: AsyncThrowingStream<Data, Error>.Continuation?
    lock.lock()
    guard !finished else {
      lock.unlock()
      return
    }
    finished = true
    continuation = self.continuation
    lock.unlock()
    if let error {
      continuation?.finish(throwing: error)
    } else {
      continuation?.finish()
    }
  }
}

private final class CStringArena {
  private var storage: [UnsafeMutablePointer<CChar>] = []

  init(_ strings: [String]) {
    storage = strings.compactMap { strdup($0) }
  }

  var count: Int { storage.count }

  subscript(index: Int) -> UnsafePointer<CChar>? {
    UnsafePointer(storage[index])
  }

  func release() {
    for pointer in storage {
      free(pointer)
    }
    storage.removeAll()
  }
}

private final class TTSAssetDownloadDelegate: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
  var continuation: CheckedContinuation<URL, Error>?

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
        .appendingPathExtension((suggestedFileName as NSString).pathExtension.nonEmptyOr("tar.bz2"))
      if FileManager.default.fileExists(atPath: tempURL.path) {
        try FileManager.default.removeItem(at: tempURL)
      }
      try FileManager.default.moveItem(at: location, to: tempURL)
      resume(.success(tempURL))
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

  private func resume(_ result: Result<URL, Error>) {
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

private extension String {
  func nonEmptyOr(_ fallback: String) -> String {
    isEmpty ? fallback : self
  }
}
