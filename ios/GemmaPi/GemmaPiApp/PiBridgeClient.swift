import Foundation
import AVFoundation
import UIKit

@MainActor
final class PiBridgeClient: ObservableObject {
  @Published var bridgeURLString = "ws://pi3:8765/worker"
  @Published private(set) var connectionState = "Disconnected"
  @Published private(set) var lastPromptSnippet = ""
  @Published private(set) var generatedSnippet = ""
  @Published private(set) var totalRequests = 0
  @Published private(set) var inputTokens = 0
  @Published private(set) var outputTokens = 0
  @Published private(set) var lastTokensPerSecond = 0.0
  @Published private(set) var lastError = ""
  @Published private(set) var runtimeName: String
  @Published private(set) var runtimeStatus: String
  @Published private(set) var runtimeReady = false
  @Published private(set) var isLoadingRuntime = false
  @Published private(set) var loadedModelPath = ""
  @Published private(set) var loadedModelSize = ""
  @Published private(set) var runtimeLoadCount = 0
  @Published var localTestPrompt = "hi"
  @Published private(set) var localTestResponse = ""
  @Published private(set) var localTestStatus = "Load the model to run a local test."
  @Published private(set) var isRunningLocalTest = false
  @Published private(set) var isRecordingAudio = false
  @Published private(set) var audioInputStatus = "Hold mic to send a raw Gemma audio prompt."
  @Published private(set) var isSpeakingLocalTest = false
  @Published var selectedTTSBackend = PhoneTTSBackend.fluidKokoroAne.rawValue
  @Published var selectedTTSVoice = PhoneTTSBackend.fluidKokoroAne.defaultVoice
  @Published private(set) var lastProbeResponse = ""
  @Published private(set) var backend: InferenceBackend = .gpu
  @Published private(set) var poseRequests = 0
  @Published private(set) var lastPoseStatus = "No pose requests yet."
  @Published private(set) var lastPoseLatency = 0.0
  @Published private(set) var lastPoseBackend = ""
  @Published private(set) var lastPoseModel = ""
  @Published private(set) var ttsRequests = 0
  @Published private(set) var lastTTSStatus = "No TTS requests yet."
  @Published private(set) var lastTTSBackend = ""
  @Published private(set) var lastTTSLatency = 0.0

  private let runtime: GemmaRuntime
  private let poseRuntime = PoseRuntime()
  private let ttsRuntime = PhoneTTSRuntime()
  private var localAudioRecorder: AVAudioRecorder?
  private var retiredAudioRecorders: [AVAudioRecorder] = []
  private var localAudioURL: URL?
  private var localAudioPlayer: AVAudioPlayer?
  private var socket: URLSessionWebSocketTask?
  private var receiveTask: Task<Void, Never>?
  private var generationTask: Task<Void, Never>?
  private var poseTask: Task<Void, Never>?
  private var ttsTask: Task<Void, Never>?
  private var pendingPoseFrames: [String: PendingPoseFrame] = [:]
  private static let poseBinaryMagic = Data("G4POSE01".utf8)
  private static let generateBinaryMagic = Data("G4GEN01".utf8)
  private static let ttsBinaryMagic = Data("G4TTS01".utf8)
  private static let minimumAudioPromptSeconds = 0.20
  private static let maximumAudioPromptSeconds = 20.0

  init(runtime: GemmaRuntime = RuntimeFactory.make()) {
    self.runtime = runtime
    self.runtimeName = runtime.name
    self.runtimeStatus = runtime.status
    self.runtimeReady = runtime.isReady
    if let launchBridgeURL = Self.launchBridgeURL() {
      self.bridgeURLString = launchBridgeURL
    }
  }

  var isConnected: Bool {
    socket != nil
  }

  func useRecommendedBackend() {
    backend = .gpu
  }

  func loadRuntime(modelURL: URL?, projectorURL: URL? = nil) async {
    runtimeLoadCount += 1
    runtimeReady = runtime.isReady

    guard let modelURL else {
      runtimeStatus = "Download the model before loading runtime."
      loadedModelPath = ""
      loadedModelSize = ""
      AppLog.error("Gemma runtime load failed before start: no model URL")
      return
    }

    guard FileManager.default.fileExists(atPath: modelURL.path) else {
      runtimeStatus = "Load failed: file is missing at \(modelURL.path)"
      loadedModelPath = modelURL.path
      loadedModelSize = ""
      runtimeReady = false
      AppLog.error("Gemma runtime load failed before start: file missing at \(modelURL.path)")
      return
    }

    let size = Self.fileSize(modelURL)
    loadedModelPath = modelURL.path
    loadedModelSize = size.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) } ?? "unknown size"
    isLoadingRuntime = true
    runtimeStatus = "Loading \(modelURL.lastPathComponent) (\(loadedModelSize))..."
    AppLog.info(
      "Gemma runtime load starting: runtime=\(runtime.name), path=\(modelURL.path), size=\(loadedModelSize), projector=\(projectorURL?.path ?? "none"), backend=\(backend.rawValue)"
    )

    do {
      try await runtime.loadModel(at: modelURL, projectorURL: projectorURL, backend: backend)
      runtimeReady = runtime.isReady
      runtimeStatus = runtime.status
      isLoadingRuntime = false
      AppLog.info("Gemma runtime load finished: ready=\(runtimeReady), runtime=\(runtime.name), status=\(runtime.status)")
      Task { [weak self] in
        await self?.sendReady()
      }
      await runLocalTest(prompt: "hi", isProbe: true)
    } catch {
      runtimeReady = false
      isLoadingRuntime = false
      let detail = AppLog.describe(error)
      runtimeStatus = "Load failed: \(detail)"
      AppLog.error("Gemma runtime load failed: \(detail)")
    }
  }

  func sendLocalTestPrompt() async {
    let prompt = localTestPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !prompt.isEmpty else {
      localTestStatus = "Enter a prompt first."
      return
    }

    await runLocalTest(prompt: prompt, isProbe: false)
  }

  func startAudioCapture() async {
    guard !isRecordingAudio else { return }
    guard !isRunningLocalTest else { return }

    do {
      AppLog.info("Local Gemma audio capture starting: requesting microphone permission")
      let microphoneAllowed = await Self.requestMicrophonePermission()
      AppLog.info("Local microphone permission allowed: \(microphoneAllowed)")
      guard microphoneAllowed else {
        throw AudioInputError.permissionDenied("Microphone permission was denied.")
      }

      retireStaleAudioRecorderIfNeeded()
      audioInputStatus = "Recording raw audio..."

      let session = AVAudioSession.sharedInstance()
      try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .allowBluetoothHFP])
      try session.setActive(true, options: .notifyOthersOnDeactivation)
      let url = FileManager.default.temporaryDirectory
        .appendingPathComponent("gemma-local-audio-\(UUID().uuidString)")
        .appendingPathExtension("wav")
      let settings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM,
        AVSampleRateKey: 16_000,
        AVNumberOfChannelsKey: 1,
        AVLinearPCMBitDepthKey: 16,
        AVLinearPCMIsFloatKey: false,
        AVLinearPCMIsBigEndianKey: false
      ]
      let recorder = try AVAudioRecorder(url: url, settings: settings)
      recorder.isMeteringEnabled = true
      recorder.prepareToRecord()
      guard recorder.record() else {
        throw AudioInputError.recordingFailed("AVAudioRecorder.record returned false.")
      }
      localAudioRecorder = recorder
      localAudioURL = url
      isRecordingAudio = true
      AppLog.info("Local Gemma audio capture recording started: path=\(url.path), sample_rate=16000, channels=1, format=pcm_s16le")
    } catch {
      stopAudioCapture()
      audioInputStatus = "Audio failed: \(error.localizedDescription)"
      AppLog.error("Local Gemma audio capture failed: \(AppLog.describe(error))")
    }
  }

  func finishAudioCaptureAndSend() async {
    guard isRecordingAudio else { return }
    AppLog.info("Local Gemma audio capture release received")
    let url = localAudioURL
    let recorderDuration = localAudioRecorder?.currentTime ?? 0
    stopAudioCapture()

    guard let url, FileManager.default.fileExists(atPath: url.path) else {
      audioInputStatus = "No audio recorded."
      AppLog.error("Local Gemma audio capture finished without a WAV file")
      return
    }
    guard let audio = try? Data(contentsOf: url), !audio.isEmpty else {
      audioInputStatus = "Recorded audio was empty."
      AppLog.error("Local Gemma audio capture file was empty: \(url.path)")
      return
    }

    let diagnostics = Self.audioFileDiagnostics(url: url, byteCount: audio.count)
    AppLog.info(
      String(
        format: "Local Gemma audio prompt file: bytes=%d, recorder_duration=%.3fs, %@",
        audio.count,
        recorderDuration,
        diagnostics.logSummary
      )
    )

    guard diagnostics.durationSeconds >= Self.minimumAudioPromptSeconds else {
      audioInputStatus = String(format: "Recorded audio was too short: %.2fs.", diagnostics.durationSeconds)
      AppLog.error("Local Gemma audio prompt rejected as too short: \(diagnostics.logSummary)")
      return
    }

    guard diagnostics.durationSeconds <= Self.maximumAudioPromptSeconds else {
      audioInputStatus = String(format: "Recorded audio is too long: %.1fs max.", Self.maximumAudioPromptSeconds)
      AppLog.error("Local Gemma audio prompt rejected as too long: \(diagnostics.logSummary)")
      return
    }

    let prompt = "Answer the spoken user request. Do not use emojis."
    localTestPrompt = prompt
    audioInputStatus = String(
      format: "Sending %@ raw audio (%.2fs) to Gemma.",
      Self.formatBytes(Int64(audio.count)),
      diagnostics.durationSeconds
    )
    AppLog.info("Local Gemma audio prompt sending: bytes=\(audio.count), duration=\(String(format: "%.3f", diagnostics.durationSeconds))s, path=\(url.path)")
    await runLocalTest(
      prompt: prompt,
      media: [GemmaMediaInput(data: audio, mimeType: "audio/wav", displayName: url.lastPathComponent)],
      isProbe: false
    )
  }

  func runAudioCaptureSmokeTest(seconds: Double = 1.25) async {
    guard runtimeReady else {
      audioInputStatus = "Audio smoke skipped: runtime is not ready."
      AppLog.error("Local Gemma audio smoke skipped because runtime is not ready")
      return
    }
    guard !isRunningLocalTest, !isRecordingAudio else {
      audioInputStatus = "Audio smoke skipped: app is busy."
      AppLog.error("Local Gemma audio smoke skipped because app is busy")
      return
    }

    AppLog.info(String(format: "Local Gemma audio smoke starting: %.2fs recording", seconds))
    await startAudioCapture()
    guard isRecordingAudio else {
      AppLog.error("Local Gemma audio smoke could not start recording")
      return
    }
    let nanoseconds = UInt64(max(0.2, seconds) * 1_000_000_000)
    try? await Task.sleep(nanoseconds: nanoseconds)
    await finishAudioCaptureAndSend()
    AppLog.info("Local Gemma audio smoke finished")
  }

  func speakLocalTestResponse() async {
    let text = localTestResponse.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty else {
      lastTTSStatus = "No local response to speak."
      return
    }
    await playLocalTTS(text: text, statusPrefix: "Spoke")
  }

  func previewSelectedTTSVoice() async {
    await playLocalTTS(text: "hello, how are you?", statusPrefix: "Voice preview")
  }

  private func playLocalTTS(text: String, statusPrefix: String) async {
    let backend = PhoneTTSBackend.parse(selectedTTSBackend)
    guard PhoneTTSBackend.selectableCases.contains(backend) else {
      lastTTSStatus = "\(backend.displayName) is not enabled because it has not been validated with Gemma loaded."
      AppLog.error("Blocked unsupported local TTS backend from UI playback: \(backend.rawValue)")
      return
    }
    guard !isSpeakingLocalTest else { return }
    isSpeakingLocalTest = true
    defer { isSpeakingLocalTest = false }

    do {
      localAudioPlayer?.stop()
      lastTTSStatus = "\(statusPrefix): synthesizing \(selectedTTSVoice)..."
      let collector = AudioChunkCollector()
      let result = try await ttsRuntime.synthesizeStreaming(
        text: text,
        backend: selectedTTSBackend,
        voice: selectedTTSVoice
      ) { chunk in
        await collector.append(chunk)
      }
      let pcm = await collector.data()
      let wav = Self.wavData(pcm: pcm, sampleRate: result.sampleRate, channels: 1)
      try Self.configurePlaybackSession()
      let player = try AVAudioPlayer(data: wav)
      localAudioPlayer = player
      player.prepareToPlay()
      player.play()
      ttsRequests += 1
      lastTTSBackend = result.backend.displayName
      lastTTSLatency = result.elapsedSeconds
      lastTTSStatus = String(
        format: "%@: %@ %.2fs audio, first %.2fs, wall %.2fs",
        statusPrefix,
        result.backend.displayName,
        result.audioSeconds,
        result.firstAudioSeconds,
        result.elapsedSeconds
      )
    } catch {
      lastTTSStatus = "TTS failed: \(error.localizedDescription)"
      AppLog.error("Local TTS playback failed: \(AppLog.describe(error))")
    }
  }

  func connect() {
    guard socket == nil else { return }
    guard let url = URL(string: bridgeURLString) else {
      lastError = "Invalid bridge URL."
      return
    }

    let task = URLSession.shared.webSocketTask(with: url)
    socket = task
    connectionState = "Connecting"
    lastError = ""
    task.resume()

    receiveTask = Task { [weak self] in
      await self?.sendReady()
      await self?.receiveLoop()
    }
  }

  func disconnect() {
    generationTask?.cancel()
    poseTask?.cancel()
    ttsTask?.cancel()
    receiveTask?.cancel()
    socket?.cancel(with: .goingAway, reason: nil)
    socket = nil
    connectionState = "Disconnected"
  }

  func cancelGeneration() {
    generationTask?.cancel()
    runtime.cancel()
  }

  private func stopAudioCapture() {
    let recorder = localAudioRecorder
    localAudioRecorder = nil
    localAudioURL = nil
    isRecordingAudio = false
    if let recorder {
      if recorder.isRecording {
        recorder.stop()
      }
      retireAudioRecorder(recorder, reason: "capture stopped")
    }
    scheduleAudioSessionDeactivation()
  }

  private func retireStaleAudioRecorderIfNeeded() {
    guard let recorder = localAudioRecorder else { return }
    localAudioRecorder = nil
    if recorder.isRecording {
      recorder.stop()
    }
    retireAudioRecorder(recorder, reason: "stale recorder before new capture")
  }

  private func retireAudioRecorder(_ recorder: AVAudioRecorder, reason: String) {
    retiredAudioRecorders.append(recorder)
    AppLog.info("Local Gemma audio recorder retired temporarily: reason=\(reason), retired_count=\(retiredAudioRecorders.count)")

    Task { [weak self, recorder] in
      try? await Task.sleep(nanoseconds: 1_500_000_000)
      await MainActor.run {
        guard let self else { return }
        self.retiredAudioRecorders.removeAll { $0 === recorder }
        AppLog.info("Local Gemma audio recorder released after callback drain: retired_count=\(self.retiredAudioRecorders.count)")
      }
    }
  }

  private func scheduleAudioSessionDeactivation() {
    Task { [weak self] in
      try? await Task.sleep(nanoseconds: 600_000_000)
      await MainActor.run {
        guard let self, !self.isRecordingAudio else { return }
        do {
          try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
          AppLog.info("Local Gemma audio capture session deactivated")
        } catch {
          AppLog.error("Local Gemma audio capture session deactivate failed: \(AppLog.describe(error))")
        }
      }
    }
  }

  private nonisolated static func requestMicrophonePermission() async -> Bool {
    await withCheckedContinuation { continuation in
      AVAudioSession.sharedInstance().requestRecordPermission { allowed in
        continuation.resume(returning: allowed)
      }
    }
  }

  private nonisolated static func configurePlaybackSession() throws {
    let session = AVAudioSession.sharedInstance()
    try session.setCategory(.playback, mode: .spokenAudio, options: [.duckOthers])
    try session.setActive(true, options: .notifyOthersOnDeactivation)
  }

  private func receiveLoop() async {
    connectionState = "Connected"

    while let socket {
      do {
        let message = try await socket.receive()
        switch message {
        case .string(let text):
          await handleMessage(text)
        case .data(let data):
          if data.prefix(Self.poseBinaryMagic.count) == Self.poseBinaryMagic
            || data.prefix(Self.generateBinaryMagic.count) == Self.generateBinaryMagic
          {
            await handleBinaryMessage(data)
          } else if let text = String(data: data, encoding: .utf8) {
            await handleMessage(text)
          }
        @unknown default:
          break
        }
      } catch {
        if !Task.isCancelled {
          lastError = error.localizedDescription
          connectionState = "Disconnected"
          AppLog.error("Gemma bridge receive loop failed: \(AppLog.describe(error))")
        }
        self.socket = nil
        break
      }
    }
  }

  private func handleMessage(_ text: String) async {
    guard let data = text.data(using: .utf8) else { return }

    do {
      let request = try JSONDecoder().decode(BridgeRequest.self, from: data)
      switch request.type {
      case "ping":
        await sendJSON(["type": "pong", "time": Date().timeIntervalSince1970])
      case "generate":
        startGeneration(request)
      case "pose":
        startPose(request)
      case "pose_start":
        startChunkedPose(request)
      case "pose_chunk":
        appendPoseChunk(request)
      case "tts":
        startTTS(request)
      case "tts_benchmark":
        startTTSBenchmark(request)
      default:
        break
      }
    } catch {
      lastError = "Bad bridge message: \(error.localizedDescription)"
      AppLog.error("Gemma bridge received bad message: \(AppLog.describe(error))")
    }
  }

  private func handleBinaryMessage(_ data: Data) async {
    do {
      if data.prefix(Self.generateBinaryMagic.count) == Self.generateBinaryMagic {
        let parsed = try Self.parseGenerateBinaryFrame(data)
        generationTask?.cancel()
        generationTask = Task { [weak self] in
          guard let self else { return }
          await self.runGeneration(
            id: parsed.header.id,
            prompt: parsed.header.prompt,
            media: parsed.media,
            maxTokens: parsed.header.maxTokens ?? 128
          )
        }
        return
      }

      let parsed = try Self.parsePoseBinaryFrame(data)
      let request = try JSONDecoder().decode(BridgeRequest.self, from: parsed.header)
      guard request.type == "pose_binary", let requestID = request.id else {
        throw PoseRuntimeError.invalidFrame("binary bridge message must be pose_binary with id")
      }
      poseTask?.cancel()
      poseTask = Task { [weak self] in
        guard let self else { return }
        await self.runPose(
          id: requestID,
          format: request.format,
          width: request.width,
          height: request.height,
          dataText: nil,
          frameData: parsed.frame,
          poseBackendText: request.poseBackend,
          poseModelText: request.poseModel
        )
      }
    } catch {
      lastError = "Bad binary bridge message: \(error.localizedDescription)"
      AppLog.error("Gemma bridge received bad binary message: \(AppLog.describe(error))")
    }
  }

  private func startPose(_ request: BridgeRequest) {
    guard let requestID = request.id else { return }

    poseTask?.cancel()
    poseTask = Task { [weak self] in
      guard let self else { return }
      await self.runPose(id: requestID, request: request)
    }
  }

  private func startChunkedPose(_ request: BridgeRequest) {
    guard let requestID = request.id else { return }
    guard let format = request.format?.lowercased(),
          let width = request.width,
          let height = request.height,
          let chunkCount = request.chunkCount,
          chunkCount > 0
    else {
      Task { await sendJSON(["type": "error", "id": requestID, "message": "pose_start needs format, width, height, and chunk_count"]) }
      return
    }

    pendingPoseFrames[requestID] = PendingPoseFrame(
      format: format,
      width: width,
      height: height,
      poseBackend: request.poseBackend,
      poseModel: request.poseModel,
      chunks: Array(repeating: nil, count: chunkCount)
    )
  }

  private func appendPoseChunk(_ request: BridgeRequest) {
    guard let requestID = request.id,
          let chunkIndex = request.chunkIndex,
          let chunk = request.data,
          var pending = pendingPoseFrames[requestID]
    else { return }

    guard chunkIndex >= 0 && chunkIndex < pending.chunks.count else {
      Task { await sendJSON(["type": "error", "id": requestID, "message": "pose_chunk index out of range"]) }
      return
    }

    pending.chunks[chunkIndex] = chunk
    pendingPoseFrames[requestID] = pending

    guard pending.chunks.allSatisfy({ $0 != nil }) else { return }
    pendingPoseFrames.removeValue(forKey: requestID)
    let dataText = pending.chunks.compactMap { $0 }.joined()
    poseTask?.cancel()
    poseTask = Task { [weak self] in
      guard let self else { return }
      await self.runPose(
        id: requestID,
        format: pending.format,
        width: pending.width,
        height: pending.height,
        dataText: dataText,
        poseBackendText: pending.poseBackend,
        poseModelText: pending.poseModel
      )
    }
  }

  private func startGeneration(_ request: BridgeRequest) {
    guard let requestID = request.id, let prompt = request.prompt else { return }

    generationTask?.cancel()
    generationTask = Task { [weak self] in
      guard let self else { return }
      await self.runGeneration(id: requestID, prompt: prompt, media: [], maxTokens: request.maxTokens ?? 128)
    }
  }

  private func startTTS(_ request: BridgeRequest) {
    guard let requestID = request.id else { return }
    let text = request.text ?? generatedSnippet
    ttsTask?.cancel()
    ttsTask = Task { [weak self] in
      guard let self else { return }
      await self.runTTS(
        id: requestID,
        text: text,
        backend: request.ttsBackend,
        voice: request.voice
      )
    }
  }

  private func startTTSBenchmark(_ request: BridgeRequest) {
    guard let requestID = request.id else { return }
    let text = request.text ?? "Hello from the iPhone TTS benchmark."
    ttsTask?.cancel()
    ttsTask = Task { [weak self] in
      guard let self else { return }
      await self.runTTSBenchmark(id: requestID, text: text)
    }
  }

  private func runGeneration(id: String, prompt: String, media: [GemmaMediaInput], maxTokens: Int) async {
    guard runtime.isReady else {
      await sendJSON(["type": "error", "id": id, "message": "Runtime is not ready"])
      return
    }

    totalRequests += 1
    generatedSnippet = ""
    let mediaSummary = media.isEmpty ? "" : " [media: \(media.map(\.mimeType).joined(separator: ", "))]"
    lastPromptSnippet = String((prompt + mediaSummary).prefix(220))
    await sendJSON(["type": "started", "id": id])

    do {
      let result = try await runtime.generate(prompt: prompt, media: media, maxTokens: maxTokens) { [weak self] token in
        guard let self else { return }
        self.generatedSnippet = Self.trailingSnippet(self.generatedSnippet + token)
        self.outputTokens += max(1, token.split { $0.isWhitespace || $0.isNewline }.count)
        await self.sendJSON(["type": "token", "id": id, "text": token])
      }

      inputTokens += result.inputTokensEstimate
      lastTokensPerSecond = result.tokensPerSecond
      await sendJSON([
        "type": "done",
        "id": id,
        "text": result.text,
        "input_tokens_estimate": result.inputTokensEstimate,
        "output_tokens_estimate": result.outputTokensEstimate,
        "tokens_per_second": result.tokensPerSecond,
        "elapsed_seconds": result.elapsedSeconds
      ])
      await sendReady()
    } catch {
      await sendJSON(["type": "error", "id": id, "message": error.localizedDescription])
      AppLog.error("Gemma bridge generation failed for request \(id): \(AppLog.describe(error))")
    }
  }

  private func runLocalTest(prompt: String, media: [GemmaMediaInput] = [], isProbe: Bool) async {
    guard runtime.isReady else {
      localTestStatus = "Runtime is not ready."
      return
    }
    guard !isRunningLocalTest else {
      localTestStatus = "A local test is already running."
      return
    }

    isRunningLocalTest = true
    localTestResponse = ""
    localTestStatus = isProbe ? "Running post-load probe..." : "Generating..."
    let mediaSummary = media.isEmpty ? "" : " [media: \(media.map(\.mimeType).joined(separator: ", "))]"
    lastPromptSnippet = String((prompt + mediaSummary).prefix(220))
    generatedSnippet = ""

    do {
      let result = try await runtime.generate(prompt: prompt, media: media, maxTokens: 128) { [weak self] token in
        guard let self else { return }
        self.localTestResponse = Self.trailingSnippet(self.localTestResponse + token)
        self.generatedSnippet = Self.trailingSnippet(self.generatedSnippet + token)
        self.outputTokens += max(1, token.split { $0.isWhitespace || $0.isNewline }.count)
      }

      inputTokens += result.inputTokensEstimate
      lastTokensPerSecond = result.tokensPerSecond
      localTestResponse = result.text
      localTestStatus = String(format: "Done: %.1f tok/s", result.tokensPerSecond)
      if isProbe {
        lastProbeResponse = result.text
        AppLog.info("Gemma post-load probe output: \(result.text)")
      } else {
        AppLog.info("Gemma local test output: \(result.text)")
      }
    } catch {
      localTestStatus = "Test failed: \(error.localizedDescription)"
      if isProbe {
        AppLog.error("Gemma post-load probe failed: \(AppLog.describe(error))")
      } else {
        AppLog.error("Gemma local test failed: \(AppLog.describe(error))")
      }
    }

    isRunningLocalTest = false
  }

  private func runTTS(id: String, text: String, backend: String?, voice: String?) async {
    let requestedBackend = PhoneTTSBackend.parse(backend)
    guard PhoneTTSBackend.selectableCases.contains(requestedBackend) else {
      let message = "\(requestedBackend.displayName) is not enabled because it has not been validated with Gemma loaded."
      lastTTSStatus = message
      await sendJSON(["type": "error", "id": id, "message": message])
      AppLog.error("Blocked unsupported bridge TTS backend: \(requestedBackend.rawValue)")
      return
    }
    do {
      await sendJSON(["type": "tts_started", "id": id])
      let result = try await ttsRuntime.synthesizeStreaming(
        text: text,
        backend: requestedBackend.rawValue,
        voice: voice ?? requestedBackend.defaultVoice
      ) { [weak self] chunk in
        guard let self else { return }
        await self.sendTTSBinary(id: id, audio: chunk)
      }
      ttsRequests += 1
      lastTTSBackend = result.backend.displayName
      lastTTSLatency = result.elapsedSeconds
      lastTTSStatus = String(
        format: "%@ %.2fs audio, first %.2fs, wall %.2fs",
        result.backend.displayName,
        result.audioSeconds,
        result.firstAudioSeconds,
        result.elapsedSeconds
      )
      var payload = result.payload
      payload["type"] = "tts_done"
      payload["id"] = id
      await sendJSON(payload)
      await sendReady()
    } catch {
      lastTTSStatus = "TTS failed: \(error.localizedDescription)"
      await sendJSON(["type": "error", "id": id, "message": error.localizedDescription])
      AppLog.error("TTS bridge request \(id) failed: \(AppLog.describe(error))")
    }
  }

  private func runTTSBenchmark(id: String, text: String) async {
    let rows = await ttsRuntime.benchmark(text: text).map(\.payload)
    lastTTSStatus = "TTS benchmark complete."
    lastTTSBackend = "benchmark"
    await sendJSON([
      "type": "tts_benchmark_done",
      "id": id,
      "text": text,
      "results": rows
    ])
    await sendReady()
  }

  private func runPose(id: String, request: BridgeRequest) async {
    await runPose(
      id: id,
      format: request.format?.lowercased(),
      width: request.width,
      height: request.height,
      dataText: request.data,
      poseBackendText: request.poseBackend,
      poseModelText: request.poseModel
    )
  }

  private func runPose(
    id: String,
    format maybeFormat: String?,
    width maybeWidth: Int?,
    height maybeHeight: Int?,
    dataText maybeDataText: String?,
    frameData maybeFrameData: Data? = nil,
    poseBackendText: String?,
    poseModelText: String?
  ) async {
    do {
      guard let format = maybeFormat?.lowercased() else {
        throw PoseRuntimeError.invalidFrame("pose request needs format")
      }
      guard let width = maybeWidth, let height = maybeHeight, width > 0, height > 0 else {
        throw PoseRuntimeError.invalidFrame("pose request needs positive width and height")
      }
      let data: Data
      if let maybeFrameData {
        data = maybeFrameData
      } else if let dataText = maybeDataText, let decoded = Data(base64Encoded: dataText) {
        data = decoded
      } else {
        throw PoseRuntimeError.invalidFrame("pose request needs binary frame data or valid base64 data")
      }
      let poseBackend = PoseBackend(rawValue: poseBackendText?.lowercased() ?? "gpu") ?? .gpu
      let poseModel = PoseModelQuality(rawValue: poseModelText?.lowercased() ?? "lite") ?? .lite
      let input = PoseBridgeInput(
        format: format,
        width: width,
        height: height,
        data: data,
        backend: poseBackend,
        modelQuality: poseModel
      )
      let output = try await poseRuntime.detect(input)
      poseRequests += 1
      lastPoseBackend = poseBackend.displayName
      lastPoseModel = poseModel.displayName
      lastPoseLatency = output.payload["total_seconds"] as? Double ?? 0
      lastPoseStatus = String(
        format: "%@ %@ %@ %.1f ms",
        poseBackend.displayName,
        poseModel.displayName,
        format,
        lastPoseLatency * 1000
      )
      var payload = output.payload
      payload["type"] = "pose_done"
      payload["id"] = id
      await sendJSON(payload)
      await sendReady()
    } catch {
      lastPoseStatus = "Pose failed: \(error.localizedDescription)"
      await sendJSON(["type": "error", "id": id, "message": error.localizedDescription])
      AppLog.error("Pose bridge request \(id) failed: \(AppLog.describe(error))")
    }
  }

  private func sendReady() async {
    runtimeName = runtime.name
    runtimeStatus = runtime.status
    runtimeReady = runtime.isReady

    await sendJSON([
      "type": "ready",
      "device": UIDevice.current.name,
      "model": "gemma-4-E2B-it",
      "runtime": runtime.name,
      "runtime_ready": runtime.isReady,
      "runtime_status": runtime.status,
      "pose_requests": poseRequests,
      "pose_status": lastPoseStatus,
      "pose_model": lastPoseModel,
      "tts_requests": ttsRequests,
      "tts_status": lastTTSStatus,
      "tts_backend": lastTTSBackend,
      "input_tokens": inputTokens,
      "output_tokens": outputTokens,
      "requests": totalRequests
    ])
  }

  private func sendJSON(_ object: [String: Any]) async {
    guard let socket else { return }

    do {
      let data = try JSONSerialization.data(withJSONObject: object)
      guard let text = String(data: data, encoding: .utf8) else { return }
      try await socket.send(.string(text))
    } catch {
      lastError = error.localizedDescription
      AppLog.error("Gemma bridge send failed: \(AppLog.describe(error))")
    }
  }

  private func sendTTSBinary(id: String, audio: Data) async {
    guard let socket else { return }

    do {
      let header: [String: Any] = [
        "type": "tts_audio",
        "id": id,
        "audio_format": "s16le",
        "sample_rate": PhoneTTSRuntime.sampleRate,
        "channels": 1,
        "bytes": audio.count
      ]
      let headerData = try JSONSerialization.data(withJSONObject: header)
      var frame = Data()
      frame.append(Self.ttsBinaryMagic)
      var headerLength = UInt32(headerData.count).bigEndian
      withUnsafeBytes(of: &headerLength) { frame.append(contentsOf: $0) }
      frame.append(headerData)
      frame.append(audio)
      try await socket.send(.data(frame))
    } catch {
      lastError = error.localizedDescription
      AppLog.error("Gemma bridge TTS binary send failed: \(AppLog.describe(error))")
    }
  }

  private static func trailingSnippet(_ text: String) -> String {
    let maxLength = 420
    guard text.count > maxLength else { return text }
    return String(text.suffix(maxLength))
  }

  private static func fileSize(_ url: URL) -> Int64? {
    guard
      let values = try? url.resourceValues(forKeys: [.fileSizeKey]),
      let size = values.fileSize
    else {
      return nil
    }
    return Int64(size)
  }

  private static func formatBytes(_ bytes: Int64) -> String {
    ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
  }

  private static func parsePoseBinaryFrame(_ data: Data) throws -> (header: Data, frame: Data) {
    let prefixCount = poseBinaryMagic.count + 4
    guard data.count >= prefixCount else {
      throw PoseRuntimeError.invalidFrame("binary pose frame is too short")
    }
    guard data.prefix(poseBinaryMagic.count) == poseBinaryMagic else {
      throw PoseRuntimeError.invalidFrame("binary pose frame has bad magic")
    }
    let lengthOffset = poseBinaryMagic.count
    let headerLength = data[lengthOffset..<(lengthOffset + 4)].reduce(UInt32(0)) { partial, byte in
      (partial << 8) | UInt32(byte)
    }
    let headerStart = prefixCount
    let headerEnd = headerStart + Int(headerLength)
    guard headerEnd <= data.count else {
      throw PoseRuntimeError.invalidFrame("binary pose frame header is truncated")
    }
    return (
      header: data.subdata(in: headerStart..<headerEnd),
      frame: data.subdata(in: headerEnd..<data.count)
    )
  }

  private static func parseGenerateBinaryFrame(_ data: Data) throws -> (header: GenerateBinaryHeader, media: [GemmaMediaInput]) {
    let prefixCount = generateBinaryMagic.count + 4
    guard data.count >= prefixCount else {
      throw PoseRuntimeError.invalidFrame("binary generate frame is too short")
    }
    guard data.prefix(generateBinaryMagic.count) == generateBinaryMagic else {
      throw PoseRuntimeError.invalidFrame("binary generate frame has bad magic")
    }
    let lengthOffset = generateBinaryMagic.count
    let headerLength = data[lengthOffset..<(lengthOffset + 4)].reduce(UInt32(0)) { partial, byte in
      (partial << 8) | UInt32(byte)
    }
    let headerStart = prefixCount
    let headerEnd = headerStart + Int(headerLength)
    guard headerEnd <= data.count else {
      throw PoseRuntimeError.invalidFrame("binary generate frame header is truncated")
    }
    let header = try JSONDecoder().decode(
      GenerateBinaryHeader.self,
      from: data.subdata(in: headerStart..<headerEnd)
    )
    guard header.type == "generate_media" else {
      throw PoseRuntimeError.invalidFrame("binary generate frame type must be generate_media")
    }

    let payloadStart = headerEnd
    let media = try header.media.map { item -> GemmaMediaInput in
      let start = payloadStart + item.offset
      let end = start + item.bytes
      guard start >= payloadStart, end <= data.count, start <= end else {
        throw PoseRuntimeError.invalidFrame("binary generate media range is out of bounds")
      }
      return GemmaMediaInput(
        data: data.subdata(in: start..<end),
        mimeType: item.mimeType,
        displayName: item.displayName
      )
    }
    return (header, media)
  }

  private static func launchBridgeURL() -> String? {
    let arguments = ProcessInfo.processInfo.arguments
    let environment = ProcessInfo.processInfo.environment
    if let index = arguments.firstIndex(of: "--bridge-url"),
       arguments.indices.contains(index + 1)
    {
      return arguments[index + 1]
    }
    return environment["GEMMAPI_BRIDGE_URL"]
  }

  private static func wavData(pcm: Data, sampleRate: Int, channels: Int) -> Data {
    let bitsPerSample = 16
    let byteRate = sampleRate * channels * bitsPerSample / 8
    let blockAlign = channels * bitsPerSample / 8
    var data = Data()
    data.append(contentsOf: "RIFF".utf8)
    appendLittleEndian(UInt32(36 + pcm.count), to: &data)
    data.append(contentsOf: "WAVE".utf8)
    data.append(contentsOf: "fmt ".utf8)
    appendLittleEndian(UInt32(16), to: &data)
    appendLittleEndian(UInt16(1), to: &data)
    appendLittleEndian(UInt16(channels), to: &data)
    appendLittleEndian(UInt32(sampleRate), to: &data)
    appendLittleEndian(UInt32(byteRate), to: &data)
    appendLittleEndian(UInt16(blockAlign), to: &data)
    appendLittleEndian(UInt16(bitsPerSample), to: &data)
    data.append(contentsOf: "data".utf8)
    appendLittleEndian(UInt32(pcm.count), to: &data)
    data.append(pcm)
    return data
  }

  private static func appendLittleEndian<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
    var littleEndian = value.littleEndian
    withUnsafeBytes(of: &littleEndian) { bytes in
      data.append(contentsOf: bytes)
    }
  }

  private static func audioFileDiagnostics(url: URL, byteCount: Int) -> LocalAudioDiagnostics {
    do {
      let file = try AVAudioFile(forReading: url)
      let sampleRate = file.processingFormat.sampleRate
      let duration = sampleRate > 0 ? Double(file.length) / sampleRate : 0
      let channels = Int(file.processingFormat.channelCount)
      let format = String(describing: file.processingFormat.commonFormat)
      let firstBytes = (try? Data(contentsOf: url, options: [.mappedIfSafe]).prefix(12)) ?? Data()
      let magic = firstBytes.map { String(format: "%02x", $0) }.joined()
      return LocalAudioDiagnostics(
        durationSeconds: duration,
        sampleRate: sampleRate,
        channels: channels,
        format: format,
        byteCount: byteCount,
        magic: magic
      )
    } catch {
      AppLog.error("Local Gemma audio prompt diagnostic read failed: \(AppLog.describe(error))")
      return LocalAudioDiagnostics(
        durationSeconds: 0,
        sampleRate: 0,
        channels: 0,
        format: "unknown",
        byteCount: byteCount,
        magic: ""
      )
    }
  }
}

private actor AudioChunkCollector {
  private var chunks = Data()

  func append(_ chunk: Data) {
    chunks.append(chunk)
  }

  func data() -> Data {
    chunks
  }
}

private enum AudioInputError: LocalizedError {
  case permissionDenied(String)
  case recordingFailed(String)

  var errorDescription: String? {
    switch self {
    case .permissionDenied(let detail):
      return detail
    case .recordingFailed(let detail):
      return detail
    }
  }
}

private struct LocalAudioDiagnostics {
  let durationSeconds: Double
  let sampleRate: Double
  let channels: Int
  let format: String
  let byteCount: Int
  let magic: String

  var logSummary: String {
    String(
      format: "duration=%.3fs, sample_rate=%.0f, channels=%d, format=%@, bytes=%d, magic=%@",
      durationSeconds,
      sampleRate,
      channels,
      format,
      byteCount,
      magic
    )
  }
}

private struct BridgeRequest: Decodable {
  let type: String
  let id: String?
  let prompt: String?
  let maxTokens: Int?
  let format: String?
  let width: Int?
  let height: Int?
  let data: String?
  let poseBackend: String?
  let poseModel: String?
  let text: String?
  let ttsBackend: String?
  let voice: String?
  let chunkIndex: Int?
  let chunkCount: Int?

  enum CodingKeys: String, CodingKey {
    case type
    case id
    case prompt
    case maxTokens = "max_tokens"
    case format
    case width
    case height
    case data
    case poseBackend = "pose_backend"
    case poseModel = "pose_model"
    case text
    case ttsBackend = "tts_backend"
    case voice
    case chunkIndex = "chunk_index"
    case chunkCount = "chunk_count"
  }
}

private struct GenerateBinaryHeader: Decodable {
  let type: String
  let id: String
  let prompt: String
  let maxTokens: Int?
  let media: [GenerateBinaryMediaItem]

  enum CodingKeys: String, CodingKey {
    case type
    case id
    case prompt
    case maxTokens = "max_tokens"
    case media
  }
}

private struct GenerateBinaryMediaItem: Decodable {
  let mimeType: String
  let displayName: String?
  let offset: Int
  let bytes: Int

  enum CodingKeys: String, CodingKey {
    case mimeType = "mime_type"
    case displayName = "display_name"
    case offset
    case bytes
  }
}

private struct PendingPoseFrame {
  let format: String
  let width: Int
  let height: Int
  let poseBackend: String?
  let poseModel: String?
  var chunks: [String?]
}
