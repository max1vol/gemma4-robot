import Foundation
import AVFoundation
import UIKit

@MainActor
final class PiBridgeClient: NSObject, ObservableObject {
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
  @Published private(set) var isStartingAudioCapture = false
  @Published private(set) var isRecordingAudio = false
  @Published private(set) var audioInputStatus = "Hold mic to send a raw Gemma audio prompt."
  @Published private(set) var isSpeakingLocalTest = false
  @Published var selectedTTSBackend = PhoneTTSBackend.piperRyanHigh.rawValue
  @Published var selectedTTSVoice = PhoneTTSBackend.piperRyanHigh.defaultVoice
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
  @Published private(set) var ttsDownloadProgress: PhoneTTSDownloadProgress?

  private let runtime: GemmaRuntime
  private let poseRuntime = PoseRuntime()
  private let ttsRuntime = PhoneTTSRuntime()
  private var audioCaptureState = AudioCaptureState.idle
  private var audioCaptureFinishPending = false
  private var localAudioRecorder: AVAudioRecorder?
  private var retiredAudioRecorders: [AVAudioRecorder] = []
  private var localAudioURL: URL?
  private var localAudioPlayer: AVAudioPlayer?
  private var localAudioPlaybackURL: URL?
  private var localAudioPlaybackID: UUID?
  private var localAudioPlayerObjectID: ObjectIdentifier?
  private var localAudioEngine: AVAudioEngine?
  private var localAudioPlayerNode: AVAudioPlayerNode?
  private var localAudioPlaybackBuffer: AVAudioPCMBuffer?
  private var localPlaybackFinishTask: Task<Void, Never>?
  private var audioSessionDeactivationTask: Task<Void, Never>?
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
  private static let recorderCallbackDrainNanoseconds: UInt64 = 10_000_000_000

  init(runtime: GemmaRuntime = RuntimeFactory.make()) {
    self.runtime = runtime
    self.runtimeName = runtime.name
    self.runtimeStatus = runtime.status
    self.runtimeReady = runtime.isReady
    super.init()
    installAudioSessionObservers()
    if let launchBridgeURL = Self.launchBridgeURL() {
      self.bridgeURLString = launchBridgeURL
    }
  }

  deinit {
    NotificationCenter.default.removeObserver(self)
  }

  var isConnected: Bool {
    socket != nil
  }

  var isAudioCaptureActive: Bool {
    isStartingAudioCapture || isRecordingAudio
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
    if isSpeakingLocalTest {
      AppLog.info("Stopping local TTS playback before Gemma runtime load")
      stopLocalPlaybackObjects()
      isSpeakingLocalTest = false
      localPlaybackFinishTask?.cancel()
      localPlaybackFinishTask = nil
      cleanupLocalPlaybackSessionNow(reason: "Gemma runtime load")
    }
    ttsDownloadProgress = nil
    await ttsRuntime.releaseCachedModels(reason: "Gemma runtime load")
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
    guard !isAudioCaptureActive else {
      localTestStatus = "Wait for audio recording to finish."
      AppLog.info("Gemma local text test blocked because local audio capture is active")
      return
    }
    guard !isSpeakingLocalTest else {
      localTestStatus = "Wait for speech playback to finish."
      AppLog.info("Gemma local text test blocked because local TTS playback is active")
      return
    }
    let prompt = localTestPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !prompt.isEmpty else {
      localTestStatus = "Enter a prompt first."
      return
    }

    await runLocalTest(prompt: prompt, isProbe: false)
  }

  func beginHoldToTalkCapture() {
    guard audioCaptureState.isIdle else {
      AppLog.info("Local Gemma audio capture start ignored because capture state is \(audioCaptureState.logName)")
      return
    }
    guard !isRunningLocalTest else {
      audioInputStatus = "Wait for the current model test to finish."
      AppLog.info("Local Gemma audio capture blocked because local generation is active")
      return
    }
    guard !isSpeakingLocalTest else {
      audioInputStatus = "Wait for speech playback to finish."
      AppLog.info("Local Gemma audio capture blocked because local TTS playback is active")
      return
    }

    let captureID = UUID()
    audioCaptureFinishPending = false
    setAudioCaptureState(.starting(captureID))
    audioInputStatus = "Starting microphone..."

    Task { [weak self] in
      await self?.startPreparedAudioCapture(captureID: captureID)
    }
  }

  func endHoldToTalkCapture() {
    Task { [weak self] in
      await self?.finishAudioCaptureAndSend()
    }
  }

  func startAudioCapture() async {
    guard audioCaptureState.isIdle else {
      AppLog.info("Local Gemma audio capture start ignored because capture state is \(audioCaptureState.logName)")
      return
    }
    guard !isRunningLocalTest else {
      audioInputStatus = "Wait for the current model test to finish."
      AppLog.info("Local Gemma audio capture blocked because local generation is active")
      return
    }
    guard !isSpeakingLocalTest else {
      audioInputStatus = "Wait for speech playback to finish."
      AppLog.info("Local Gemma audio capture blocked because local TTS playback is active")
      return
    }

    let captureID = UUID()
    audioCaptureFinishPending = false
    setAudioCaptureState(.starting(captureID))
    audioInputStatus = "Starting microphone..."
    await startPreparedAudioCapture(captureID: captureID)
  }

  private func startPreparedAudioCapture(captureID: UUID) async {
    do {
      AppLog.info("Local Gemma audio capture starting: id=\(captureID.uuidString), requesting microphone permission")
      let microphoneAllowed = await Self.requestMicrophonePermission()
      AppLog.info("Local microphone permission result: id=\(captureID.uuidString), allowed=\(microphoneAllowed)")
      guard microphoneAllowed else {
        throw AudioInputError.permissionDenied("Microphone permission was denied.")
      }
      guard audioCaptureState.matchesStarting(captureID) else {
        throw AudioInputError.cancelled("Audio capture was cancelled before recording started.")
      }

      retireStaleAudioRecorderIfNeeded()
      stopLocalPlaybackObjects()
      audioInputStatus = "Recording raw audio..."

      let session = AVAudioSession.sharedInstance()
      Self.logAudioSession("record configure before id=\(captureID.uuidString)")
      try session.setCategory(.playAndRecord, mode: .measurement, options: [.defaultToSpeaker, .allowBluetoothHFP])
      try session.setActive(true, options: .notifyOthersOnDeactivation)
      Self.logAudioSession("record configure active id=\(captureID.uuidString)")
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
      setAudioCaptureState(.recording(captureID))
      AppLog.info("Local Gemma audio capture recording started: id=\(captureID.uuidString), path=\(url.path), sample_rate=16000, channels=1, format=pcm_s16le")

      if audioCaptureFinishPending {
        audioCaptureFinishPending = false
        AppLog.info("Local Gemma audio capture release was pending during startup; finishing now: id=\(captureID.uuidString)")
        await finishAudioCaptureAndSend()
      }
    } catch {
      cancelAudioCapture(reason: "start failed", status: "Audio failed: \(error.localizedDescription)")
      audioInputStatus = "Audio failed: \(error.localizedDescription)"
      AppLog.error("Local Gemma audio capture failed: \(AppLog.describe(error))")
    }
  }

  func finishAudioCaptureAndSend() async {
    guard !isSpeakingLocalTest else {
      AppLog.info("Local Gemma audio capture release ignored because local TTS playback is active")
      return
    }

    switch audioCaptureState {
    case .starting(let captureID):
      audioCaptureFinishPending = true
      audioInputStatus = "Finishing microphone startup..."
      AppLog.info("Local Gemma audio capture release received during startup: id=\(captureID.uuidString)")
      return
    case .recording(let captureID):
      AppLog.info("Local Gemma audio capture release received: id=\(captureID.uuidString)")
    case .idle:
      AppLog.info("Local Gemma audio capture release ignored because capture state is idle")
      return
    }

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
    guard !isRunningLocalTest, !isAudioCaptureActive, !isSpeakingLocalTest else {
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

  func runAudioThenTTSSmokeTest(seconds: Double = 0.75) async {
    guard !isRunningLocalTest, !isAudioCaptureActive, !isSpeakingLocalTest else {
      audioInputStatus = "Audio to TTS smoke skipped: app is busy."
      AppLog.error("Audio to TTS smoke skipped because app is busy")
      return
    }

    AppLog.info(String(format: "Audio to TTS smoke starting: recording %.2fs, then local playback", seconds))
    await startAudioCapture()
    guard isRecordingAudio else {
      AppLog.error("Audio to TTS smoke could not start recording: state=\(audioCaptureState.logName)")
      return
    }

    let nanoseconds = UInt64(max(0.2, seconds) * 1_000_000_000)
    try? await Task.sleep(nanoseconds: nanoseconds)
    let recorderDuration = localAudioRecorder?.currentTime ?? 0
    stopAudioCapture()
    AppLog.info(String(format: "Audio to TTS smoke recording phase stopped: duration=%.3fs", recorderDuration))

    try? await Task.sleep(nanoseconds: 900_000_000)
    await previewSelectedTTSVoice()
    AppLog.info("Audio to TTS smoke finished")
  }

  func runAudioGemmaSpeakSmokeTest(seconds: Double = 1.25) async {
    guard runtimeReady else {
      audioInputStatus = "Audio Gemma Speak smoke skipped: runtime is not ready."
      AppLog.error("Audio Gemma Speak smoke skipped because runtime is not ready")
      return
    }
    guard !isRunningLocalTest, !isAudioCaptureActive, !isSpeakingLocalTest else {
      audioInputStatus = "Audio Gemma Speak smoke skipped: app is busy."
      AppLog.error("Audio Gemma Speak smoke skipped because app is busy")
      return
    }

    AppLog.info(String(format: "Audio Gemma Speak smoke starting: recording %.2fs, then Gemma audio prompt, then local Speak", seconds))
    await startAudioCapture()
    guard isRecordingAudio else {
      AppLog.error("Audio Gemma Speak smoke could not start recording: state=\(audioCaptureState.logName)")
      return
    }

    let nanoseconds = UInt64(max(0.2, seconds) * 1_000_000_000)
    try? await Task.sleep(nanoseconds: nanoseconds)
    await finishAudioCaptureAndSend()

    let response = localTestResponse.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !response.isEmpty else {
      AppLog.error("Audio Gemma Speak smoke skipped local Speak because Gemma response is empty. status=\(localTestStatus)")
      return
    }
    AppLog.info("Audio Gemma Speak smoke invoking local Speak: response_chars=\(response.count), preview=\(Self.trailingSnippet(response, maxLength: 160))")
    await speakLocalTestResponse()
    AppLog.info("Audio Gemma Speak smoke finished")
  }

  func speakLocalTestResponse() async {
    guard !isRunningLocalTest else {
      lastTTSStatus = "Wait for generation to finish before speaking."
      AppLog.info("Local TTS speak blocked because local generation is active")
      return
    }
    let text = localTestResponse.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty else {
      lastTTSStatus = "No local response to speak."
      return
    }
    await playLocalTTS(text: text, statusPrefix: "Spoke")
  }

  func previewSelectedTTSVoice() async {
    guard !isRunningLocalTest else {
      lastTTSStatus = "Wait for generation to finish before previewing voice."
      AppLog.info("Local TTS preview blocked because local generation is active")
      return
    }
    await playLocalTTS(text: "hello, how are you?", statusPrefix: "Voice preview")
  }

  private func playLocalTTS(text: String, statusPrefix: String) async {
    let backend = PhoneTTSBackend.parse(selectedTTSBackend)
    guard PhoneTTSBackend.selectableCases.contains(backend) else {
      lastTTSStatus = "\(backend.displayName) is not enabled because it has not been validated with Gemma loaded."
      AppLog.error("Blocked unsupported local TTS backend from UI playback: \(backend.rawValue)")
      return
    }
    guard !isAudioCaptureActive else {
      lastTTSStatus = "Wait for audio recording to finish."
      AppLog.info("Local TTS playback blocked because local audio capture is active")
      return
    }
    guard !isRunningLocalTest else {
      lastTTSStatus = "Wait for generation to finish."
      AppLog.info("Local TTS playback blocked because local generation is active")
      return
    }
    guard !isSpeakingLocalTest else {
      AppLog.info("Local TTS playback ignored because another local playback is active")
      return
    }
    isSpeakingLocalTest = true

    do {
      await waitForRetiredAudioRecordersToDrain(context: statusPrefix)
      guard !isAudioCaptureActive else {
        throw LocalPlaybackError.playbackEngineUnavailable("Audio capture became active before playback could start.")
      }
      stopLocalPlaybackObjects()
      if let playbackURL = localAudioPlaybackURL {
        try? FileManager.default.removeItem(at: playbackURL)
        localAudioPlaybackURL = nil
      }
      localPlaybackFinishTask?.cancel()
      localPlaybackFinishTask = nil
      lastTTSStatus = "\(statusPrefix): synthesizing \(selectedTTSVoice)..."
      AppLog.info("Local TTS playback request: prefix=\(statusPrefix), voice=\(selectedTTSVoice), chars=\(text.count)")
      let collector = AudioChunkCollector()
      let result = try await ttsRuntime.synthesizeStreaming(
        text: text,
        backend: selectedTTSBackend,
        voice: selectedTTSVoice,
        onDownloadProgress: { [weak self] progress in
          self?.updateTTSDownloadProgress(progress)
        }
      ) { chunk, _ in
        await collector.append(chunk)
      }
      let pcm = await collector.data()
      guard !pcm.isEmpty else { throw LocalPlaybackError.emptyAudio }
      let pcmStats = Self.pcmS16LEStats(pcm)
      let wav = Self.wavData(pcm: pcm, sampleRate: result.sampleRate, channels: 1)
      let playback = try await startLocalTTSPlayback(wav: wav, context: statusPrefix)
      let expectedPlaybackSeconds = max(0.5, playback.duration)
      localPlaybackFinishTask = Task { [weak self] in
        let graceNanoseconds = UInt64((expectedPlaybackSeconds + 2.0) * 1_000_000_000)
        try? await Task.sleep(nanoseconds: graceNanoseconds)
        await MainActor.run {
          guard let self else { return }
          self.finishLocalAudioPlayback(
            playbackID: playback.id,
            successfully: false,
            errorDescription: "Playback completion callback timed out.",
            errorLog: "Local TTS playback completion timeout after \(String(format: "%.2f", expectedPlaybackSeconds + 2.0))s"
          )
        }
      }
      ttsRequests += 1
      lastTTSBackend = result.backend.displayName
      lastTTSLatency = result.elapsedSeconds
      lastTTSStatus = String(
        format: "%@: %@ %.2fs audio, first %.2fs, wall %.2fs, route %@",
        statusPrefix,
        result.backend.displayName,
        result.audioSeconds,
        result.firstAudioSeconds,
        result.elapsedSeconds,
        playback.route
      )
      AppLog.info(
        String(
          format: "Local TTS playback started: method=%@, bytes=%d, pcm_bytes=%d, sample_rate=%d, duration=%.2fs, route=%@",
          playback.method,
          wav.count,
          pcm.count,
          result.sampleRate,
          playback.duration,
          playback.route
        )
      )
      AppLog.info(
        String(
          format: "Local TTS PCM stats: samples=%d, peak=%.4f, rms=%.4f, zero_ratio=%.3f",
          pcmStats.samples,
          pcmStats.peak,
          pcmStats.rms,
          pcmStats.zeroRatio
        )
      )
    } catch {
      isSpeakingLocalTest = false
      ttsDownloadProgress = nil
      localPlaybackFinishTask?.cancel()
      localPlaybackFinishTask = nil
      if let playbackURL = localAudioPlaybackURL {
        try? FileManager.default.removeItem(at: playbackURL)
        localAudioPlaybackURL = nil
      }
      cleanupLocalPlaybackSessionNow(reason: "local TTS failure after \(statusPrefix)")
      lastTTSStatus = "TTS failed: \(error.localizedDescription)"
      AppLog.error("Local TTS playback failed: \(AppLog.describe(error))")
    }
  }

  private func updateTTSDownloadProgress(_ progress: PhoneTTSDownloadProgress?) {
    guard ttsDownloadProgress != progress else { return }
    ttsDownloadProgress = progress
    guard let progress else { return }
    lastTTSStatus = progress.detail
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
    audioCaptureFinishPending = false
    setAudioCaptureState(.idle)
    if let recorder {
      if recorder.isRecording {
        recorder.stop()
      }
      retireAudioRecorder(recorder, reason: "capture stopped")
    }
    scheduleAudioSessionDeactivation()
  }

  private func cancelAudioCapture(reason: String, status: String) {
    let recorder = localAudioRecorder
    localAudioRecorder = nil
    localAudioURL = nil
    audioCaptureFinishPending = false
    let previousState = audioCaptureState
    setAudioCaptureState(.idle)
    audioInputStatus = status

    if let recorder {
      if recorder.isRecording {
        recorder.stop()
      }
      retireAudioRecorder(recorder, reason: reason)
    }

    AppLog.info("Local Gemma audio capture cancelled: reason=\(reason), previous_state=\(previousState.logName)")
    scheduleAudioSessionDeactivation()
  }

  private func setAudioCaptureState(_ state: AudioCaptureState) {
    audioCaptureState = state
    isStartingAudioCapture = state.isStarting
    isRecordingAudio = state.isRecording
    AppLog.info("Local Gemma audio capture state: \(state.logName)")
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
    if !retiredAudioRecorders.contains(where: { $0 === recorder }) {
      retiredAudioRecorders.append(recorder)
    }
    AppLog.info(
      "Local Gemma audio recorder retired while AudioQueue callbacks drain: reason=\(reason), retired_count=\(retiredAudioRecorders.count)"
    )

    Task { [weak self, recorder] in
      try? await Task.sleep(nanoseconds: Self.recorderCallbackDrainNanoseconds)
      await MainActor.run {
        guard let self else { return }
        self.retiredAudioRecorders.removeAll { $0 === recorder }
        AppLog.info("Local Gemma audio recorder released after callback drain: retired_count=\(self.retiredAudioRecorders.count)")
        self.scheduleAudioSessionDeactivation()
      }
    }
  }

  private func scheduleAudioSessionDeactivation() {
    audioSessionDeactivationTask?.cancel()
    audioSessionDeactivationTask = Task { [weak self] in
      try? await Task.sleep(nanoseconds: 1_500_000_000)
      await MainActor.run {
        guard let self else { return }
        if self.isAudioCaptureActive || self.isSpeakingLocalTest {
          AppLog.info("Local Gemma audio capture session deactivation skipped: capture=\(self.audioCaptureState.logName), speaking=\(self.isSpeakingLocalTest)")
          return
        }
        if !self.retiredAudioRecorders.isEmpty {
          AppLog.info("Local Gemma audio capture session deactivation delayed: retired_recorders=\(self.retiredAudioRecorders.count)")
          return
        }
        do {
          Self.logAudioSession("record deactivate before")
          try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
          Self.logAudioSession("record deactivate after")
          AppLog.info("Local Gemma audio capture session deactivated")
        } catch {
          AppLog.error("Local Gemma audio capture session deactivate failed: \(AppLog.describe(error))")
        }
      }
    }
  }

  private nonisolated static func requestMicrophonePermission() async -> Bool {
    if #available(iOS 17.0, *) {
      return await AVAudioApplication.requestRecordPermission()
    } else {
      return await withCheckedContinuation { continuation in
        AVAudioSession.sharedInstance().requestRecordPermission { allowed in
          continuation.resume(returning: allowed)
        }
      }
    }
  }

  private func configureLocalPlaybackSession(context: String) throws -> String {
    let session = AVAudioSession.sharedInstance()
    Self.logAudioSession("local playback configure before context=\(context)")
    try Self.clearSpeakerOverrideIfNeeded()
    if session.category == .playAndRecord {
      if retiredAudioRecorders.isEmpty {
        try session.setActive(false, options: .notifyOthersOnDeactivation)
        Self.logAudioSession("local playback configure deactivated previous record session context=\(context)")
      } else {
        AppLog.info(
          "Local TTS playback keeping audio session active while recorder callbacks drain: context=\(context), retired_recorders=\(retiredAudioRecorders.count)"
        )
        try session.setActive(true, options: .notifyOthersOnDeactivation)
        Self.logAudioSession("local playback configure kept playAndRecord active context=\(context)")
        return Self.audioRouteDescription(session)
      }
    }
    try session.setCategory(.playback, mode: .default, options: [])
    try session.setActive(true, options: .notifyOthersOnDeactivation)
    Self.logAudioSession("local playback configure active context=\(context)")
    return Self.audioRouteDescription(session)
  }

  private func startLocalTTSPlayback(wav: Data, context: String) async throws -> LocalPlaybackStart {
    let playbackURL = FileManager.default.temporaryDirectory
      .appendingPathComponent("gemma-local-tts-\(UUID().uuidString)")
      .appendingPathExtension("wav")
    try wav.write(to: playbackURL, options: [.atomic])
    localAudioPlaybackURL = playbackURL
    AppLog.info("Local TTS playback file written: path=\(playbackURL.path), bytes=\(wav.count)")

    do {
      return try await startLocalTTSPlaybackWithAudioPlayer(url: playbackURL, audioBytes: wav.count, context: context)
    } catch {
      AppLog.error("Local TTS AVAudioPlayer start failed, trying AVAudioEngine fallback: \(AppLog.describe(error))")
      stopLocalPlaybackObjects()
      return try await startLocalTTSPlaybackWithAudioEngine(url: playbackURL, audioBytes: wav.count, context: context)
    }
  }

  private func startLocalTTSPlaybackWithAudioPlayer(url playbackURL: URL, audioBytes: Int, context: String) async throws -> LocalPlaybackStart {
    let route = try configureLocalPlaybackSession(context: "\(context) AVAudioPlayer")
    try await Task.sleep(nanoseconds: 250_000_000)
    let player = try AVAudioPlayer(contentsOf: playbackURL)
    player.delegate = self
    player.prepareToPlay()
    let playbackID = UUID()
    localAudioPlaybackID = playbackID
    localAudioPlayerObjectID = ObjectIdentifier(player)
    localAudioPlayer = player

    AppLog.info(
      String(
        format: "Local TTS AVAudioPlayer start: duration=%.2fs, file_bytes=%d, route=%@",
        player.duration,
        audioBytes,
        route
      )
    )

    guard player.play() else {
      localAudioPlayer = nil
      localAudioPlaybackID = nil
      localAudioPlayerObjectID = nil
      throw LocalPlaybackError.playbackEngineUnavailable("AVAudioPlayer.play returned false.")
    }

    AppLog.info("Local TTS AVAudioPlayer started: playing=\(player.isPlaying), route=\(route)")
    return LocalPlaybackStart(
      id: playbackID,
      duration: player.duration,
      route: route,
      method: "AVAudioPlayer"
    )
  }

  private func startLocalTTSPlaybackWithAudioEngine(url playbackURL: URL, audioBytes: Int, context: String) async throws -> LocalPlaybackStart {
    let route = try configureLocalPlaybackSession(context: "\(context) AVAudioEngine")
    try await Task.sleep(nanoseconds: 250_000_000)
    let file = try AVAudioFile(forReading: playbackURL)
    let frameCount = AVAudioFrameCount(min(file.length, Int64(UInt32.max)))
    guard let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: frameCount) else {
      throw LocalPlaybackError.playbackEngineUnavailable("Could not allocate AVAudioEngine playback buffer.")
    }
    try file.read(into: buffer)
    guard buffer.frameLength > 0 else {
      throw LocalPlaybackError.emptyAudio
    }

    let engine = AVAudioEngine()
    let playerNode = AVAudioPlayerNode()
    engine.attach(playerNode)
    engine.connect(playerNode, to: engine.mainMixerNode, format: buffer.format)
    engine.prepare()
    try engine.start()

    let playbackID = UUID()
    localAudioPlaybackID = playbackID
    localAudioPlayerObjectID = nil
    localAudioEngine = engine
    localAudioPlayerNode = playerNode
    localAudioPlaybackBuffer = buffer

    let duration = Double(buffer.frameLength) / buffer.format.sampleRate
    AppLog.info(
      String(
        format: "Local TTS AVAudioEngine start: duration=%.2fs, file_bytes=%d, sample_rate=%.0f, route=%@",
        duration,
        audioBytes,
        buffer.format.sampleRate,
        route
      )
    )

    playerNode.scheduleBuffer(buffer, completionCallbackType: .dataPlayedBack) { [weak self] _ in
      Task { @MainActor [weak self] in
        guard let self, self.localAudioPlaybackID == playbackID else { return }
        self.finishLocalAudioPlayback(
          playbackID: playbackID,
          successfully: true,
          errorDescription: nil,
          errorLog: nil
        )
      }
    }
    playerNode.play()
    guard playerNode.isPlaying else {
      stopLocalPlaybackObjects()
      throw LocalPlaybackError.playbackEngineUnavailable("AVAudioEngine player node did not start.")
    }

    AppLog.info("Local TTS AVAudioEngine started: playing=\(playerNode.isPlaying), route=\(route)")
    return LocalPlaybackStart(
      id: playbackID,
      duration: duration,
      route: route,
      method: "AVAudioEngine"
    )
  }

  private func waitForRetiredAudioRecordersToDrain(context: String) async {
    guard !retiredAudioRecorders.isEmpty else { return }
    AppLog.info(
      "Local TTS playback waiting for recorder callback drain: context=\(context), retired_recorders=\(retiredAudioRecorders.count)"
    )
    while !retiredAudioRecorders.isEmpty {
      try? await Task.sleep(nanoseconds: 250_000_000)
    }
    AppLog.info(
      "Local TTS playback recorder callback drain wait finished: context=\(context), retired_recorders=\(retiredAudioRecorders.count)"
    )
  }

  private func stopLocalPlaybackObjects() {
    localAudioPlayer?.delegate = nil
    localAudioPlayer?.stop()
    localAudioPlayer = nil
    localAudioPlayerNode?.stop()
    localAudioEngine?.stop()
    localAudioPlayerNode = nil
    localAudioEngine = nil
    localAudioPlaybackBuffer = nil
    localAudioPlaybackID = nil
    localAudioPlayerObjectID = nil
  }

  private func cleanupLocalPlaybackSessionNow(reason: String) {
    do {
      Self.logAudioSession("local playback cleanup before reason=\(reason)")
      try Self.clearSpeakerOverrideIfNeeded()
      if !isAudioCaptureActive && !isSpeakingLocalTest && retiredAudioRecorders.isEmpty {
        try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
      } else if !retiredAudioRecorders.isEmpty {
        AppLog.info(
          "Local TTS playback cleanup kept audio session active while recorder callbacks drain: reason=\(reason), retired_recorders=\(retiredAudioRecorders.count)"
        )
        scheduleAudioSessionDeactivation()
      }
      Self.logAudioSession("local playback cleanup after reason=\(reason)")
    } catch {
      AppLog.error("Local TTS playback cleanup failed: reason=\(reason), error=\(AppLog.describe(error))")
    }
  }

  private nonisolated static func clearSpeakerOverrideIfNeeded() throws {
    let session = AVAudioSession.sharedInstance()
    guard session.category == .playAndRecord else { return }
    try session.overrideOutputAudioPort(.none)
  }

  private nonisolated static func audioRouteDescription(_ session: AVAudioSession = .sharedInstance()) -> String {
    audioRouteDescription(session.currentRoute)
  }

  private nonisolated static func audioRouteDescription(_ route: AVAudioSessionRouteDescription) -> String {
    let inputs = route.inputs
      .map { "\($0.portName)(\($0.portType.rawValue))" }
      .joined(separator: ",")
    let outputs = route.outputs
      .map { "\($0.portName)(\($0.portType.rawValue))" }
      .joined(separator: ",")
    return "in=[\(inputs.isEmpty ? "none" : inputs)] out=[\(outputs.isEmpty ? "none" : outputs)]"
  }

  private nonisolated static func logAudioSession(_ event: String) {
    let session = AVAudioSession.sharedInstance()
    let options = session.categoryOptions.rawValue
    AppLog.info(
      String(
        format: "AudioSession %@: category=%@, mode=%@, options=0x%llx, outputVolume=%.2f, otherAudio=%@, silencedHint=%@, route=%@",
        event,
        session.category.rawValue,
        session.mode.rawValue,
        UInt64(options),
        session.outputVolume,
        session.isOtherAudioPlaying ? "true" : "false",
        session.secondaryAudioShouldBeSilencedHint ? "true" : "false",
        audioRouteDescription(session)
      )
    )
  }

  private nonisolated static func routeChangeReasonDescription(_ rawValue: UInt?) -> String {
    guard let rawValue,
          let reason = AVAudioSession.RouteChangeReason(rawValue: rawValue)
    else {
      return rawValue.map { "unknown(\($0))" } ?? "nil"
    }

    switch reason {
    case .unknown:
      return "unknown"
    case .newDeviceAvailable:
      return "newDeviceAvailable"
    case .oldDeviceUnavailable:
      return "oldDeviceUnavailable"
    case .categoryChange:
      return "categoryChange"
    case .override:
      return "override"
    case .wakeFromSleep:
      return "wakeFromSleep"
    case .noSuitableRouteForCategory:
      return "noSuitableRouteForCategory"
    case .routeConfigurationChange:
      return "routeConfigurationChange"
    @unknown default:
      return "unknown(\(rawValue))"
    }
  }

  private func finishLocalAudioPlayback(playbackID: UUID, successfully flag: Bool, errorDescription: String?, errorLog: String?) {
    guard localAudioPlaybackID == playbackID else { return }
    let route = Self.audioRouteDescription()
    localPlaybackFinishTask?.cancel()
    localPlaybackFinishTask = nil
    stopLocalPlaybackObjects()
    isSpeakingLocalTest = false
    if let playbackURL = localAudioPlaybackURL {
      do {
        try FileManager.default.removeItem(at: playbackURL)
        AppLog.info("Local TTS playback temp file removed: \(playbackURL.path)")
      } catch {
        AppLog.error("Local TTS playback temp file remove failed: \(AppLog.describe(error)), path=\(playbackURL.path)")
      }
      localAudioPlaybackURL = nil
    }

    if let errorDescription {
      lastTTSStatus = "TTS playback failed: \(errorDescription)"
      AppLog.error("Local TTS playback decode error: route=\(route), \(errorLog ?? errorDescription)")
    } else {
      lastTTSStatus += flag ? " Finished." : " Stopped before finishing."
      AppLog.info("Local TTS playback finished: success=\(flag), route=\(route)")
    }

    schedulePlaybackSessionDeactivation()
  }

  private func schedulePlaybackSessionDeactivation() {
    Task { [weak self] in
      try? await Task.sleep(nanoseconds: 250_000_000)
      await MainActor.run {
        guard let self else { return }
        if self.isAudioCaptureActive || self.isSpeakingLocalTest {
          AppLog.info("Local TTS playback session deactivation skipped: capture=\(self.audioCaptureState.logName), speaking=\(self.isSpeakingLocalTest)")
          return
        }
        if !self.retiredAudioRecorders.isEmpty {
          AppLog.info("Local TTS playback session deactivation delayed: retired_recorders=\(self.retiredAudioRecorders.count)")
          self.scheduleAudioSessionDeactivation()
          return
        }
        do {
          Self.logAudioSession("local playback deactivate before")
          try Self.clearSpeakerOverrideIfNeeded()
          try AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
          Self.logAudioSession("local playback deactivate after")
          AppLog.info("Local TTS playback session deactivated")
        } catch {
          AppLog.error("Local TTS playback session deactivate failed: \(AppLog.describe(error))")
        }
      }
    }
  }

  private func installAudioSessionObservers() {
    let center = NotificationCenter.default
    let session = AVAudioSession.sharedInstance()
    center.addObserver(
      self,
      selector: #selector(audioSessionInterruptionNotification(_:)),
      name: AVAudioSession.interruptionNotification,
      object: session
    )
    center.addObserver(
      self,
      selector: #selector(audioSessionRouteChangeNotification(_:)),
      name: AVAudioSession.routeChangeNotification,
      object: session
    )
    center.addObserver(
      self,
      selector: #selector(audioSessionMediaServicesLostNotification(_:)),
      name: AVAudioSession.mediaServicesWereLostNotification,
      object: session
    )
    center.addObserver(
      self,
      selector: #selector(audioSessionMediaServicesResetNotification(_:)),
      name: AVAudioSession.mediaServicesWereResetNotification,
      object: session
    )
  }

  @objc private nonisolated func audioSessionInterruptionNotification(_ notification: Notification) {
    let typeRaw = notification.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt
    let optionRaw = notification.userInfo?[AVAudioSessionInterruptionOptionKey] as? UInt
    AppLog.info("AudioSession interruption notification: type=\(typeRaw.map(String.init) ?? "nil"), options=\(optionRaw.map(String.init) ?? "nil"), route=\(Self.audioRouteDescription())")
    Task { @MainActor [weak self] in
      self?.handleAudioSessionInterruption(typeRaw: typeRaw, optionRaw: optionRaw)
    }
  }

  @objc private nonisolated func audioSessionRouteChangeNotification(_ notification: Notification) {
    let reasonRaw = notification.userInfo?[AVAudioSessionRouteChangeReasonKey] as? UInt
    let previous = notification.userInfo?[AVAudioSessionRouteChangePreviousRouteKey] as? AVAudioSessionRouteDescription
    let previousDescription = previous.map(Self.audioRouteDescription) ?? "none"
    let reason = Self.routeChangeReasonDescription(reasonRaw)
    AppLog.info("AudioSession route change notification: reason=\(reason), previous=\(previousDescription), current=\(Self.audioRouteDescription())")
    Task { @MainActor [weak self] in
      self?.handleAudioSessionRouteChange(reason: reason)
    }
  }

  @objc private nonisolated func audioSessionMediaServicesLostNotification(_ notification: Notification) {
    AppLog.error("AudioSession media services lost: route=\(Self.audioRouteDescription())")
    Task { @MainActor [weak self] in
      self?.handleAudioSessionReset(reason: "media services lost")
    }
  }

  @objc private nonisolated func audioSessionMediaServicesResetNotification(_ notification: Notification) {
    AppLog.error("AudioSession media services reset: route=\(Self.audioRouteDescription())")
    Task { @MainActor [weak self] in
      self?.handleAudioSessionReset(reason: "media services reset")
    }
  }

  private func handleAudioSessionInterruption(typeRaw: UInt?, optionRaw: UInt?) {
    guard let typeRaw,
          let type = AVAudioSession.InterruptionType(rawValue: typeRaw)
    else { return }

    switch type {
    case .began:
      if isAudioCaptureActive {
        cancelAudioCapture(reason: "audio session interruption", status: "Recording interrupted by iOS.")
      }
      if let playbackID = localAudioPlaybackID {
        finishLocalAudioPlayback(
          playbackID: playbackID,
          successfully: false,
          errorDescription: "Playback was interrupted by iOS.",
          errorLog: "Audio session interruption began; options=\(optionRaw.map(String.init) ?? "nil")"
        )
      }
    case .ended:
      AppLog.info("AudioSession interruption ended: options=\(optionRaw.map(String.init) ?? "nil"), route=\(Self.audioRouteDescription())")
    @unknown default:
      AppLog.info("AudioSession interruption unknown type=\(typeRaw), route=\(Self.audioRouteDescription())")
    }
  }

  private func handleAudioSessionRouteChange(reason: String) {
    if isSpeakingLocalTest {
      AppLog.info("Local TTS playback route after change: reason=\(reason), route=\(Self.audioRouteDescription())")
    }
    if isAudioCaptureActive {
      AppLog.info("Local audio capture route after change: reason=\(reason), state=\(audioCaptureState.logName), route=\(Self.audioRouteDescription())")
    }
  }

  private func handleAudioSessionReset(reason: String) {
    if isAudioCaptureActive {
      cancelAudioCapture(reason: reason, status: "Recording stopped: \(reason).")
    }
    if let playbackID = localAudioPlaybackID {
      finishLocalAudioPlayback(
        playbackID: playbackID,
        successfully: false,
        errorDescription: "Audio services reset during playback.",
        errorLog: reason
      )
    }
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
      let resolvedVoice = voice ?? requestedBackend.defaultVoice
      AppLog.info("TTS bridge request \(id) starting: backend=\(requestedBackend.rawValue), voice=\(resolvedVoice), chars=\(text.count)")
      await sendJSON(["type": "tts_started", "id": id])
      let stats = TTSStreamStats(requestID: id)
      let result = try await ttsRuntime.synthesizeStreaming(
        text: text,
        backend: requestedBackend.rawValue,
        voice: resolvedVoice
      ) { [weak self] chunk, sampleRate in
        guard let self else { return }
        let event = await stats.record(chunk)
        if event.isFirstChunk {
          AppLog.info(
            String(
              format: "TTS bridge request %@ first chunk: bytes=%d, first_audio=%.2fs",
              id,
              chunk.count,
              event.elapsedSeconds
            )
          )
        }
        await self.sendTTSBinary(id: id, audio: chunk, sampleRate: sampleRate)
      }
      let snapshot = await stats.snapshot()
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
      AppLog.info(
        String(
          format: "TTS bridge request %@ complete: chunks=%d, streamed_bytes=%d, result_bytes=%d, audio=%.2fs, wall=%.2fs",
          id,
          snapshot.chunks,
          snapshot.bytes,
          result.audioBytes,
          result.audioSeconds,
          result.elapsedSeconds
        )
      )
      var payload = result.payload
      payload["type"] = "tts_done"
      payload["id"] = id
      await sendJSON(payload)
      await sendReady()
    } catch is CancellationError {
      lastTTSStatus = "TTS cancelled."
      AppLog.info("TTS bridge request \(id) cancelled")
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

  private func sendTTSBinary(id: String, audio: Data, sampleRate: Int) async {
    guard let socket else { return }

    do {
      let header: [String: Any] = [
        "type": "tts_audio",
        "id": id,
        "audio_format": "s16le",
        "sample_rate": sampleRate,
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

  private static func trailingSnippet(_ text: String, maxLength: Int = 420) -> String {
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

  private static func pcmS16LEStats(_ pcm: Data) -> LocalPCMStats {
    var samples = 0
    var zeroes = 0
    var peak = 0.0
    var squareSum = 0.0
    let alignedCount = pcm.count - (pcm.count % 2)

    pcm.withUnsafeBytes { rawBuffer in
      guard let base = rawBuffer.bindMemory(to: UInt8.self).baseAddress else { return }
      var offset = 0
      while offset < alignedCount {
        let lo = UInt16(base[offset])
        let hi = UInt16(base[offset + 1]) << 8
        let value = Int16(bitPattern: hi | lo)
        let normalized = Double(value) / 32768.0
        let magnitude = abs(normalized)
        samples += 1
        if value == 0 {
          zeroes += 1
        }
        peak = max(peak, magnitude)
        squareSum += normalized * normalized
        offset += 2
      }
    }

    let rms = samples > 0 ? sqrt(squareSum / Double(samples)) : 0.0
    let zeroRatio = samples > 0 ? Double(zeroes) / Double(samples) : 0.0
    return LocalPCMStats(samples: samples, peak: peak, rms: rms, zeroRatio: zeroRatio)
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

extension PiBridgeClient: AVAudioPlayerDelegate {
  nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
    let playerObjectID = ObjectIdentifier(player)
    Task { @MainActor [weak self, playerObjectID] in
      guard let self, let playbackID = self.localAudioPlaybackID else {
        return
      }
      guard playerObjectID == self.localAudioPlayerObjectID else {
        AppLog.info("Ignoring stale AVAudioPlayer finish callback")
        return
      }
      self.finishLocalAudioPlayback(
        playbackID: playbackID,
        successfully: flag,
        errorDescription: nil,
        errorLog: nil
      )
    }
  }

  nonisolated func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: Error?) {
    let playerObjectID = ObjectIdentifier(player)
    Task { @MainActor [weak self, playerObjectID] in
      guard let self, let playbackID = self.localAudioPlaybackID else {
        return
      }
      guard playerObjectID == self.localAudioPlayerObjectID else {
        AppLog.info("Ignoring stale AVAudioPlayer decode error callback")
        return
      }
      let detail = error.map(AppLog.describe) ?? "unknown AVAudioPlayer decode error"
      self.finishLocalAudioPlayback(
        playbackID: playbackID,
        successfully: false,
        errorDescription: "AVAudioPlayer decode failed.",
        errorLog: detail
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

private actor TTSStreamStats {
  private let requestID: String
  private let started = Date()
  private var chunkCount = 0
  private var byteCount = 0

  init(requestID: String) {
    self.requestID = requestID
  }

  func record(_ chunk: Data) -> TTSStreamEvent {
    chunkCount += 1
    byteCount += chunk.count
    return TTSStreamEvent(
      requestID: requestID,
      chunks: chunkCount,
      bytes: byteCount,
      isFirstChunk: chunkCount == 1,
      elapsedSeconds: Date().timeIntervalSince(started)
    )
  }

  func snapshot() -> (chunks: Int, bytes: Int) {
    (chunkCount, byteCount)
  }
}

private struct TTSStreamEvent: Sendable {
  let requestID: String
  let chunks: Int
  let bytes: Int
  let isFirstChunk: Bool
  let elapsedSeconds: Double
}

private struct LocalPlaybackStart {
  let id: UUID
  let duration: Double
  let route: String
  let method: String
}

private enum LocalPlaybackError: LocalizedError {
  case emptyAudio
  case playbackEngineUnavailable(String)

  var errorDescription: String? {
    switch self {
    case .emptyAudio:
      return "TTS generated no audio."
    case .playbackEngineUnavailable(let detail):
      return "iPhone audio playback is unavailable: \(detail)"
    }
  }
}

private enum AudioInputError: LocalizedError {
  case permissionDenied(String)
  case recordingFailed(String)
  case cancelled(String)

  var errorDescription: String? {
    switch self {
    case .permissionDenied(let detail):
      return detail
    case .recordingFailed(let detail):
      return detail
    case .cancelled(let detail):
      return detail
    }
  }
}

private enum AudioCaptureState: Equatable {
  case idle
  case starting(UUID)
  case recording(UUID)

  var isIdle: Bool {
    if case .idle = self { return true }
    return false
  }

  var isStarting: Bool {
    if case .starting = self { return true }
    return false
  }

  var isRecording: Bool {
    if case .recording = self { return true }
    return false
  }

  var logName: String {
    switch self {
    case .idle:
      return "idle"
    case .starting(let id):
      return "starting(\(id.uuidString))"
    case .recording(let id):
      return "recording(\(id.uuidString))"
    }
  }

  func matchesStarting(_ id: UUID) -> Bool {
    if case .starting(let current) = self {
      return current == id
    }
    return false
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

private struct LocalPCMStats {
  let samples: Int
  let peak: Double
  let rms: Double
  let zeroRatio: Double
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
