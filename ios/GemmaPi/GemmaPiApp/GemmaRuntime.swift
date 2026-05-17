import Foundation
import Darwin
import OSLog

enum AppLog {
  private static let logger = Logger(subsystem: "com.gemma4robot.gemmapi", category: "GemmaInferenceServer")

  static func info(_ message: String) {
    logger.info("\(message, privacy: .public)")
    write("INFO", message)
  }

  static func error(_ message: String) {
    logger.error("\(message, privacy: .public)")
    write("ERROR", message)
  }

  static func describe(_ error: Error) -> String {
    let nsError = error as NSError
    var parts = [error.localizedDescription]
    parts.append("type=\(String(reflecting: type(of: error)))")
    parts.append("domain=\(nsError.domain)")
    parts.append("code=\(nsError.code)")
    if !nsError.userInfo.isEmpty {
      parts.append("userInfo=\(nsError.userInfo)")
    }
    return parts.joined(separator: ", ")
  }

  private static func write(_ level: String, _ message: String) {
    let line = "[GemmaInferenceServer] \(level): \(message)"
    print(line)
    fflush(stdout)
    fputs("\(line)\n", stderr)
    fflush(stderr)
    NSLog("%@", line)
  }
}

enum InferenceBackend: String, CaseIterable, Identifiable {
  case gpu
  case cpu

  var id: String { rawValue }

  var displayName: String {
    switch self {
    case .gpu:
      return "GPU (Metal)"
    case .cpu:
      return "CPU"
    }
  }
}

struct GenerationResult {
  var text: String
  var inputTokensEstimate: Int
  var outputTokensEstimate: Int
  var elapsedSeconds: Double

  var tokensPerSecond: Double {
    guard elapsedSeconds > 0 else { return 0 }
    return Double(outputTokensEstimate) / elapsedSeconds
  }
}

struct GemmaMediaInput: Sendable {
  var data: Data
  var mimeType: String
  var displayName: String?

  var isAudio: Bool {
    mimeType.lowercased().hasPrefix("audio/")
  }

  var isImage: Bool {
    mimeType.lowercased().hasPrefix("image/")
  }
}

@MainActor
protocol GemmaRuntime: AnyObject {
  var name: String { get }
  var status: String { get }
  var isReady: Bool { get }

  func loadModel(at modelURL: URL, backend: InferenceBackend) async throws
  func loadModel(at modelURL: URL, projectorURL: URL?, backend: InferenceBackend) async throws
  func generate(
    prompt: String,
    maxTokens: Int,
    onToken: @escaping @MainActor (String) async -> Void
  ) async throws -> GenerationResult
  func generate(
    prompt: String,
    media: [GemmaMediaInput],
    maxTokens: Int,
    onToken: @escaping @MainActor (String) async -> Void
  ) async throws -> GenerationResult
  func cancel()
}

extension GemmaRuntime {
  func loadModel(at modelURL: URL, projectorURL: URL?, backend: InferenceBackend) async throws {
    _ = projectorURL
    try await loadModel(at: modelURL, backend: backend)
  }

  func generate(
    prompt: String,
    media: [GemmaMediaInput],
    maxTokens: Int,
    onToken: @escaping @MainActor (String) async -> Void
  ) async throws -> GenerationResult {
    guard media.isEmpty else {
      throw RuntimeError.multimodalUnavailable("This runtime is loaded without a multimodal projector.")
    }
    return try await generate(prompt: prompt, maxTokens: maxTokens, onToken: onToken)
  }
}

enum RuntimeFactory {
  static func make() -> GemmaRuntime {
    #if canImport(llama)
      return LlamaGemmaRuntime()
    #elseif canImport(LiteRTLM)
      return LiteRTGemmaRuntime()
    #elseif canImport(LiteRTLMSwift)
      return DirectCLiteRTGemmaRuntime()
    #else
      return UnavailableGemmaRuntime()
    #endif
  }
}

final class UnavailableGemmaRuntime: GemmaRuntime {
  private(set) var isReady = false
  private(set) var status = "No real Gemma runtime is linked in this app build."
  let name = "unavailable"

  func loadModel(at modelURL: URL, backend: InferenceBackend) async throws {
    guard FileManager.default.fileExists(atPath: modelURL.path) else {
      throw RuntimeError.modelMissing(modelURL.path)
    }
    isReady = false
    status = "Load failed: no llama.cpp or LiteRT-LM runtime is linked, so Gemma cannot run."
    throw RuntimeError.runtimeUnavailable
  }

  func generate(
    prompt: String,
    maxTokens: Int,
    onToken: @escaping @MainActor (String) async -> Void
  ) async throws -> GenerationResult {
    _ = prompt
    _ = maxTokens
    _ = onToken
    throw RuntimeError.runtimeUnavailable
  }

  func cancel() {}
}

enum RuntimeError: LocalizedError {
  case modelMissing(String)
  case runtimeNotReady
  case runtimeUnavailable
  case multimodalUnavailable(String)
  case nativeLoadFailed(String)

  var errorDescription: String? {
    switch self {
    case .modelMissing(let path):
      return "Model file is missing at \(path)"
    case .runtimeNotReady:
      return "Runtime is not ready"
    case .runtimeUnavailable:
      return "Real Gemma runtime is unavailable because neither llama.cpp nor LiteRT-LM is linked in this app build"
    case .multimodalUnavailable(let detail):
      return detail
    case .nativeLoadFailed(let detail):
      return detail
    }
  }
}

#if canImport(LiteRTLM)
  import LiteRTLM

  final class LiteRTGemmaRuntime: GemmaRuntime {
    private var engine: Engine?
    private var conversation: Conversation?
    private(set) var isReady = false
    private(set) var status = "LiteRT-LM runtime available."
    let name = "litert-lm"

    func loadModel(at modelURL: URL, backend: InferenceBackend) async throws {
      AppLog.info("LiteRT-LM official load begin: path=\(modelURL.path), backend=\(backend.rawValue)")
      let cacheDirectory = try FileManager.default.url(
        for: .cachesDirectory,
        in: .userDomainMask,
        appropriateFor: nil,
        create: true
      )

      let nativeBackend: Backend = backend == .gpu ? .gpu : .cpu()
      let config = try EngineConfig(
        modelPath: modelURL.path,
        backend: nativeBackend,
        cacheDir: cacheDirectory.path
      )
      let newEngine = Engine(engineConfig: config)
      try await newEngine.initialize()
      engine = newEngine
      conversation = try await newEngine.createConversation()
      isReady = true
      status = "Gemma 4 E2B loaded with LiteRT-LM \(backend.rawValue)."
      AppLog.info("LiteRT-LM official load complete: ready=\(isReady), status=\(status)")
    }

    func generate(
      prompt: String,
      maxTokens: Int,
      onToken: @escaping @MainActor (String) async -> Void
    ) async throws -> GenerationResult {
      guard let conversation else { throw RuntimeError.runtimeNotReady }

      let started = Date()
      var text = ""
      var outputTokens = 0

      for try await chunk in conversation.sendMessageStream(Message(prompt)) {
        if outputTokens >= maxTokens { break }
        guard let firstContent = chunk.contents.first else { continue }
        switch firstContent {
        case .text(let token):
          text += token
          outputTokens += max(1, token.split { $0.isWhitespace || $0.isNewline }.count)
          await onToken(token)
        default:
          break
        }
      }

      return GenerationResult(
        text: text,
        inputTokensEstimate: max(1, prompt.split { $0.isWhitespace || $0.isNewline }.count),
        outputTokensEstimate: outputTokens,
        elapsedSeconds: Date().timeIntervalSince(started)
      )
    }

    func cancel() {
      conversation = nil
      engine = nil
      isReady = false
      status = "Runtime reset."
    }
  }
#elseif canImport(LiteRTLMSwift)
  import LiteRTLMSwift

  private typealias CLiteRTLMStreamCallback = @convention(c) (
    UnsafeMutableRawPointer?,
    UnsafePointer<CChar>?,
    Bool,
    UnsafePointer<CChar>?
  ) -> Void
  private typealias CLiteRTLMSetBoolEngineSetting = @convention(c) (OpaquePointer?, Bool) -> Void
  private typealias CLiteRTLMSetCStringEngineSetting = @convention(c) (
    OpaquePointer?,
    UnsafePointer<CChar>
  ) -> Void

  private struct CLiteRTLMSamplerParams {
    var type: Int32
    var top_k: Int32
    var top_p: Float
    var temperature: Float
    var seed: Int32
  }

  private struct CLiteRTLMInputData {
    var type: Int32
    var data: UnsafeRawPointer?
    var size: Int
  }

  @_silgen_name("litert_lm_set_min_log_level")
  private func cLitertLMSetMinLogLevel(_ level: Int32)

  @_silgen_name("litert_lm_engine_settings_create")
  private func cLitertLMEngineSettingsCreate(
    _ modelPath: UnsafePointer<CChar>,
    _ backend: UnsafePointer<CChar>,
    _ visionBackend: UnsafePointer<CChar>?,
    _ audioBackend: UnsafePointer<CChar>?
  ) -> OpaquePointer?

  @_silgen_name("litert_lm_engine_settings_delete")
  private func cLitertLMEngineSettingsDelete(_ settings: OpaquePointer?)

  @_silgen_name("litert_lm_engine_settings_set_max_num_tokens")
  private func cLitertLMEngineSettingsSetMaxNumTokens(_ settings: OpaquePointer?, _ maxNumTokens: Int32)

  @_silgen_name("litert_lm_engine_settings_set_cache_dir")
  private func cLitertLMEngineSettingsSetCacheDir(_ settings: OpaquePointer?, _ cacheDir: UnsafePointer<CChar>)

  @_silgen_name("litert_lm_engine_settings_set_activation_data_type")
  private func cLitertLMEngineSettingsSetActivationDataType(_ settings: OpaquePointer?, _ activationDataType: Int32)

  @_silgen_name("litert_lm_engine_settings_enable_benchmark")
  private func cLitertLMEngineSettingsEnableBenchmark(_ settings: OpaquePointer?)

  @_silgen_name("litert_lm_engine_create")
  private func cLitertLMEngineCreate(_ settings: OpaquePointer?) -> OpaquePointer?

  @_silgen_name("litert_lm_engine_delete")
  private func cLitertLMEngineDelete(_ engine: OpaquePointer?)

  @_silgen_name("litert_lm_session_config_create")
  private func cLitertLMSessionConfigCreate() -> OpaquePointer?

  @_silgen_name("litert_lm_session_config_set_max_output_tokens")
  private func cLitertLMSessionConfigSetMaxOutputTokens(_ config: OpaquePointer?, _ maxOutputTokens: Int32)

  @_silgen_name("litert_lm_session_config_set_sampler_params")
  private func cLitertLMSessionConfigSetSamplerParams(
    _ config: OpaquePointer?,
    _ samplerParams: UnsafePointer<CLiteRTLMSamplerParams>
  )

  @_silgen_name("litert_lm_session_config_delete")
  private func cLitertLMSessionConfigDelete(_ config: OpaquePointer?)

  @_silgen_name("litert_lm_engine_create_session")
  private func cLitertLMEngineCreateSession(_ engine: OpaquePointer?, _ config: OpaquePointer?) -> OpaquePointer?

  @_silgen_name("litert_lm_session_delete")
  private func cLitertLMSessionDelete(_ session: OpaquePointer?)

  @_silgen_name("litert_lm_session_generate_content_stream")
  private func cLitertLMSessionGenerateContentStream(
    _ session: OpaquePointer?,
    _ inputs: UnsafePointer<CLiteRTLMInputData>,
    _ numInputs: Int,
    _ callback: CLiteRTLMStreamCallback?,
    _ callbackData: UnsafeMutableRawPointer?
  ) -> Int32

  @_silgen_name("litert_lm_session_get_benchmark_info")
  private func cLitertLMSessionGetBenchmarkInfo(_ session: OpaquePointer?) -> OpaquePointer?

  @_silgen_name("litert_lm_benchmark_info_delete")
  private func cLitertLMBenchmarkInfoDelete(_ benchmarkInfo: OpaquePointer?)

  @_silgen_name("litert_lm_benchmark_info_get_time_to_first_token")
  private func cLitertLMBenchmarkInfoGetTimeToFirstToken(_ benchmarkInfo: OpaquePointer?) -> Double

  @_silgen_name("litert_lm_benchmark_info_get_total_init_time_in_second")
  private func cLitertLMBenchmarkInfoGetTotalInitTime(_ benchmarkInfo: OpaquePointer?) -> Double

  @_silgen_name("litert_lm_benchmark_info_get_num_prefill_turns")
  private func cLitertLMBenchmarkInfoGetNumPrefillTurns(_ benchmarkInfo: OpaquePointer?) -> Int32

  @_silgen_name("litert_lm_benchmark_info_get_num_decode_turns")
  private func cLitertLMBenchmarkInfoGetNumDecodeTurns(_ benchmarkInfo: OpaquePointer?) -> Int32

  @_silgen_name("litert_lm_benchmark_info_get_prefill_token_count_at")
  private func cLitertLMBenchmarkInfoGetPrefillTokenCountAt(
    _ benchmarkInfo: OpaquePointer?,
    _ index: Int32
  ) -> Int32

  @_silgen_name("litert_lm_benchmark_info_get_decode_token_count_at")
  private func cLitertLMBenchmarkInfoGetDecodeTokenCountAt(
    _ benchmarkInfo: OpaquePointer?,
    _ index: Int32
  ) -> Int32

  @_silgen_name("litert_lm_benchmark_info_get_prefill_tokens_per_sec_at")
  private func cLitertLMBenchmarkInfoGetPrefillTokensPerSecAt(
    _ benchmarkInfo: OpaquePointer?,
    _ index: Int32
  ) -> Double

  @_silgen_name("litert_lm_benchmark_info_get_decode_tokens_per_sec_at")
  private func cLitertLMBenchmarkInfoGetDecodeTokensPerSecAt(
    _ benchmarkInfo: OpaquePointer?,
    _ index: Int32
  ) -> Double

  private final class DirectCLiteRTLMStreamState: @unchecked Sendable {
    let continuation: AsyncThrowingStream<String, Error>.Continuation
    let doneSemaphore: DispatchSemaphore

    init(
      continuation: AsyncThrowingStream<String, Error>.Continuation,
      doneSemaphore: DispatchSemaphore
    ) {
      self.continuation = continuation
      self.doneSemaphore = doneSemaphore
    }
  }

  private let directCLiteRTLMStreamCallback: CLiteRTLMStreamCallback = {
    callbackData,
    chunk,
    isFinal,
    errorMessage in
    guard let callbackData else { return }
    let state = Unmanaged<DirectCLiteRTLMStreamState>.fromOpaque(callbackData).takeUnretainedValue()

    let errorText: String? = {
      guard let errorMessage else { return nil }
      let message = String(cString: errorMessage)
      return message.isEmpty ? nil : message
    }()

    if let chunk, errorText == nil {
      let text = String(cString: chunk)
      if !text.isEmpty {
        state.continuation.yield(text)
      }
    }

    if isFinal || errorText != nil {
      if let errorText {
        state.continuation.finish(throwing: RuntimeError.nativeLoadFailed("LiteRT-LM stream failed: \(errorText)"))
      } else {
        state.continuation.finish()
      }
      let semaphore = state.doneSemaphore
      Unmanaged<DirectCLiteRTLMStreamState>.fromOpaque(callbackData).release()
      semaphore.signal()
    }
  }

  private final class DirectCLiteRTLMWorker: @unchecked Sendable {
    private let inferenceQueue = DispatchQueue(label: "com.gemma4robot.gemmapi.litertlm", qos: .userInitiated)
    private var engine: OpaquePointer?

    deinit {
      unload()
    }

    func load(modelPath: String, cacheDirectory: String, backend: String) async throws {
      try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
        inferenceQueue.async {
          do {
            self.deleteEngine()
            cLitertLMSetMinLogLevel(0)
            if backend == InferenceBackend.gpu.rawValue {
              Self.preloadGPUAcceleratorDylibs()
            }
            try FileManager.default.createDirectory(
              atPath: cacheDirectory,
              withIntermediateDirectories: true
            )

            let settings = modelPath.withCString { modelPathPtr in
              backend.withCString { backendPtr in
                cLitertLMEngineSettingsCreate(
                  modelPathPtr,
                  backendPtr,
                  nil,
                  nil
                )
              }
            }

            guard let settings else {
              throw RuntimeError.nativeLoadFailed(
                "Direct CLiteRTLM failed to create engine settings: backend=\(backend), path=\(modelPath)"
              )
            }
            defer { cLitertLMEngineSettingsDelete(settings) }

            cLitertLMEngineSettingsSetMaxNumTokens(settings, 4096)
            cacheDirectory.withCString { cacheDirPtr in
              cLitertLMEngineSettingsSetCacheDir(settings, cacheDirPtr)
            }
            if backend == InferenceBackend.gpu.rawValue {
              cLitertLMEngineSettingsSetActivationDataType(settings, 1)
              AppLog.info("LiteRT-LM GPU activation data type set to F16")
            }
            Self.configureOptionalEngineSettings(settings, backend: backend)
            cLitertLMEngineSettingsEnableBenchmark(settings)

            guard let createdEngine = cLitertLMEngineCreate(settings) else {
              throw RuntimeError.nativeLoadFailed(
                "Direct CLiteRTLM engine create returned NULL: backend=\(backend), visionBackend=NULL, audioBackend=NULL, path=\(modelPath)"
              )
            }

            self.engine = createdEngine
            continuation.resume()
          } catch {
            self.deleteEngine()
            continuation.resume(throwing: error)
          }
        }
      }
    }

    func generateStreaming(
      prompt: String,
      temperature: Float,
      maxTokens: Int32
    ) -> AsyncThrowingStream<String, Error> {
      AsyncThrowingStream { continuation in
        inferenceQueue.async {
          guard let engine = self.engine else {
            continuation.finish(throwing: RuntimeError.runtimeNotReady)
            return
          }

          guard let sessionConfig = cLitertLMSessionConfigCreate() else {
            continuation.finish(
              throwing: RuntimeError.nativeLoadFailed("Direct CLiteRTLM failed to create session config")
            )
            return
          }
          cLitertLMSessionConfigSetMaxOutputTokens(sessionConfig, maxTokens)
          var samplerParams = CLiteRTLMSamplerParams(
            type: 2,
            top_k: 40,
            top_p: 0.95,
            temperature: temperature,
            seed: 0
          )
          cLitertLMSessionConfigSetSamplerParams(sessionConfig, &samplerParams)

          guard let session = cLitertLMEngineCreateSession(engine, sessionConfig) else {
            cLitertLMSessionConfigDelete(sessionConfig)
            continuation.finish(
              throwing: RuntimeError.nativeLoadFailed("Direct CLiteRTLM failed to create session")
            )
            return
          }

          let streamDone = DispatchSemaphore(value: 0)
          let state = DirectCLiteRTLMStreamState(
            continuation: continuation,
            doneSemaphore: streamDone
          )
          let statePointer = Unmanaged.passRetained(state).toOpaque()
          let promptBytes = Array(prompt.utf8) + [0]

          let result = promptBytes.withUnsafeBufferPointer { buffer -> Int32 in
            var input = CLiteRTLMInputData(
              type: 0,
              data: UnsafeRawPointer(buffer.baseAddress),
              size: max(0, promptBytes.count - 1)
            )
            let startResult = cLitertLMSessionGenerateContentStream(
              session,
              &input,
              1,
              directCLiteRTLMStreamCallback,
              statePointer
            )
            if startResult == 0 {
              streamDone.wait()
            }
            return startResult
          }

          if result != 0 {
            Unmanaged<DirectCLiteRTLMStreamState>.fromOpaque(statePointer).release()
            continuation.finish(
              throwing: RuntimeError.nativeLoadFailed("Direct CLiteRTLM failed to start stream: code=\(result)")
            )
          } else {
            self.logSessionBenchmark(session)
          }

          cLitertLMSessionDelete(session)
          cLitertLMSessionConfigDelete(sessionConfig)
        }
      }
    }

    func unload() {
      inferenceQueue.sync {
        deleteEngine()
      }
    }

    private func deleteEngine() {
      if let engine {
        cLitertLMEngineDelete(engine)
        self.engine = nil
      }
    }

    private static func runtimeSymbol<T>(_ name: String, as type: T.Type) -> T? {
      guard let handle = dlopen(nil, RTLD_NOW) else {
        let message = dlerror().map { String(cString: $0) } ?? "unknown dlopen error"
        AppLog.error("Failed to open process symbol table: \(message)")
        return nil
      }
      guard let symbol = dlsym(handle, name) else {
        return nil
      }
      return unsafeBitCast(symbol, to: type)
    }

    private static func configureOptionalEngineSettings(_ settings: OpaquePointer, backend: String) {
      if let setSpeculativeDecoding = runtimeSymbol(
        "litert_lm_engine_settings_set_enable_speculative_decoding",
        as: CLiteRTLMSetBoolEngineSetting.self
      ) {
        setSpeculativeDecoding(settings, true)
        AppLog.info("LiteRT-LM speculative decoding enabled through native C API")
      } else {
        AppLog.info("LiteRT-LM speculative decoding C API is not exported by the linked native framework")
      }

      guard backend == InferenceBackend.gpu.rawValue else { return }
      guard let frameworkDirectory = Bundle.main.privateFrameworksURL?.path else { return }
      guard let setDispatchLibDirectory = runtimeSymbol(
        "litert_lm_engine_settings_set_litert_dispatch_lib_dir",
        as: CLiteRTLMSetCStringEngineSetting.self
      ) else {
        AppLog.info("LiteRT-LM dispatch-library-directory C API is not exported by the linked native framework")
        return
      }

      frameworkDirectory.withCString { path in
        setDispatchLibDirectory(settings, path)
      }
      AppLog.info("LiteRT-LM dispatch library directory set to \(frameworkDirectory)")
    }

    private static func preloadGPUAcceleratorDylibs() {
      guard let directory = Bundle.main.privateFrameworksURL else { return }
      let url = directory.appendingPathComponent("libLiteRtMetalAccelerator.dylib")
      guard FileManager.default.fileExists(atPath: url.path) else {
        AppLog.error("LiteRT Metal accelerator dylib is missing at \(url.path)")
        return
      }
      if dlopen(url.path, RTLD_NOW | RTLD_GLOBAL) != nil {
        AppLog.info("Preloaded LiteRT GPU dylib: \(url.path)")
      } else {
        let message = dlerror().map { String(cString: $0) } ?? "unknown dlopen error"
        AppLog.error("Failed to preload LiteRT GPU dylib: \(url.path), error=\(message)")
      }
    }

    private func logSessionBenchmark(_ session: OpaquePointer?) {
      guard let info = cLitertLMSessionGetBenchmarkInfo(session) else { return }
      defer { cLitertLMBenchmarkInfoDelete(info) }

      let initTime = cLitertLMBenchmarkInfoGetTotalInitTime(info)
      let ttft = cLitertLMBenchmarkInfoGetTimeToFirstToken(info)
      let prefillTurns = max(0, Int(cLitertLMBenchmarkInfoGetNumPrefillTurns(info)))
      let decodeTurns = max(0, Int(cLitertLMBenchmarkInfoGetNumDecodeTurns(info)))

      AppLog.info(
        "Direct CLiteRTLM benchmark: init=\(String(format: "%.2f", initTime))s, TTFT=\(String(format: "%.2f", ttft))s"
      )

      for index in 0..<prefillTurns {
        let cIndex = Int32(index)
        let count = cLitertLMBenchmarkInfoGetPrefillTokenCountAt(info, cIndex)
        let speed = cLitertLMBenchmarkInfoGetPrefillTokensPerSecAt(info, cIndex)
        AppLog.info("Direct CLiteRTLM prefill[\(index)]: \(count) tokens @ \(String(format: "%.1f", speed)) tok/s")
      }

      for index in 0..<decodeTurns {
        let cIndex = Int32(index)
        let count = cLitertLMBenchmarkInfoGetDecodeTokenCountAt(info, cIndex)
        let speed = cLitertLMBenchmarkInfoGetDecodeTokensPerSecAt(info, cIndex)
        AppLog.info("Direct CLiteRTLM decode[\(index)]: \(count) tokens @ \(String(format: "%.1f", speed)) tok/s")
      }
    }
  }

  final class DirectCLiteRTGemmaRuntime: GemmaRuntime {
    private let worker = DirectCLiteRTLMWorker()
    private(set) var isReady = false
    private(set) var status = "Direct CLiteRTLM runtime available."
    let name = "CLiteRTLM"

    func loadModel(at modelURL: URL, backend: InferenceBackend) async throws {
      isReady = false
      let size = Self.fileSize(modelURL).map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) } ?? "unknown"
      AppLog.info("Direct CLiteRTLM load begin: path=\(modelURL.path), size=\(size), backend=\(backend.rawValue)")

      do {
        let cacheDirectory = try FileManager.default.url(
          for: .cachesDirectory,
          in: .userDomainMask,
          appropriateFor: nil,
          create: true
        )
        .appendingPathComponent("litertlm_cache")

        try await worker.load(
          modelPath: modelURL.path,
          cacheDirectory: cacheDirectory.path,
          backend: backend.rawValue
        )
        isReady = true
        status = "Gemma 4 E2B loaded with direct CLiteRTLM \(backend.rawValue)."
        AppLog.info("Direct CLiteRTLM load complete: ready=\(isReady), status=\(status)")
      } catch {
        worker.unload()
        let detail = "Direct CLiteRTLM load failed: backend=\(backend.rawValue), path=\(modelURL.path), size=\(size), error=\(AppLog.describe(error))"
        status = detail
        AppLog.error(detail)
        throw RuntimeError.nativeLoadFailed(detail)
      }
    }

    func generate(
      prompt: String,
      maxTokens: Int,
      onToken: @escaping @MainActor (String) async -> Void
    ) async throws -> GenerationResult {
      guard isReady else { throw RuntimeError.runtimeNotReady }

      let started = Date()
      let formattedPrompt = Self.gemmaTurnPrompt(prompt)
      var text = ""
      var outputTokens = 0

      for try await token in worker.generateStreaming(
        prompt: formattedPrompt,
        temperature: 0.7,
        maxTokens: Int32(maxTokens)
      ) {
        if Task.isCancelled { throw CancellationError() }
        text += token
        outputTokens += max(1, token.split { $0.isWhitespace || $0.isNewline }.count)
        await onToken(token)
      }

      return GenerationResult(
        text: text,
        inputTokensEstimate: max(1, prompt.split { $0.isWhitespace || $0.isNewline }.count),
        outputTokensEstimate: outputTokens,
        elapsedSeconds: Date().timeIntervalSince(started)
      )
    }

    func cancel() {
      worker.unload()
      isReady = false
      status = "Runtime reset."
    }

    private static func gemmaTurnPrompt(_ prompt: String) -> String {
      "<|turn>user\n\(prompt)\n<turn|>\n<|turn>model\n"
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
  }
#endif
