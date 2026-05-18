import Foundation

#if canImport(llama)
  import llama

  private let llamaQuietLogCallback: ggml_log_callback = { level, text, _ in
    guard level == GGML_LOG_LEVEL_WARN || level == GGML_LOG_LEVEL_ERROR else { return }
    guard let text else { return }
    let message = String(cString: text).trimmingCharacters(in: .whitespacesAndNewlines)
    guard !message.isEmpty else { return }
    AppLog.info("llama.cpp: \(message)")
  }

  private enum LlamaRuntimeError: LocalizedError {
    case modelLoadFailed(String)
    case projectorLoadFailed(String)
    case contextLoadFailed
    case tokenizationFailed(String)
    case mediaPreparationFailed(String)
    case decodeFailed(String)

    var errorDescription: String? {
      switch self {
      case .modelLoadFailed(let path):
        return "llama.cpp failed to load model at \(path)"
      case .projectorLoadFailed(let path):
        return "llama.cpp failed to load multimodal projector at \(path)"
      case .contextLoadFailed:
        return "llama.cpp failed to create inference context"
      case .tokenizationFailed(let text):
        return "llama.cpp failed to tokenize prompt: \(text.prefix(80))"
      case .mediaPreparationFailed(let detail):
        return "llama.cpp failed to prepare media input: \(detail)"
      case .decodeFailed(let detail):
        return "llama.cpp decode failed: \(detail)"
      }
    }
  }

  private enum LlamaBackendBootstrap {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var initialized = false

    static func ensureInitialized() {
      lock.lock()
      defer { lock.unlock() }

      guard !initialized else { return }
      llama_log_set(llamaQuietLogCallback, nil)
      llama_backend_init()
      initialized = true
      AppLog.info("llama.cpp backend initialized")
    }
  }

  private func llamaBatchClear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
  }

  private func llamaBatchAdd(
    _ batch: inout llama_batch,
    token: llama_token,
    position: llama_pos,
    sequenceIDs: [llama_seq_id],
    logits: Bool
  ) {
    let index = Int(batch.n_tokens)
    batch.token[index] = token
    batch.pos[index] = position
    batch.n_seq_id[index] = Int32(sequenceIDs.count)
    for sequenceIndex in 0..<sequenceIDs.count {
      batch.seq_id[index]![sequenceIndex] = sequenceIDs[sequenceIndex]
    }
    batch.logits[index] = logits ? 1 : 0
    batch.n_tokens += 1
  }

  private final class LlamaContextWorker: @unchecked Sendable {
    private let queue: DispatchQueue
    private let cancelLock = NSLock()
    private var cancelled = false
    private var model: OpaquePointer?
    private var context: OpaquePointer?
    private var mtmdContext: OpaquePointer?
    private var vocab: OpaquePointer?
    private var backend: InferenceBackend = .gpu
    private var mediaMarker = "<__media__>"
    private var supportsVision = false
    private var supportsAudio = false

    init(labelSuffix: String) {
      queue = DispatchQueue(
        label: "com.gemma4robot.gemmapi.llama.\(labelSuffix)",
        qos: .userInitiated
      )
    }

    deinit {
      unload()
    }

    func load(modelPath: String, backend: InferenceBackend, contextSize: UInt32 = 2048) async throws {
      try await load(modelPath: modelPath, projectorPath: nil, backend: backend, contextSize: contextSize)
    }

    func load(
      modelPath: String,
      projectorPath: String?,
      backend: InferenceBackend,
      contextSize: UInt32 = 2048
    ) async throws {
      try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
        queue.async {
          do {
            self.deleteLoadedModel()
            self.backend = backend
            self.resetCancellation()
            LlamaBackendBootstrap.ensureInitialized()

            var modelParams = llama_model_default_params()
            modelParams.n_gpu_layers = backend == .gpu ? -1 : 0
            AppLog.info("llama.cpp load begin: path=\(modelPath), backend=\(backend.rawValue), n_gpu_layers=\(modelParams.n_gpu_layers)")

            guard let loadedModel = llama_model_load_from_file(modelPath, modelParams) else {
              throw LlamaRuntimeError.modelLoadFailed(modelPath)
            }

            let nThreads = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
            var contextParams = llama_context_default_params()
            contextParams.n_ctx = contextSize
            contextParams.n_batch = contextSize
            contextParams.n_ubatch = min(contextSize, UInt32(512))
            contextParams.n_seq_max = 2
            contextParams.n_threads = Int32(nThreads)
            contextParams.n_threads_batch = Int32(nThreads)

            guard let loadedContext = llama_init_from_model(loadedModel, contextParams) else {
              llama_model_free(loadedModel)
              throw LlamaRuntimeError.contextLoadFailed
            }

            var loadedMtmd: OpaquePointer?
            if let projectorPath {
              mtmd_helper_log_set(llamaQuietLogCallback, nil)
              var mtmdParams = mtmd_context_params_default()
              mtmdParams.use_gpu = backend == .gpu
              mtmdParams.print_timings = true
              mtmdParams.n_threads = Int32(nThreads)
              AppLog.info("llama.cpp mtmd load begin: path=\(projectorPath), backend=\(backend.rawValue)")
              guard let mtmd = mtmd_init_from_file(projectorPath, loadedModel, mtmdParams) else {
                llama_free(loadedContext)
                llama_model_free(loadedModel)
                throw LlamaRuntimeError.projectorLoadFailed(projectorPath)
              }
              loadedMtmd = mtmd
              self.mediaMarker = String(cString: mtmd_default_marker())
              self.supportsVision = mtmd_support_vision(mtmd)
              self.supportsAudio = mtmd_support_audio(mtmd)
              AppLog.info(
                "llama.cpp mtmd load complete: marker=\(self.mediaMarker), vision=\(self.supportsVision), audio=\(self.supportsAudio)"
              )
            } else {
              self.mediaMarker = "<__media__>"
              self.supportsVision = false
              self.supportsAudio = false
              AppLog.info("llama.cpp mtmd projector not loaded; media prompts will fail until a projector is loaded")
            }

            self.model = loadedModel
            self.context = loadedContext
            self.mtmdContext = loadedMtmd
            self.vocab = llama_model_get_vocab(loadedModel)
            AppLog.info("llama.cpp load complete: backend=\(backend.rawValue), threads=\(nThreads), n_ctx=\(contextSize)")
            continuation.resume()
          } catch {
            self.deleteLoadedModel()
            continuation.resume(throwing: error)
          }
        }
      }
    }

    func generateStreaming(
      prompt: String,
      media: [GemmaMediaInput],
      maxTokens: Int,
      temperature: Float
    ) -> AsyncThrowingStream<String, Error> {
      AsyncThrowingStream { continuation in
        queue.async {
          do {
            try self.generate(
              prompt: prompt,
              media: media,
              maxTokens: maxTokens,
              temperature: temperature
            ) { token in
              continuation.yield(token)
            }
            continuation.finish()
          } catch {
            continuation.finish(throwing: error)
          }
        }
      }
    }

    func generateStreaming(
      prompt: String,
      maxTokens: Int,
      temperature: Float
    ) -> AsyncThrowingStream<String, Error> {
      generateStreaming(prompt: prompt, media: [], maxTokens: maxTokens, temperature: temperature)
    }

    func cancelGeneration() {
      cancelLock.lock()
      cancelled = true
      cancelLock.unlock()
    }

    func unload() {
      queue.sync {
        deleteLoadedModel()
      }
    }

    private func generate(
      prompt: String,
      media: [GemmaMediaInput],
      maxTokens: Int,
      temperature: Float,
      onToken: (String) -> Void
    ) throws {
      guard let context, let vocab else { throw RuntimeError.runtimeNotReady }
      resetCancellation()
      llama_memory_clear(llama_get_memory(context), true)

      let initialCursor: Int32
      let batchCapacity = max(8, maxTokens + 8)
      var batch = llama_batch_init(Int32(batchCapacity), 0, 1)
      defer { llama_batch_free(batch) }

      if media.isEmpty {
        let promptTokens = try tokenize(prompt, addBOS: true, parseSpecial: true)
        let promptBatchCapacity = max(batchCapacity, promptTokens.count + maxTokens + 8)
        if promptBatchCapacity > batchCapacity {
          llama_batch_free(batch)
          batch = llama_batch_init(Int32(promptBatchCapacity), 0, 1)
        }
        llamaBatchClear(&batch)
        for (index, token) in promptTokens.enumerated() {
          llamaBatchAdd(
            &batch,
            token: token,
            position: Int32(index),
            sequenceIDs: [0],
            logits: index == promptTokens.count - 1
          )
        }

        let prefillCode = llama_decode(context, batch)
        guard prefillCode == 0 else {
          throw LlamaRuntimeError.decodeFailed("prefill code=\(prefillCode)")
        }
        initialCursor = Int32(promptTokens.count)
      } else {
        initialCursor = try prefillMultimodalPrompt(prompt, media: media, context: context)
      }

      let sampler = Self.makeSampler(temperature: temperature)
      defer { llama_sampler_free(sampler) }

      var cursor = initialCursor
      var partialUTF8: [CChar] = []

      for _ in 0..<maxTokens {
        if isCancelled() { throw CancellationError() }

        let nextToken = llama_sampler_sample(sampler, context, -1)
        if llama_vocab_is_eog(vocab, nextToken) {
          break
        }
        llama_sampler_accept(sampler, nextToken)

        let pieceBytes = tokenToPiece(nextToken)
        if !pieceBytes.isEmpty {
          partialUTF8.append(contentsOf: pieceBytes)
          if let text = Self.stringFromValidUTF8(partialUTF8) {
            partialUTF8.removeAll()
            if !text.isEmpty {
              onToken(text)
            }
          }
        }

        llamaBatchClear(&batch)
        llamaBatchAdd(
          &batch,
          token: nextToken,
          position: cursor,
          sequenceIDs: [0],
          logits: true
        )
        let decodeCode = llama_decode(context, batch)
        guard decodeCode == 0 else {
          throw LlamaRuntimeError.decodeFailed("decode code=\(decodeCode)")
        }
        cursor += 1
      }

      if !partialUTF8.isEmpty, let tail = Self.stringFromValidUTF8(partialUTF8), !tail.isEmpty {
        onToken(tail)
      }
    }

    private func prefillMultimodalPrompt(
      _ prompt: String,
      media: [GemmaMediaInput],
      context: OpaquePointer
    ) throws -> Int32 {
      guard let mtmdContext else {
        throw RuntimeError.multimodalUnavailable("A multimodal projector is required for audio/image input.")
      }
      for input in media {
        if input.isAudio && !supportsAudio {
          throw RuntimeError.multimodalUnavailable("The loaded projector does not report audio input support.")
        }
        if input.isImage && !supportsVision {
          throw RuntimeError.multimodalUnavailable("The loaded projector does not report image input support.")
        }
      }

      let markerCount = prompt.components(separatedBy: mediaMarker).count - 1
      let mediaSummary = media
        .map { "\($0.mimeType):\($0.data.count)b" }
        .joined(separator: ",")
      AppLog.info(
        "llama.cpp mtmd prefill begin: media_count=\(media.count), marker_count=\(markerCount), prompt_utf8=\(prompt.utf8.count), media=[\(mediaSummary)]"
      )

      let bitmaps = try makeBitmaps(media, mtmdContext: mtmdContext)
      defer {
        bitmaps.forEach { mtmd_bitmap_free($0) }
      }

      guard let chunks = mtmd_input_chunks_init() else {
        throw LlamaRuntimeError.mediaPreparationFailed("mtmd_input_chunks_init returned NULL")
      }
      defer { mtmd_input_chunks_free(chunks) }

      var bitmapPointers = bitmaps.map { Optional($0) }
      let bitmapCount = bitmapPointers.count
      let tokenized = prompt.withCString { promptPointer -> Int32 in
        var inputText = mtmd_input_text(
          text: promptPointer,
          add_special: true,
          parse_special: true
        )
        return bitmapPointers.withUnsafeMutableBufferPointer { bitmapBuffer in
          mtmd_tokenize(
            mtmdContext,
            chunks,
            &inputText,
            bitmapBuffer.baseAddress,
            bitmapCount
          )
        }
      }
      guard tokenized == 0 else {
        throw LlamaRuntimeError.tokenizationFailed("mtmd_tokenize returned \(tokenized); markers=\(media.count)")
      }

      let chunkCount = mtmd_input_chunks_size(chunks)
      let chunkTokens = mtmd_helper_get_n_tokens(chunks)
      let chunkPositions = mtmd_helper_get_n_pos(chunks)
      AppLog.info(
        "llama.cpp mtmd tokenize complete: chunks=\(chunkCount), tokens=\(chunkTokens), positions=\(chunkPositions)"
      )

      var newPast: llama_pos = 0
      let evalStart = CFAbsoluteTimeGetCurrent()
      let evalCode = mtmd_helper_eval_chunks(
        mtmdContext,
        context,
        chunks,
        0,
        0,
        512,
        true,
        &newPast
      )
      guard evalCode == 0 else {
        throw LlamaRuntimeError.decodeFailed("mtmd prefill code=\(evalCode)")
      }
      let evalElapsed = CFAbsoluteTimeGetCurrent() - evalStart
      AppLog.info(
        String(format: "llama.cpp mtmd eval complete: new_past=%d, elapsed=%.3fs", Int(newPast), evalElapsed)
      )
      return Int32(newPast)
    }

    private func makeBitmaps(
      _ media: [GemmaMediaInput],
      mtmdContext: OpaquePointer
    ) throws -> [OpaquePointer] {
      var bitmaps: [OpaquePointer] = []
      do {
        for input in media {
          let bitmap = input.data.withUnsafeBytes { rawBuffer -> OpaquePointer? in
            guard let base = rawBuffer.bindMemory(to: UInt8.self).baseAddress else { return nil }
            return mtmd_helper_bitmap_init_from_buf(mtmdContext, base, input.data.count)
          }
          guard let bitmap else {
            throw LlamaRuntimeError.mediaPreparationFailed(
              "\(input.displayName ?? input.mimeType) could not be decoded by mtmd"
            )
          }
          let bitmapIsAudio = mtmd_bitmap_is_audio(bitmap)
          let bitmapBytes = mtmd_bitmap_get_n_bytes(bitmap)
          let sampleCount = bitmapIsAudio ? bitmapBytes / MemoryLayout<Float>.stride : 0
          AppLog.info(
            "llama.cpp mtmd media decoded: name=\(input.displayName ?? "inline"), mime=\(input.mimeType), input_bytes=\(input.data.count), bitmap_audio=\(bitmapIsAudio), bitmap_bytes=\(bitmapBytes), samples=\(sampleCount), nx=\(mtmd_bitmap_get_nx(bitmap)), ny=\(mtmd_bitmap_get_ny(bitmap))"
          )
          if let displayName = input.displayName {
            displayName.withCString { mtmd_bitmap_set_id(bitmap, $0) }
          }
          bitmaps.append(bitmap)
        }
        return bitmaps
      } catch {
        bitmaps.forEach { mtmd_bitmap_free($0) }
        throw error
      }
    }

    private func tokenize(_ text: String, addBOS: Bool, parseSpecial: Bool) throws -> [llama_token] {
      guard let vocab else { throw RuntimeError.runtimeNotReady }
      let utf8Count = text.utf8.count
      var capacity = max(32, utf8Count + 16)
      var tokens = [llama_token](repeating: 0, count: capacity)

      var tokenCount = tokens.withUnsafeMutableBufferPointer { tokenBuffer in
        text.withCString { textPointer in
          llama_tokenize(
            vocab,
            textPointer,
            Int32(utf8Count),
            tokenBuffer.baseAddress,
            Int32(capacity),
            addBOS,
            parseSpecial
          )
        }
      }

      if tokenCount < 0 {
        capacity = Int(-tokenCount)
        tokens = [llama_token](repeating: 0, count: capacity)
        tokenCount = tokens.withUnsafeMutableBufferPointer { tokenBuffer in
          text.withCString { textPointer in
            llama_tokenize(
              vocab,
              textPointer,
              Int32(utf8Count),
              tokenBuffer.baseAddress,
              Int32(capacity),
              addBOS,
              parseSpecial
            )
          }
        }
      }

      guard tokenCount > 0 else {
        throw LlamaRuntimeError.tokenizationFailed(text)
      }

      return Array(tokens.prefix(Int(tokenCount)))
    }

    private func tokenToPiece(_ token: llama_token) -> [CChar] {
      guard let vocab else { return [] }
      var capacity: Int32 = 16
      var buffer = [CChar](repeating: 0, count: Int(capacity))
      var count = buffer.withUnsafeMutableBufferPointer { pointer in
        llama_token_to_piece(vocab, token, pointer.baseAddress, capacity, 0, false)
      }

      if count < 0 {
        capacity = -count
        buffer = [CChar](repeating: 0, count: Int(capacity))
        count = buffer.withUnsafeMutableBufferPointer { pointer in
          llama_token_to_piece(vocab, token, pointer.baseAddress, capacity, 0, false)
        }
      }

      guard count > 0 else { return [] }
      return Array(buffer.prefix(Int(count)))
    }

    private static func stringFromValidUTF8(_ bytes: [CChar]) -> String? {
      let codeUnits = bytes.map { UInt8(bitPattern: $0) }
      return String(bytes: codeUnits, encoding: .utf8)
    }

    private static func makeSampler(temperature: Float) -> UnsafeMutablePointer<llama_sampler>? {
      let params = llama_sampler_chain_default_params()
      let sampler = llama_sampler_chain_init(params)
      if temperature <= 0 {
        llama_sampler_chain_add(sampler, llama_sampler_init_greedy())
      } else {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95, 1))
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature))
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234))
      }
      return sampler
    }

    private func isCancelled() -> Bool {
      cancelLock.lock()
      let value = cancelled
      cancelLock.unlock()
      return value
    }

    private func resetCancellation() {
      cancelLock.lock()
      cancelled = false
      cancelLock.unlock()
    }

    private func deleteLoadedModel() {
      if let mtmdContext {
        mtmd_free(mtmdContext)
        self.mtmdContext = nil
      }
      if let context {
        llama_free(context)
        self.context = nil
      }
      if let model {
        llama_model_free(model)
        self.model = nil
      }
      vocab = nil
      supportsVision = false
      supportsAudio = false
    }
  }

  final class LlamaGemmaRuntime: GemmaRuntime {
    private let worker = LlamaContextWorker(labelSuffix: "bridge")
    private(set) var isReady = false
    private(set) var status = "llama.cpp runtime available."
    let name = "llama.cpp"

    func loadModel(at modelURL: URL, backend: InferenceBackend) async throws {
      try await loadModel(at: modelURL, projectorURL: nil, backend: backend)
    }

    func loadModel(at modelURL: URL, projectorURL: URL?, backend: InferenceBackend) async throws {
      isReady = false
      let size = Self.fileSize(modelURL).map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) } ?? "unknown"
      let projectorSize = projectorURL.flatMap(Self.fileSize).map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) } ?? "none"
      AppLog.info(
        "llama.cpp runtime load starting: path=\(modelURL.path), size=\(size), projector=\(projectorURL?.path ?? "none"), projector_size=\(projectorSize), backend=\(backend.rawValue)"
      )

      do {
        try await worker.load(modelPath: modelURL.path, projectorPath: projectorURL?.path, backend: backend)
        isReady = true
        status = projectorURL == nil
          ? "Gemma 4 E2B loaded with llama.cpp \(backend.rawValue)."
          : "Gemma 4 E2B loaded with llama.cpp \(backend.rawValue) + mtmd."
        AppLog.info("llama.cpp runtime load finished: ready=\(isReady), status=\(status)")
      } catch {
        isReady = false
        let detail = "llama.cpp load failed: backend=\(backend.rawValue), path=\(modelURL.path), size=\(size), projector=\(projectorURL?.path ?? "none"), projector_size=\(projectorSize), error=\(AppLog.describe(error))"
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
      try await generate(prompt: prompt, media: [], maxTokens: maxTokens, onToken: onToken)
    }

    func generate(
      prompt: String,
      media: [GemmaMediaInput],
      maxTokens: Int,
      onToken: @escaping @MainActor (String) async -> Void
    ) async throws -> GenerationResult {
      guard isReady else { throw RuntimeError.runtimeNotReady }

      let started = Date()
      let formattedPrompt = Self.gemmaTurnPrompt(prompt, mediaCount: media.count)
      var text = ""
      var outputTokens = 0

      for try await token in worker.generateStreaming(
        prompt: formattedPrompt,
        media: media,
        maxTokens: maxTokens,
        temperature: 0.7
      ) {
        if Task.isCancelled { throw CancellationError() }
        text += token
        outputTokens += 1
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
      worker.cancelGeneration()
      status = "Runtime cancellation requested."
    }

    private static func gemmaTurnPrompt(_ prompt: String, mediaCount: Int) -> String {
      let marker = Array(repeating: "<__media__>", count: mediaCount).joined(separator: "\n")
      let mediaPrefix = marker.isEmpty ? "" : "\(marker)\n"
      return """
      <|turn>system
      Do not use emojis.
      <turn|>
      <|turn>user
      \(mediaPrefix)\(prompt)
      <turn|>
      <|turn>model
      """
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
