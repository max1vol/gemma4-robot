import AVFoundation
import Foundation

#if canImport(FluidAudio)
  import FluidAudio
#endif

enum PhoneTTSBackend: String, CaseIterable, Sendable {
  case fluidKokoroAne = "fluid-kokoro-ane"
  case fluidPocket = "fluid-pocket"

  static let selectableCases: [PhoneTTSBackend] = [.fluidKokoroAne]

  static func parse(_ text: String?) -> PhoneTTSBackend {
    guard let text, !text.isEmpty else { return .fluidKokoroAne }
    return PhoneTTSBackend(rawValue: text.lowercased()) ?? .fluidKokoroAne
  }

  var displayName: String {
    switch self {
    case .fluidKokoroAne:
      return "FluidAudio KokoroAne"
    case .fluidPocket:
      return "FluidAudio PocketTTS"
    }
  }

  var defaultVoice: String {
    switch self {
    case .fluidKokoroAne:
      return "af_heart"
    case .fluidPocket:
      return "alba"
    }
  }

  var availableVoices: [String] {
    switch self {
    case .fluidKokoroAne:
      return ["af_heart"]
    case .fluidPocket:
      return [
        "alba",
        "anna",
        "azelma",
        "bill_boerst",
        "caro_davy",
        "charles",
        "cosette",
        "eponine",
        "estelle",
        "eve",
        "fantine",
        "george",
        "giovanni",
        "jane",
        "javert",
        "jean",
        "juergen",
        "lola",
        "marius",
        "mary",
        "michael",
        "paul",
        "peter_yearsley",
        "rafael",
        "stuart_bell",
        "vera"
      ]
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
  case unavailable
  case emptyText

  var errorDescription: String? {
    switch self {
    case .unavailable:
      return "FluidAudio is not linked in this app build."
    case .emptyText:
      return "TTS text is empty."
    }
  }
}

actor PhoneTTSRuntime {
  static let sampleRate = 24_000

  #if canImport(FluidAudio)
    private var kokoroAne: KokoroAneManager?
    private var pocket: PocketTtsManager?
  #endif

  func synthesizeStreaming(
    text rawText: String,
    backend backendName: String?,
    voice requestedVoice: String?,
    onAudioChunk: @escaping @Sendable (Data) async -> Void
  ) async throws -> PhoneTTSResult {
    let text = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
    guard !text.isEmpty else { throw PhoneTTSError.emptyText }

    let backend = PhoneTTSBackend.parse(backendName)
    let voice = normalizedVoice(requestedVoice, backend: backend)

    #if canImport(FluidAudio)
      switch backend {
      case .fluidKokoroAne:
        return try await synthesizeKokoroAne(text: text, voice: voice, onAudioChunk: onAudioChunk)
      case .fluidPocket:
        return try await synthesizePocket(text: text, voice: voice, onAudioChunk: onAudioChunk)
      }
    #else
      _ = voice
      _ = onAudioChunk
      throw PhoneTTSError.unavailable
    #endif
  }

  func benchmark(text: String) async -> [PhoneTTSBenchmarkRow] {
    var rows: [PhoneTTSBenchmarkRow] = []
    for backend in PhoneTTSBackend.selectableCases {
      do {
        let result = try await synthesizeStreaming(
          text: text,
          backend: backend.rawValue,
          voice: backend.defaultVoice
        ) { _ in }
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
    return trimmed.isEmpty ? backend.defaultVoice : trimmed
  }

  #if canImport(FluidAudio)
    private func ensureKokoroAne() async throws -> KokoroAneManager {
      if let kokoroAne {
        return kokoroAne
      }
      let manager = KokoroAneManager(variant: .english, defaultVoice: PhoneTTSBackend.fluidKokoroAne.defaultVoice)
      let started = Date()
      AppLog.info("FluidAudio KokoroAne load starting")
      try await manager.initialize(preloadVoices: [PhoneTTSBackend.fluidKokoroAne.defaultVoice])
      kokoroAne = manager
      AppLog.info(String(format: "FluidAudio KokoroAne load complete in %.2fs", Date().timeIntervalSince(started)))
      return manager
    }

    private func ensurePocket() async throws -> PocketTtsManager {
      if let pocket {
        return pocket
      }
      let manager = PocketTtsManager(defaultVoice: PhoneTTSBackend.fluidPocket.defaultVoice, language: .english)
      let started = Date()
      AppLog.info("FluidAudio PocketTTS load starting")
      try await manager.initialize()
      pocket = manager
      AppLog.info(String(format: "FluidAudio PocketTTS load complete in %.2fs", Date().timeIntervalSince(started)))
      return manager
    }

    private func synthesizeKokoroAne(
      text: String,
      voice: String,
      onAudioChunk: @escaping @Sendable (Data) async -> Void
    ) async throws -> PhoneTTSResult {
      let manager = try await ensureKokoroAne()
      let started = Date()
      AppLog.info("FluidAudio KokoroAne synth starting: voice=\(voice), chars=\(text.count)")
      let result = try await manager.synthesizeDetailed(text: text, voice: voice)
      var chunks = 0
      var audioBytes = 0
      var firstAudioSeconds = 0.0
      let samplesPerChunk = Self.sampleRate / 5
      var index = 0
      while index < result.samples.count {
        let end = min(index + samplesPerChunk, result.samples.count)
        let chunk = Self.pcmS16LEData(samples: Array(result.samples[index..<end]))
        if chunks == 0 {
          firstAudioSeconds = Date().timeIntervalSince(started)
        }
        chunks += 1
        audioBytes += chunk.count
        await onAudioChunk(chunk)
        index = end
      }
      let elapsed = Date().timeIntervalSince(started)
      AppLog.info(String(format: "FluidAudio KokoroAne synth complete: %.2fs audio, %.2fs wall, chunks=%d", result.durationSeconds, elapsed, chunks))
      return PhoneTTSResult(
        backend: .fluidKokoroAne,
        voice: voice,
        sampleRate: result.sampleRate,
        audioSeconds: result.durationSeconds,
        elapsedSeconds: elapsed,
        firstAudioSeconds: firstAudioSeconds,
        chunks: chunks,
        audioBytes: audioBytes
      )
    }

    private func synthesizePocket(
      text: String,
      voice: String,
      onAudioChunk: @escaping @Sendable (Data) async -> Void
    ) async throws -> PhoneTTSResult {
      let manager = try await ensurePocket()
      let started = Date()
      AppLog.info("FluidAudio PocketTTS synth starting: voice=\(voice), chars=\(text.count)")
      let stream = try await manager.synthesizeStreaming(text: text, voice: voice)
      var chunks = 0
      var audioBytes = 0
      var sampleCount = 0
      var firstAudioSeconds = 0.0
      for try await frame in stream {
        let chunk = Self.pcmS16LEData(samples: frame.samples)
        if chunks == 0 {
          firstAudioSeconds = Date().timeIntervalSince(started)
        }
        chunks += 1
        sampleCount += frame.samples.count
        audioBytes += chunk.count
        await onAudioChunk(chunk)
      }
      let elapsed = Date().timeIntervalSince(started)
      let audioSeconds = Double(sampleCount) / Double(Self.sampleRate)
      AppLog.info(String(format: "FluidAudio PocketTTS synth complete: %.2fs audio, %.2fs wall, chunks=%d", audioSeconds, elapsed, chunks))
      return PhoneTTSResult(
        backend: .fluidPocket,
        voice: voice,
        sampleRate: Self.sampleRate,
        audioSeconds: audioSeconds,
        elapsedSeconds: elapsed,
        firstAudioSeconds: firstAudioSeconds,
        chunks: chunks,
        audioBytes: audioBytes
      )
    }
  #endif

  nonisolated static func pcmS16LEData(samples: [Float]) -> Data {
    var data = Data()
    data.reserveCapacity(samples.count * 2)
    for sample in samples {
      let clamped = max(-1.0, min(1.0, sample))
      var value = Int16((clamped * 32767.0).rounded()).littleEndian
      withUnsafeBytes(of: &value) { bytes in
        data.append(contentsOf: bytes)
      }
    }
    return data
  }
}
