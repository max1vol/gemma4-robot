import SwiftUI

struct ContentView: View {
  @EnvironmentObject private var modelStore: ModelStore
  @EnvironmentObject private var bridge: PiBridgeClient
  @State private var didAutoLoadRuntime = false

  var body: some View {
    NavigationStack {
      Form {
        Section("Model") {
          TextField("Model URL", text: $modelStore.modelURLString, axis: .vertical)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .font(.footnote.monospaced())
            .onChange(of: modelStore.modelURLString) { _, _ in
              modelStore.refresh()
            }

          TextField("Projector URL", text: $modelStore.projectorURLString, axis: .vertical)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .font(.footnote.monospaced())
            .onChange(of: modelStore.projectorURLString) { _, _ in
              modelStore.refresh()
            }

          HStack {
            Button {
              Task { await modelStore.downloadModel() }
            } label: {
              Label(modelStore.hasModel ? "Re-download Model" : "Download Model", systemImage: "arrow.down.circle")
            }
            .disabled(modelStore.isDownloading)

            Button(role: .destructive) {
              modelStore.deleteModel()
            } label: {
              Label("Delete Model", systemImage: "trash")
            }
            .disabled(!modelStore.hasModel || modelStore.isDownloading)
          }

          Text(modelStore.status)
            .font(.footnote)
            .foregroundStyle(.secondary)

          HStack {
            Button {
              Task { await modelStore.downloadProjector() }
            } label: {
              Label(modelStore.hasProjector ? "Re-download Projector" : "Download Projector", systemImage: "arrow.down.doc")
            }
            .disabled(modelStore.isDownloadingProjector)

            Button(role: .destructive) {
              modelStore.deleteProjector()
            } label: {
              Label("Delete Projector", systemImage: "trash")
            }
            .disabled(!modelStore.hasProjector || modelStore.isDownloadingProjector)
          }

          Text(modelStore.projectorStatus)
            .font(.footnote)
            .foregroundStyle(.secondary)

          if modelStore.isDownloading {
            if let progress = modelStore.downloadProgressFraction {
              ProgressView(value: progress) {
                Text(modelStore.downloadDetail)
                  .font(.footnote.monospacedDigit())
                  .lineLimit(2)
                  .minimumScaleFactor(0.85)
              }
              .progressViewStyle(.linear)
            } else {
              ProgressView {
                Text(modelStore.downloadDetail)
                  .font(.footnote.monospacedDigit())
                  .lineLimit(2)
                  .minimumScaleFactor(0.85)
              }
            }
          }

          if modelStore.isDownloadingProjector {
            if let progress = modelStore.projectorDownloadProgressFraction {
              ProgressView(value: progress) {
                Text(modelStore.projectorDownloadDetail)
                  .font(.footnote.monospacedDigit())
                  .lineLimit(2)
                  .minimumScaleFactor(0.85)
              }
              .progressViewStyle(.linear)
            } else {
              ProgressView {
                Text(modelStore.projectorDownloadDetail)
                  .font(.footnote.monospacedDigit())
                  .lineLimit(2)
                  .minimumScaleFactor(0.85)
              }
            }
          }
        }

        Section("Runtime") {
          LabeledContent("Backend", value: bridge.backend.displayName)

          Button {
            Task {
              await bridge.loadRuntime(
                modelURL: modelStore.modelURLForLoading,
                projectorURL: modelStore.projectorURLForLoading
              )
            }
          } label: {
            Label(
              bridge.isLoadingRuntime ? "Loading..." : bridge.runtimeReady ? "Reload GPU Model" : "Load GPU Model",
              systemImage: "memorychip"
            )
          }
          .disabled(!modelStore.hasModel || !modelStore.hasProjector || bridge.isLoadingRuntime)

          if bridge.isLoadingRuntime {
            ProgressView()
          }

          LabeledContent("Runtime", value: bridge.runtimeName)
          LabeledContent("Ready", value: bridge.runtimeReady ? "yes" : "no")
          LabeledContent("Load attempts", value: "\(bridge.runtimeLoadCount)")
          if !bridge.loadedModelSize.isEmpty {
            LabeledContent("Model size", value: bridge.loadedModelSize)
          }
          Text(bridge.runtimeStatus)
            .font(.footnote)
            .foregroundStyle(.secondary)
          if !bridge.loadedModelPath.isEmpty {
            Text(bridge.loadedModelPath)
              .font(.caption2.monospaced())
              .foregroundStyle(.secondary)
              .textSelection(.enabled)
          }
        }

        Section("Pi Bridge") {
          TextField("Bridge WebSocket", text: $bridge.bridgeURLString)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .font(.footnote.monospaced())

          HStack {
            Button {
              bridge.isConnected ? bridge.disconnect() : bridge.connect()
            } label: {
              Label(
                bridge.isConnected ? "Disconnect" : "Connect",
                systemImage: bridge.isConnected ? "bolt.slash" : "bolt.horizontal.circle"
              )
            }

            Button {
              bridge.cancelGeneration()
            } label: {
              Label("Cancel", systemImage: "stop.circle")
            }
          }

          LabeledContent("State", value: bridge.connectionState)
          if !bridge.lastError.isEmpty {
            Text(bridge.lastError)
              .font(.footnote)
              .foregroundStyle(.red)
          }
        }

        Section("TTS Config") {
          Picker("TTS model", selection: $bridge.selectedTTSBackend) {
            ForEach(PhoneTTSBackend.selectableCases, id: \.rawValue) { backend in
              Text(backend.displayName).tag(backend.rawValue)
            }
          }
          .pickerStyle(.menu)
          .onChange(of: bridge.selectedTTSBackend) { _, newValue in
            bridge.selectedTTSVoice = PhoneTTSBackend.parse(newValue).defaultVoice
            Task { await bridge.previewSelectedTTSVoice() }
          }

          Picker("TTS voice", selection: $bridge.selectedTTSVoice) {
            ForEach(PhoneTTSBackend.parse(bridge.selectedTTSBackend).availableVoices, id: \.self) { voice in
              Text(voice).tag(voice)
            }
          }
          .pickerStyle(.menu)
          .onChange(of: bridge.selectedTTSVoice) { _, _ in
            Task { await bridge.previewSelectedTTSVoice() }
          }

          Button {
            Task { await bridge.previewSelectedTTSVoice() }
          } label: {
            Label("Preview Voice", systemImage: "speaker.wave.2")
          }
          .disabled(bridge.isSpeakingLocalTest)

          Text(bridge.lastTTSStatus)
            .font(.footnote)
            .foregroundStyle(.secondary)
        }

        Section("Model Test") {
          TextField("Prompt", text: $bridge.localTestPrompt, axis: .vertical)
            .textInputAutocapitalization(.never)
            .autocorrectionDisabled()
            .font(.footnote.monospaced())
            .lineLimit(2...5)

          HStack {
            Button {
              Task { await bridge.speakLocalTestResponse() }
            } label: {
              Label("Speak", systemImage: "speaker.wave.2")
            }
            .disabled(bridge.localTestResponse.isEmpty || bridge.isSpeakingLocalTest)

            Button {} label: {
              Label(
                bridge.isRecordingAudio ? "Recording Audio..." : "Hold to Talk",
                systemImage: bridge.isRecordingAudio ? "mic.fill" : "mic"
              )
            }
            .disabled(!bridge.runtimeReady || bridge.isRunningLocalTest)
            .simultaneousGesture(
              DragGesture(minimumDistance: 0)
                .onChanged { _ in
                  guard !bridge.isRecordingAudio else { return }
                  Task { await bridge.startAudioCapture() }
                }
                .onEnded { _ in
                  Task { await bridge.finishAudioCaptureAndSend() }
                }
            )
          }

          Button {
            Task { await bridge.sendLocalTestPrompt() }
          } label: {
            Label("Send to Model", systemImage: "paperplane")
          }
          .disabled(!bridge.runtimeReady || bridge.isRunningLocalTest || bridge.isRecordingAudio)

          if bridge.isRunningLocalTest || bridge.isRecordingAudio || bridge.isSpeakingLocalTest {
            ProgressView()
          }

          Text(bridge.localTestStatus)
            .font(.footnote)
            .foregroundStyle(.secondary)

          Text(bridge.audioInputStatus)
            .font(.footnote)
            .foregroundStyle(.secondary)

          if !bridge.lastProbeResponse.isEmpty {
            LabeledContent("Last probe", value: bridge.lastProbeResponse)
              .font(.footnote)
          }

          Text(bridge.localTestResponse.isEmpty ? "No local response yet." : bridge.localTestResponse)
            .font(.footnote.monospaced())
            .textSelection(.enabled)
        }

        Section("Stats") {
          LabeledContent("Requests", value: "\(bridge.totalRequests)")
          LabeledContent("Input tokens", value: "\(bridge.inputTokens)")
          LabeledContent("Output tokens", value: "\(bridge.outputTokens)")
          LabeledContent("Last speed", value: String(format: "%.1f tok/s", bridge.lastTokensPerSecond))
          LabeledContent("Pose requests", value: "\(bridge.poseRequests)")
          if !bridge.lastPoseBackend.isEmpty {
            LabeledContent("Pose backend", value: bridge.lastPoseBackend)
          }
          if !bridge.lastPoseModel.isEmpty {
            LabeledContent("Pose model", value: bridge.lastPoseModel)
          }
          LabeledContent("Pose latency", value: String(format: "%.1f ms", bridge.lastPoseLatency * 1000))
          Text(bridge.lastPoseStatus)
            .font(.footnote)
            .foregroundStyle(.secondary)
          LabeledContent("TTS requests", value: "\(bridge.ttsRequests)")
          if !bridge.lastTTSBackend.isEmpty {
            LabeledContent("TTS backend", value: bridge.lastTTSBackend)
          }
          LabeledContent("TTS latency", value: String(format: "%.2f s", bridge.lastTTSLatency))
          Text(bridge.lastTTSStatus)
            .font(.footnote)
            .foregroundStyle(.secondary)
        }

        Section("Last Prompt") {
          Text(bridge.lastPromptSnippet.isEmpty ? "No prompt yet." : bridge.lastPromptSnippet)
            .font(.footnote.monospaced())
            .textSelection(.enabled)
        }

        Section("Generated") {
          Text(bridge.generatedSnippet.isEmpty ? "No generated text yet." : bridge.generatedSnippet)
            .font(.footnote.monospaced())
            .textSelection(.enabled)
        }
      }
      .navigationTitle("")
      .navigationBarTitleDisplayMode(.inline)
      .toolbar {
        ToolbarItem(placement: .principal) {
          Text("Gemma Inference Server")
            .font(.system(size: 15, weight: .semibold))
            .lineLimit(1)
            .minimumScaleFactor(0.7)
            .allowsTightening(true)
        }
      }
      .task {
        await autoLoadRuntimeForDebugging()
      }
    }
  }

  @MainActor
  private func autoLoadRuntimeForDebugging() async {
    guard !didAutoLoadRuntime else { return }
    didAutoLoadRuntime = true
    if Self.noAutoLoadRequested {
      AppLog.info("Skipping launch auto-load because --no-auto-load was requested")
      if Self.autoConnectRequested {
        AppLog.info("Launch requested Pi bridge auto-connect without runtime auto-load")
        bridge.connect()
      }
      await runLaunchTTSPreviewIfRequested()
      return
    }
    modelStore.refresh()
    guard modelStore.hasModel, modelStore.hasProjector else {
      AppLog.info("Skipping launch auto-load because model or projector is missing")
      if Self.autoConnectRequested {
        AppLog.info("Launch requested Pi bridge auto-connect without runtime auto-load")
        bridge.connect()
      }
      await runLaunchTTSPreviewIfRequested()
      return
    }
    bridge.useRecommendedBackend()
    AppLog.info("Launch auto-load enabled: loading Gemma runtime on llama.cpp GPU with projector")
    await bridge.loadRuntime(
      modelURL: modelStore.modelURLForLoading,
      projectorURL: modelStore.projectorURLForLoading
    )

    if Self.autoConnectRequested {
      AppLog.info("Launch requested Pi bridge auto-connect")
      bridge.connect()
    }
    await runLaunchTTSPreviewIfRequested()
    await runLaunchAudioRecordingSmokeIfRequested()
  }

  @MainActor
  private func runLaunchTTSPreviewIfRequested() async {
    guard Self.ttsPreviewRequested else { return }
    AppLog.info("Launch requested TTS preview")
    await bridge.previewSelectedTTSVoice()
  }

  @MainActor
  private func runLaunchAudioRecordingSmokeIfRequested() async {
    guard Self.audioRecordingSmokeRequested else { return }
    AppLog.info("Launch requested audio recording smoke test")
    await bridge.runAudioCaptureSmokeTest()
  }

  private static var autoConnectRequested: Bool {
    let arguments = ProcessInfo.processInfo.arguments
    let environment = ProcessInfo.processInfo.environment
    return arguments.contains("--auto-connect")
      || environment["GEMMAPI_AUTO_CONNECT"] == "1"
  }

  private static var noAutoLoadRequested: Bool {
    let arguments = ProcessInfo.processInfo.arguments
    let environment = ProcessInfo.processInfo.environment
    return arguments.contains("--no-auto-load")
      || environment["GEMMAPI_NO_AUTO_LOAD"] == "1"
  }

  private static var ttsPreviewRequested: Bool {
    let arguments = ProcessInfo.processInfo.arguments
    let environment = ProcessInfo.processInfo.environment
    return arguments.contains("--tts-preview")
      || environment["GEMMAPI_TTS_PREVIEW"] == "1"
  }

  private static var audioRecordingSmokeRequested: Bool {
    let arguments = ProcessInfo.processInfo.arguments
    let environment = ProcessInfo.processInfo.environment
    return arguments.contains("--audio-recording-smoke")
      || environment["GEMMAPI_AUDIO_RECORDING_SMOKE"] == "1"
  }
}
