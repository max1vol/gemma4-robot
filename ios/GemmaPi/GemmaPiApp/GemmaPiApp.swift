import SwiftUI
import UIKit

@main
struct GemmaPiApp: App {
  @StateObject private var modelStore = ModelStore()
  @StateObject private var bridge = PiBridgeClient()

  var body: some Scene {
    WindowGroup {
      ContentView()
        .environmentObject(modelStore)
        .environmentObject(bridge)
        .onAppear {
          UIApplication.shared.isIdleTimerDisabled = true
          AppLog.info("Idle timer disabled while Gemma Inference Server is foregrounded")
        }
    }
  }
}
