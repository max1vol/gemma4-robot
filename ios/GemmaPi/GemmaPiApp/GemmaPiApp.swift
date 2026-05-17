import SwiftUI

@main
struct GemmaPiApp: App {
  @StateObject private var modelStore = ModelStore()
  @StateObject private var bridge = PiBridgeClient()

  var body: some Scene {
    WindowGroup {
      ContentView()
        .environmentObject(modelStore)
        .environmentObject(bridge)
    }
  }
}
