import SwiftUI

@main
struct CrossoverInsightsApp: App {
    @StateObject private var store = AppStore()

    var body: some Scene {
        WindowGroup {
            RootTabView()
                .environmentObject(store)
                .task {
                    await store.loadIfNeeded()
                }
        }
    }
}
