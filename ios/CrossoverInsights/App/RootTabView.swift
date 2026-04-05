import SwiftUI

struct RootTabView: View {
    @EnvironmentObject private var store: AppStore

    var body: some View {
        TabView(selection: $store.selectedTab) {
            NavigationStack {
                HomeView()
                    .crossoverNavigationDestinations()
            }
            .tabItem {
                Label("Home", systemImage: "house")
            }
            .tag(AppTab.home)

            NavigationStack {
                GamesHubView()
                    .crossoverNavigationDestinations()
            }
            .tabItem {
                Label("Games", systemImage: "basketball")
            }
            .tag(AppTab.games)

            NavigationStack {
                ParlayView()
                    .crossoverNavigationDestinations()
            }
            .tabItem {
                Label("Parlay", systemImage: "chart.bar")
            }
            .tag(AppTab.parlay)

            NavigationStack {
                TrendsView()
                    .crossoverNavigationDestinations()
            }
            .tabItem {
                Label("Trends", systemImage: "chart.line.uptrend.xyaxis")
            }
            .tag(AppTab.trends)
        }
        .tint(AppTheme.primary)
        .preferredColorScheme(.dark)
        .toolbarBackground(AppTheme.surface, for: .tabBar)
        .toolbarColorScheme(.dark, for: .tabBar)
    }
}

private extension View {
    func crossoverNavigationDestinations() -> some View {
        navigationDestination(for: Recommendation.self) { recommendation in
            PickDetailView(recommendation: recommendation)
        }
        .navigationDestination(for: MobileGameSummary.self) { game in
            GameDetailView(game: game)
        }
    }
}
