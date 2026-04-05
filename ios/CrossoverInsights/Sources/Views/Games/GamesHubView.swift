import SwiftUI

struct GamesHubView: View {
    @EnvironmentObject private var store: AppStore

    private let columns = [
        GridItem(.adaptive(minimum: 280), spacing: 16)
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                SectionTitle(title: "GAMES", subtitle: "Live matchup cards from the current slate.")

                if store.allGames.isEmpty {
                    switch store.homeState {
                    case .idle, .loading:
                        LoadingCard(title: "Loading live games…")
                    case let .failed(message):
                        MessageCard(
                            title: "Games Unavailable",
                            message: message,
                            buttonTitle: "Retry"
                        ) {
                            Task { await store.reload() }
                        }
                    case .empty:
                        MessageCard(
                            title: "No Games Available",
                            message: "The selected date and filters returned no live game cards."
                        )
                    case .loaded:
                        EmptyView()
                    }
                } else {
                    LazyVGrid(columns: columns, spacing: 16) {
                        ForEach(store.allGames) { game in
                            NavigationLink(value: game) {
                                SurfaceCard {
                                    VStack(alignment: .leading, spacing: 18) {
                                        HStack {
                                            Text(headerText(for: game))
                                                .font(.label(11, weight: .bold))
                                                .foregroundStyle(AppTheme.textMuted)
                                            Spacer()
                                            StatusBadge(text: "\(game.recommendationCount) picks", accent: AppTheme.primary, fill: AppTheme.primary.opacity(0.1))
                                        }

                                        HStack {
                                            VStack(alignment: .leading, spacing: 12) {
                                                matchupRow(game.awayTeam, code: game.topRecommendation.awayCode)
                                                matchupRow(game.homeTeam, code: game.topRecommendation.homeCode)
                                            }
                                            Spacer()
                                            VStack(alignment: .trailing, spacing: 10) {
                                                Text("Top Edge")
                                                    .font(.label(10, weight: .bold))
                                                    .foregroundStyle(AppTheme.textMuted)
                                                Text(game.topRecommendation.edgeText)
                                                    .font(.headline(28))
                                                    .foregroundStyle(AppTheme.primary)
                                                Text(game.topRecommendation.shortMarketLabel.uppercased())
                                                    .font(.label(10, weight: .bold))
                                                    .multilineTextAlignment(.trailing)
                                                    .foregroundStyle(AppTheme.textMuted)
                                            }
                                        }
                                    }
                                }
                            }
                            .buttonStyle(.plain)
                        }
                    }
                } 
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 120)
        }
        .background(AppTheme.background.ignoresSafeArea())
        .navigationTitle("Games")
        .navigationBarTitleDisplayMode(.inline)
    }

    private func matchupRow(_ team: String, code: String) -> some View {
        HStack(spacing: 12) {
            TeamBubble(code: code)
            Text(team.components(separatedBy: " ").last ?? team)
                .font(.headline(18))
                .foregroundStyle(AppTheme.text)
        }
    }

    private func headerText(for game: MobileGameSummary) -> String {
        guard let commenceTime = game.commenceTime else {
            return DateFormatter.shortDisplay.string(from: game.date)
        }
        let formatter = ISO8601DateFormatter()
        guard let date = formatter.date(from: commenceTime) else {
            return DateFormatter.shortDisplay.string(from: game.date)
        }
        return DateFormatter.commenceDisplay.string(from: date)
    }
}
