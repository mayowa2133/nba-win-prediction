import SwiftUI

struct HomeView: View {
    @EnvironmentObject private var store: AppStore

    private let columns = [
        GridItem(.adaptive(minimum: 280), spacing: 16, alignment: .top)
    ]

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 28) {
                statusBanner
                content
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 120)
        }
        .background(AppTheme.background.ignoresSafeArea())
        .navigationTitle("Crossover Insights")
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .topBarLeading) {
                Text("CROSSOVER INSIGHTS")
                    .font(.label(14, weight: .black))
                    .foregroundStyle(AppTheme.primary)
            }
            ToolbarItem(placement: .topBarTrailing) {
                Circle()
                    .fill(AppTheme.surfaceHighest)
                    .frame(width: 28, height: 28)
                    .overlay {
                        Text("CI")
                            .font(.label(10, weight: .bold))
                            .foregroundStyle(AppTheme.primary)
                    }
            }
        }
    }

    @ViewBuilder
    private var statusBanner: some View {
        if store.isDemoModeEnabled {
            MessageCard(
                title: "Demo Mode",
                message: "Preview fixtures are enabled because CROSSOVER_DEMO_MODE=1 is set for this debug run."
            )
        } else if let apiBaseURL = store.apiBaseURL {
            SurfaceCard(background: AppTheme.surfaceLowest) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("LIVE DATA SOURCE")
                        .font(.label(10, weight: .bold))
                        .foregroundStyle(AppTheme.textMuted)
                    Text(apiBaseURL)
                        .font(.system(size: 13, weight: .medium, design: .monospaced))
                        .foregroundStyle(AppTheme.primary)
                }
            }
        }
    }

    @ViewBuilder
    private var content: some View {
        if store.availableDates.isEmpty && store.groupedGames.isEmpty {
            switch store.homeState {
            case .idle, .loading:
                LoadingCard(title: "Loading live slate from the API…")
            case let .failed(message):
                MessageCard(
                    title: "Live Slate Unavailable",
                    message: message,
                    buttonTitle: "Retry"
                ) {
                    Task { await store.reload() }
                }
            case .empty:
                MessageCard(
                    title: "No Live Slate",
                    message: "The API returned no picks for the selected filters. Try another date or widen the filters."
                )
            case .loaded:
                EmptyView()
            }
        } else {
            dateSelector
            bestEdges
            filters
            slateSection
            trendingParlays
        }
    }

    private var dateSelector: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 12) {
                ForEach(store.availableDates, id: \.self) { date in
                    Button {
                        Task { await store.selectDate(date) }
                    } label: {
                        Text(date == store.availableDates.first ? "Today, \(formattedLong(date))" : formattedShort(date))
                            .font(.label(13, weight: .bold))
                            .foregroundStyle(store.selectedDate == date ? AppTheme.primary : AppTheme.textMuted)
                            .padding(.horizontal, 18)
                            .padding(.vertical, 12)
                            .background(store.selectedDate == date ? AppTheme.surfaceHighest : AppTheme.surfaceLow)
                            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                            .overlay(
                                RoundedRectangle(cornerRadius: 14, style: .continuous)
                                    .stroke(store.selectedDate == date ? AppTheme.primary.opacity(0.25) : .clear, lineWidth: 1)
                            )
                    }
                    .disabled(store.homeState.isLoading && store.selectedDate == date)
                }
            }
        }
    }

    private var bestEdges: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                SectionTitle(title: "TODAY'S BEST EDGES", subtitle: "Live recommendations from the current slate.")
                Spacer()
                if case .loading = store.homeState {
                    ProgressView()
                        .tint(AppTheme.primary)
                }
            }

            if store.edgeRecommendations.isEmpty {
                MessageCard(
                    title: "No Featured Picks",
                    message: "The current filter combination returned no featured recommendations."
                )
            } else {
                ScrollView(.horizontal, showsIndicators: false) {
                    HStack(spacing: 16) {
                        ForEach(store.edgeRecommendations) { recommendation in
                            NavigationLink(value: recommendation) {
                                EdgeCarouselCard(recommendation: recommendation)
                            }
                            .buttonStyle(.plain)
                        }
                    }
                    .padding(.horizontal, 2)
                }
            }
        }
    }

    private var filters: some View {
        SurfaceCard {
            VStack(alignment: .leading, spacing: 16) {
                VStack(alignment: .leading, spacing: 10) {
                    Text("MARKETS")
                        .font(.label(10, weight: .bold))
                        .foregroundStyle(AppTheme.textMuted)

                    ScrollView(.horizontal, showsIndicators: false) {
                        HStack(spacing: 10) {
                            ForEach(MarketFilter.allCases) { filter in
                                Button {
                                    Task { await store.updateMarketFilter(filter) }
                                } label: {
                                    Text(filter.title)
                                        .font(.label(12, weight: .bold))
                                        .foregroundStyle(store.marketFilter == filter ? AppTheme.primary : AppTheme.textMuted)
                                        .padding(.horizontal, 14)
                                        .padding(.vertical, 8)
                                        .background(store.marketFilter == filter ? AppTheme.primary.opacity(0.08) : AppTheme.surfaceHighest)
                                        .clipShape(Capsule())
                                        .overlay(
                                            Capsule()
                                                .stroke(store.marketFilter == filter ? AppTheme.primary : AppTheme.outline.opacity(0.5), lineWidth: 1)
                                        )
                                }
                            }
                        }
                    }
                }

                HStack {
                    Text("CONFIDENCE")
                        .font(.label(10, weight: .bold))
                        .foregroundStyle(AppTheme.textMuted)

                    Spacer()

                    HStack(spacing: 6) {
                        ForEach(ConfidenceFilter.allCases) { filter in
                            Button {
                                Task { await store.updateConfidenceFilter(filter) }
                            } label: {
                                Text(filter.title)
                                    .font(.label(11, weight: .bold))
                                    .foregroundStyle(store.confidenceFilter == filter ? AppTheme.primary : AppTheme.textMuted)
                                    .padding(.horizontal, 12)
                                    .padding(.vertical, 8)
                                    .background(store.confidenceFilter == filter ? AppTheme.surfaceBright : AppTheme.surfaceHighest)
                                    .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                            }
                        }
                    }
                }
            }
        }
    }

    private var slateSection: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack(spacing: 12) {
                Text("TODAY'S NBA SLATE")
                    .font(.headline(24))
                    .foregroundStyle(AppTheme.text)
                Rectangle()
                    .fill(AppTheme.outline.opacity(0.35))
                    .frame(height: 1)
                Text("\(store.groupedGames.count) GAMES")
                    .font(.label(10, weight: .bold))
                    .foregroundStyle(AppTheme.textMuted)
            }

            if store.groupedGames.isEmpty {
                MessageCard(
                    title: "No Matchups",
                    message: "No games matched the current live-slate filters."
                )
            } else {
                LazyVGrid(columns: columns, spacing: 16) {
                    ForEach(store.groupedGames) { game in
                        NavigationLink(value: game) {
                            GameSlateCard(game: game)
                        }
                        .buttonStyle(.plain)
                    }
                }
            }
        }
    }

    private var trendingParlays: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("TRENDING PARLAYS")
                .font(.headline(24))
                .foregroundStyle(AppTheme.text)

            if store.homeTrendingParlays.isEmpty {
                MessageCard(
                    title: "No Suggestions Yet",
                    message: "The API did not return any deterministic parlay combinations for the current slate."
                )
            } else {
                ForEach(store.homeTrendingParlays) { suggestion in
                    SurfaceCard {
                        VStack(alignment: .leading, spacing: 18) {
                            HStack {
                                StatusBadge(text: suggestion.title, accent: AppTheme.primary, fill: AppTheme.primary.opacity(0.12))
                                Spacer()
                                Text("Win Probability \(suggestion.combinedProbability.formattedPercent(digits: 0))")
                                    .font(.label(11, weight: .bold))
                                    .foregroundStyle(AppTheme.textMuted)
                            }

                            VStack(alignment: .leading, spacing: 10) {
                                ForEach(suggestion.recommendations, id: \.id) { recommendation in
                                    HStack(spacing: 10) {
                                        Image(systemName: "checkmark.circle.fill")
                                            .foregroundStyle(AppTheme.primary)
                                        Text(recommendation.player ?? recommendation.shortMarketLabel)
                                            .font(.label(13, weight: .bold))
                                            .foregroundStyle(AppTheme.text)
                                            .textCase(.uppercase)
                                    }
                                }
                            }

                            HStack {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text("PARLAY ODDS")
                                        .font(.label(10, weight: .bold))
                                        .foregroundStyle(AppTheme.textMuted)
                                    Text(suggestion.combinedOdds.americanOddsText)
                                        .font(.headline(28))
                                        .foregroundStyle(AppTheme.primary)
                                }

                                Spacer()

                                Button {
                                    suggestion.recommendations.forEach { store.addToParlay($0) }
                                    store.selectedTab = .parlay
                                } label: {
                                    Text("Add To Slip")
                                        .font(.label(13, weight: .bold))
                                        .foregroundStyle(AppTheme.surfaceLowest)
                                        .padding(.horizontal, 18)
                                        .padding(.vertical, 12)
                                        .background(AppTheme.primary)
                                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private func formattedShort(_ value: String) -> String {
        DateFormatter.shortDisplay.string(from: DateFormatter.apiDate.date(from: value) ?? .now)
    }

    private func formattedLong(_ value: String) -> String {
        DateFormatter.longDisplay.string(from: DateFormatter.apiDate.date(from: value) ?? .now)
    }
}

private struct EdgeCarouselCard: View {
    let recommendation: Recommendation

    var body: some View {
        SurfaceCard {
            VStack(alignment: .leading, spacing: 16) {
                HStack(alignment: .top) {
                    StatusBadge(text: recommendation.player == nil ? recommendation.marketLabel : "Player Prop", accent: AppTheme.primary, fill: AppTheme.primary.opacity(0.12))
                    Spacer()
                    VStack(alignment: .trailing, spacing: 4) {
                        Label(recommendation.probabilityText, systemImage: "chart.line.uptrend.xyaxis")
                            .font(.label(11, weight: .bold))
                            .foregroundStyle(AppTheme.primary)
                    }
                }

                VStack(alignment: .leading, spacing: 6) {
                    Text(recommendation.displayTitle)
                        .font(.headline(22))
                        .foregroundStyle(.white)
                    Text(recommendation.shortMarketLabel.uppercased())
                        .font(.label(12, weight: .bold))
                        .foregroundStyle(AppTheme.textMuted)
                }

                HStack {
                    MetricTile(title: "Current Odds", value: (recommendation.sportsbookOdds ?? 0).americanOddsText)
                    MetricTile(title: "Projection", value: recommendation.projectionText, accent: AppTheme.primary)
                }
            }
            .frame(width: 300, alignment: .leading)
        }
    }
}

private struct GameSlateCard: View {
    let game: MobileGameSummary

    var body: some View {
        SurfaceCard {
            VStack(alignment: .leading, spacing: 18) {
                HStack {
                    Text(headerText)
                        .font(.label(11, weight: .bold))
                        .foregroundStyle(AppTheme.textMuted)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(AppTheme.surfaceHighest)
                        .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                    Spacer()
                    StatusBadge(text: game.topRecommendation.confidence == "high" ? "Best Bet" : "\(game.recommendationCount) picks", accent: AppTheme.primary, fill: AppTheme.primary.opacity(0.1))
                }

                VStack(spacing: 14) {
                    TeamRow(code: game.topRecommendation.awayCode, name: game.awayTeam.components(separatedBy: " ").last ?? game.awayTeam, value: lineText(for: false))
                    TeamRow(code: game.topRecommendation.homeCode, name: game.homeTeam.components(separatedBy: " ").last ?? game.homeTeam, value: lineText(for: true))
                }

                HStack(spacing: 12) {
                    CapsuleButton(title: "Live Detail", filled: false)
                    CapsuleButton(title: game.topRecommendation.confidence == "high" ? "Bet Matchup" : "Details", filled: game.topRecommendation.confidence == "high")
                }
            }
        }
    }

    private var headerText: String {
        if let commenceTime = game.commenceTime, let formatted = formattedCommenceTime(commenceTime) {
            return formatted
        }
        return DateFormatter.shortDisplay.string(from: game.date)
    }

    private func lineText(for home: Bool) -> String {
        let recommendation = game.topRecommendation
        if recommendation.market == "game_spread" {
            let absLine = abs(recommendation.sportsbookLine).formattedCompact
            if recommendation.selection == "away" {
                return home ? "+\(absLine)" : "-\(absLine)"
            }
            return home ? "-\(absLine)" : "+\(absLine)"
        }
        return home ? recommendation.edgeText : recommendation.probabilityText
    }

    private func formattedCommenceTime(_ value: String) -> String? {
        let formatter = ISO8601DateFormatter()
        guard let date = formatter.date(from: value) else { return nil }
        return DateFormatter.commenceDisplay.string(from: date)
    }
}

private struct TeamRow: View {
    let code: String
    let name: String
    let value: String

    var body: some View {
        HStack {
            HStack(spacing: 12) {
                TeamBubble(code: code)
                Text(name)
                    .font(.headline(18))
                    .foregroundStyle(AppTheme.text)
            }
            Spacer()
            Text(value)
                .font(.label(16, weight: .bold))
                .foregroundStyle(AppTheme.text)
        }
    }
}

private struct CapsuleButton: View {
    let title: String
    let filled: Bool

    var body: some View {
        Text(title)
            .font(.label(11, weight: .bold))
            .foregroundStyle(filled ? AppTheme.surfaceLowest : AppTheme.text)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(filled ? AppTheme.primary : AppTheme.surfaceHighest)
            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

extension DateFormatter {
    static let commenceDisplay: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "h:mm a"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter
    }()
}
