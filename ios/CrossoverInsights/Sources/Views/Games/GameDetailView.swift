import SwiftUI

struct GameDetailView: View {
    let game: MobileGameSummary
    @EnvironmentObject private var store: AppStore

    private var detail: MobileGameDetailResponse? {
        store.gameDetail(for: game.id)
    }

    private var recommendations: [Recommendation] {
        detail?.recommendations ?? [game.topRecommendation]
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                matchupHeader
                recommendationsSection
                contextSection
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 120)
        }
        .background(AppTheme.background.ignoresSafeArea())
        .navigationBarTitleDisplayMode(.inline)
        .toolbar {
            ToolbarItem(placement: .principal) {
                Text("CROSSOVER INSIGHTS")
                    .font(.label(14, weight: .black))
                    .foregroundStyle(AppTheme.primary)
            }
        }
        .task(id: game.id) {
            await store.reloadGameDetail(game.id)
        }
    }

    private var matchupHeader: some View {
        SurfaceCard {
            VStack(spacing: 20) {
                HStack {
                    Spacer()
                    Label(headerStatusText, systemImage: "circle.fill")
                        .font(.label(10, weight: .bold))
                        .foregroundStyle(headerAccent)
                }

                HStack(alignment: .center) {
                    teamColumn(name: game.awayTeam, code: game.topRecommendation.awayCode)
                    Spacer()
                    VStack(spacing: 8) {
                        if let commenceText = formattedCommenceTime {
                            Text(commenceText)
                                .font(.headline(26))
                                .foregroundStyle(AppTheme.text)
                        } else {
                            Text(DateFormatter.shortDisplay.string(from: game.date))
                                .font(.headline(26))
                                .foregroundStyle(AppTheme.text)
                        }
                        Text("Top angle: \(game.topRecommendation.shortMarketLabel)")
                            .font(.label(10, weight: .bold))
                            .foregroundStyle(AppTheme.textMuted)
                        Text("\(game.recommendationCount) recommendations in this game")
                            .font(.label(10, weight: .bold))
                            .foregroundStyle(AppTheme.primary)
                    }
                    Spacer()
                    teamColumn(name: game.homeTeam, code: game.topRecommendation.homeCode)
                }
            }
        }
    }

    @ViewBuilder
    private var recommendationsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Top Insights")
                    .font(.headline(20))
                    .foregroundStyle(AppTheme.text)
                Spacer()
                Text("\(recommendations.count) active picks found")
                    .font(.label(11, weight: .bold))
                    .foregroundStyle(AppTheme.textMuted)
            }

            if detail == nil {
                switch store.gameLoadState(for: game.id) {
                case .idle, .loading:
                    LoadingCard(title: "Loading live game detail…")
                case let .failed(message):
                    MessageCard(
                        title: "Game Detail Unavailable",
                        message: message,
                        buttonTitle: "Retry"
                    ) {
                        Task { await store.reloadGameDetail(game.id) }
                    }
                case .empty:
                    MessageCard(
                        title: "No Game Detail",
                        message: "The API returned no recommendations for this matchup."
                    )
                case .loaded:
                    EmptyView()
                }
            } else {
                ForEach(recommendations) { recommendation in
                    SurfaceCard(background: AppTheme.surface) {
                        VStack(alignment: .leading, spacing: 18) {
                            HStack(alignment: .top) {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(recommendation.displayTitle)
                                        .font(.headline(22))
                                        .foregroundStyle(AppTheme.text)
                                    Text(recommendation.shortMarketLabel.uppercased())
                                        .font(.label(12, weight: .bold))
                                        .foregroundStyle(AppTheme.primary)
                                }
                                Spacer()
                                VStack(alignment: .trailing, spacing: 4) {
                                    Text("\(recommendation.edgeText) EDGE")
                                        .font(.label(18, weight: .bold))
                                        .foregroundStyle(AppTheme.primary)
                                    Text("\(recommendation.confidenceText) CONFIDENCE")
                                        .font(.label(10, weight: .bold))
                                        .foregroundStyle(AppTheme.textMuted)
                                }
                            }

                            HStack {
                                MetricTile(title: "Book Line", value: (recommendation.sportsbookOdds ?? 0).americanOddsText)
                                MetricTile(title: "Fair Line", value: (recommendation.fairOdds ?? 0).americanOddsText, accent: AppTheme.primary)
                                MetricTile(title: "Projected", value: recommendation.projectionText)
                            }

                            if let firstReason = recommendation.reasons.first {
                                Text(firstReason.detail)
                                    .font(.system(size: 14, weight: .medium))
                                    .foregroundStyle(AppTheme.textMuted)
                            }

                            DisclosureGroup("View Model Data Explanations") {
                                VStack(spacing: 12) {
                                    modelMeter(label: "Matchup Advantage", value: Int((recommendation.selectedProbability ?? 0.5) * 100))
                                    modelMeter(label: "Line Efficiency", value: Int(min(99, max(30, recommendation.edge * 800))))
                                }
                                .padding(.top, 12)
                            }
                            .tint(AppTheme.primary)
                            .font(.label(11, weight: .bold))

                            HStack(spacing: 12) {
                                NavigationLink(value: recommendation) {
                                    Text("Open Pick")
                                        .font(.label(12, weight: .bold))
                                        .foregroundStyle(AppTheme.text)
                                        .frame(maxWidth: .infinity)
                                        .padding(.vertical, 12)
                                        .background(AppTheme.surfaceHighest)
                                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                                }

                                Button {
                                    store.toggleParlay(recommendation)
                                } label: {
                                    Text(store.isInParlay(recommendation) ? "Queued" : "Add To Parlay")
                                        .font(.label(12, weight: .bold))
                                        .foregroundStyle(AppTheme.surfaceLowest)
                                        .frame(maxWidth: .infinity)
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

    private var contextSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Context")
                .font(.headline(20))
                .foregroundStyle(AppTheme.text)

            injuryContextCard
            lineupCard
        }
    }

    @ViewBuilder
    private var injuryContextCard: some View {
        if let detail, !detail.injuries.isEmpty {
            SurfaceCard {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Injury Context")
                        .font(.label(12, weight: .bold))
                        .foregroundStyle(AppTheme.text)

                    ForEach(detail.injuries) { injury in
                        HStack {
                            VStack(alignment: .leading, spacing: 4) {
                                Text(injury.playerName)
                                    .font(.label(13, weight: .bold))
                                    .foregroundStyle(AppTheme.text)
                                Text(injury.rawReason ?? injury.projectedAvailability ?? injury.reportStatus)
                                    .font(.system(size: 11, weight: .medium))
                                    .foregroundStyle(AppTheme.textMuted)
                            }
                            Spacer()
                            StatusBadge(
                                text: injury.normalizedStatus ?? injury.reportStatus,
                                accent: injuryStatusAccent(for: injury),
                                fill: injuryStatusAccent(for: injury).opacity(0.12)
                            )
                        }
                        .padding(14)
                        .background(AppTheme.surfaceHighest)
                        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                    }
                }
            }
        } else {
            MessageCard(
                title: "Injury Context",
                message: "No structured injury rows were returned for this game."
            )
        }
    }

    @ViewBuilder
    private var lineupCard: some View {
        if let detail, !detail.lineupSummary.isEmpty {
            SurfaceCard(background: AppTheme.surfaceLowest) {
                VStack(alignment: .leading, spacing: 16) {
                    Text("Projected Lineups")
                        .font(.label(12, weight: .bold))
                        .foregroundStyle(AppTheme.text)
                    ForEach(detail.lineupSummary) { team in
                        VStack(alignment: .leading, spacing: 10) {
                            HStack {
                                Text(team.teamAbbrev)
                                    .font(.headline(18))
                                    .foregroundStyle(AppTheme.text)
                                Spacer()
                                if let projectedReturningStarters = team.projectedReturningStarters {
                                    Text("Returning \(projectedReturningStarters)")
                                        .font(.label(10, weight: .bold))
                                        .foregroundStyle(AppTheme.primary)
                                }
                            }
                            if team.starters.isEmpty {
                                Text("No starter rows were returned for this team.")
                                    .font(.system(size: 12, weight: .medium))
                                    .foregroundStyle(AppTheme.textMuted)
                            } else {
                                ForEach(team.starters) { starter in
                                    HStack {
                                        VStack(alignment: .leading, spacing: 2) {
                                            Text(starter.playerName)
                                                .font(.label(12, weight: .bold))
                                                .foregroundStyle(AppTheme.text)
                                            Text(starter.projectionReason.replacingOccurrences(of: "_", with: " "))
                                                .font(.label(10, weight: .bold))
                                                .foregroundStyle(AppTheme.textMuted)
                                        }
                                        Spacer()
                                        Text(starter.starterProbability.formattedPercent(digits: 0))
                                            .font(.label(11, weight: .bold))
                                            .foregroundStyle(AppTheme.primary)
                                    }
                                }
                            }
                        }
                        .padding(.vertical, 4)
                    }
                }
            }
        } else {
            MessageCard(
                title: "Projected Lineups",
                message: "No lineup projection summary was returned for this game."
            )
        }
    }

    private var headerStatusText: String {
        switch store.gameLoadState(for: game.id) {
        case .loaded:
            return "LIVE DETAIL"
        case .loading:
            return "SYNCING"
        case .empty:
            return "NO DATA"
        case .failed:
            return "ERROR"
        case .idle:
            return "LOADING"
        }
    }

    private var headerAccent: Color {
        switch store.gameLoadState(for: game.id) {
        case .loaded:
            return AppTheme.primary
        case .failed:
            return AppTheme.error
        case .empty:
            return AppTheme.warning
        case .idle, .loading:
            return AppTheme.textMuted
        }
    }

    private var formattedCommenceTime: String? {
        let formatter = ISO8601DateFormatter()
        guard let commenceTime = detail?.commenceTime ?? game.commenceTime,
              let date = formatter.date(from: commenceTime) else { return nil }
        return DateFormatter.timestampDisplay.string(from: date)
    }

    private func teamColumn(name: String, code: String) -> some View {
        VStack(spacing: 10) {
            TeamBubble(code: code)
                .frame(width: 78, height: 78)
            Text(name.components(separatedBy: " ").last ?? name)
                .font(.headline(26))
                .foregroundStyle(AppTheme.text)
            Text("Live slate view")
                .font(.label(10, weight: .bold))
                .foregroundStyle(AppTheme.textMuted)
        }
    }

    private func modelMeter(label: String, value: Int) -> some View {
        VStack(spacing: 6) {
            HStack {
                Text(label.uppercased())
                    .font(.label(10, weight: .bold))
                    .foregroundStyle(AppTheme.textMuted)
                Spacer()
                Text("\(value)/100")
                    .font(.label(10, weight: .bold))
                    .foregroundStyle(AppTheme.primary)
            }
            GeometryReader { proxy in
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .fill(AppTheme.surfaceHighest)
                    .overlay(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 4, style: .continuous)
                            .fill(AppTheme.primary)
                            .frame(width: proxy.size.width * CGFloat(value) / 100)
                    }
            }
            .frame(height: 6)
        }
    }

    private func injuryStatusAccent(for injury: GameInjury) -> Color {
        switch (injury.normalizedStatus ?? injury.reportStatus).lowercased() {
        case "out", "inactive_other":
            return AppTheme.error
        case "probable", "available":
            return AppTheme.primary
        default:
            return AppTheme.warning
        }
    }
}
