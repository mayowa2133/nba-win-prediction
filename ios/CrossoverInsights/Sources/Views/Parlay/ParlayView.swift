import SwiftUI

struct ParlayView: View {
    @EnvironmentObject private var store: AppStore
    private let stake: Double = 100

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                header
                picksSection
                recommendationsSection
                summarySection
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 120)
        }
        .background(AppTheme.background.ignoresSafeArea())
        .navigationTitle("Parlay")
        .navigationBarTitleDisplayMode(.inline)
    }

    private var picks: [Recommendation] {
        store.parlayRecommendations
    }

    private var combinedOdds: Int {
        store.combinedParlayOdds(for: picks)
    }

    private var payout: Double {
        store.payout(for: combinedOdds, stake: stake)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Parlay Builder")
                .font(.headline(36))
                .foregroundStyle(AppTheme.text)
            Text("\(picks.count) active picks in betslip")
                .font(.label(12, weight: .bold))
                .foregroundStyle(AppTheme.textMuted)
        }
    }

    private var picksSection: some View {
        VStack(alignment: .leading, spacing: 14) {
            ForEach(picks) { recommendation in
                SurfaceCard(background: AppTheme.surfaceHigh) {
                    HStack(spacing: 16) {
                        RoundedRectangle(cornerRadius: 12, style: .continuous)
                            .fill(AppTheme.surfaceLowest)
                            .frame(width: 54, height: 54)
                            .overlay {
                                Text(recommendation.awayCode)
                                    .font(.label(15, weight: .black))
                                    .foregroundStyle(AppTheme.primary)
                            }

                        VStack(alignment: .leading, spacing: 6) {
                            Text(recommendation.displayTitle)
                                .font(.headline(20))
                                .foregroundStyle(AppTheme.text)
                            Text("\(recommendation.shortMarketLabel) | \(recommendation.awayCode) @ \(recommendation.homeCode)")
                                .font(.label(10, weight: .bold))
                                .foregroundStyle(AppTheme.textMuted)
                        }

                        Spacer()

                        VStack(alignment: .trailing, spacing: 8) {
                            ConfidenceMeter(probability: recommendation.selectedProbability ?? 0.5, segments: 5, compact: true)
                            Text((recommendation.sportsbookOdds ?? 0).americanOddsText)
                                .font(.label(20, weight: .bold))
                                .foregroundStyle(AppTheme.primary)
                        }

                        Button {
                            store.toggleParlay(recommendation)
                        } label: {
                            Image(systemName: "xmark")
                                .foregroundStyle(AppTheme.textMuted)
                        }
                    }
                }
            }

            if store.hasCorrelationWarning(for: picks) {
                SurfaceCard(background: AppTheme.error.opacity(0.08)) {
                    HStack(alignment: .top, spacing: 12) {
                        Image(systemName: "exclamationmark.triangle.fill")
                            .foregroundStyle(AppTheme.error)
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Correlation Warning")
                                .font(.label(11, weight: .bold))
                                .foregroundStyle(AppTheme.error)
                            Text("Multiple picks come from the same game. Sportsbooks may adjust pricing or restrict the combination.")
                                .font(.system(size: 13, weight: .medium))
                                .foregroundStyle(AppTheme.textMuted)
                        }
                    }
                }
            }

            if picks.isEmpty {
                SurfaceCard {
                    Text("Your queue is empty. Add a few picks from Home or Games to start building combinations.")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(AppTheme.textMuted)
                }
            }
        }
    }

    private var recommendationsSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Label("Model Recommendations", systemImage: "sparkles")
                .font(.headline(20))
                .foregroundStyle(AppTheme.text)

            if store.parlaySuggestions.isEmpty {
                MessageCard(
                    title: "No Suggested Parlays",
                    message: "The live home endpoint has not returned any recommended combinations yet."
                )
            } else {
                ForEach(store.parlaySuggestions) { suggestion in
                    Button {
                        suggestion.recommendations.forEach { store.addToParlay($0) }
                    } label: {
                        SurfaceCard(background: AppTheme.surfaceLowest) {
                            VStack(alignment: .leading, spacing: 12) {
                                HStack {
                                    StatusBadge(text: suggestion.title, accent: suggestion.title.contains("Value") ? AppTheme.primary : AppTheme.textMuted, fill: suggestion.title.contains("Value") ? AppTheme.primary.opacity(0.1) : AppTheme.surfaceHighest)
                                    Spacer()
                                    Text(suggestion.combinedOdds.americanOddsText)
                                        .font(.label(20, weight: .bold))
                                        .foregroundStyle(AppTheme.text)
                                }
                                Text(suggestion.recommendations.map { $0.player ?? $0.marketLabel }.joined(separator: " + "))
                                    .font(.system(size: 14, weight: .medium))
                                    .foregroundStyle(AppTheme.textMuted)
                                HStack {
                                    Text("Win Probability: \(suggestion.combinedProbability.formattedPercent(digits: 0))")
                                        .font(.label(11, weight: .bold))
                                        .foregroundStyle(AppTheme.textMuted)
                                    Spacer()
                                    Label("Add", systemImage: "plus.circle.fill")
                                        .font(.label(11, weight: .bold))
                                        .foregroundStyle(AppTheme.primary)
                                }
                            }
                        }
                    }
                    .buttonStyle(.plain)
                }
            }
        }
    }

    private var summarySection: some View {
        VStack(spacing: 16) {
            SurfaceCard(background: AppTheme.surfaceHigh) {
                VStack(alignment: .leading, spacing: 18) {
                    Text("Total Parlay Odds")
                        .font(.label(11, weight: .bold))
                        .foregroundStyle(AppTheme.textMuted)
                    Text(combinedOdds.americanOddsText)
                        .font(.headline(56))
                        .foregroundStyle(AppTheme.primary)

                    HStack {
                        Text("Risk Level")
                        Spacer()
                        Text(riskLabel)
                            .foregroundStyle(AppTheme.primary)
                    }
                    .font(.label(12, weight: .bold))

                    HStack(spacing: 6) {
                        ForEach(0..<3, id: \.self) { index in
                            RoundedRectangle(cornerRadius: 6, style: .continuous)
                                .fill(index < riskSegments ? AppTheme.primary : AppTheme.surfaceHighest)
                                .frame(height: 8)
                        }
                    }

                    VStack(spacing: 12) {
                        HStack {
                            Text("Wager")
                            Spacer()
                            Text("$\(stake, specifier: "%.2f")")
                        }
                        HStack(alignment: .bottom) {
                            VStack(alignment: .leading, spacing: 4) {
                                Text("Estimated Payout")
                                    .font(.label(10, weight: .bold))
                                    .foregroundStyle(AppTheme.textMuted)
                                Text("$\(payout, specifier: "%.2f")")
                                    .font(.headline(28))
                                    .foregroundStyle(AppTheme.text)
                            }
                            Spacer()
                            VStack(alignment: .trailing, spacing: 4) {
                                Text("Profit")
                                    .font(.label(10, weight: .bold))
                                    .foregroundStyle(AppTheme.primary)
                                Text("$\((payout - stake), specifier: "%.2f")")
                                    .font(.label(20, weight: .bold))
                                    .foregroundStyle(AppTheme.primary)
                            }
                        }
                    }

                    Button {
                    } label: {
                        Label("Copy to Sportsbook", systemImage: "bolt.fill")
                            .font(.label(13, weight: .black))
                            .foregroundStyle(AppTheme.surfaceLowest)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 16)
                            .background(
                                LinearGradient(colors: [AppTheme.primary, AppTheme.primaryDeep], startPoint: .leading, endPoint: .trailing)
                            )
                            .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                    }
                }
            }

            SurfaceCard(background: AppTheme.surfaceLowest) {
                VStack(alignment: .leading, spacing: 10) {
                    Label("Builder Insight", systemImage: "chart.bar.doc.horizontal")
                        .font(.label(11, weight: .bold))
                        .foregroundStyle(AppTheme.textMuted)
                    Text(riskSegments >= 3 ? "Your current card leans into one game cluster. Balance it with a late-window angle if you want lower dependency risk." : "A shorter card keeps the slip cleaner and reduces compounding volatility.")
                        .font(.system(size: 14, weight: .medium))
                        .foregroundStyle(AppTheme.textMuted)
                        .italic()
                }
            }
        }
    }

    private var riskSegments: Int {
        if picks.count >= 3 || store.hasCorrelationWarning(for: picks) { return 3 }
        if picks.count == 2 { return 2 }
        return 1
    }

    private var riskLabel: String {
        switch riskSegments {
        case 3: return "Moderate"
        case 2: return "Managed"
        default: return "Low"
        }
    }
}
