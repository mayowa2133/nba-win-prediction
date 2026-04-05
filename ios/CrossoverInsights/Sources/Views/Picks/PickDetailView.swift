import SwiftUI

struct PickDetailView: View {
    @EnvironmentObject private var store: AppStore
    let recommendation: Recommendation

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                hero
                rangeAndReasons
                milestones
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
    }

    private var hero: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack {
                StatusBadge(text: "\(recommendation.player == nil ? "NBA GAME" : "NBA PROP") | \(recommendation.statusText)", accent: AppTheme.primary, fill: AppTheme.surfaceHighest)
                Text("\(recommendation.awayCode) @ \(recommendation.homeCode)")
                    .font(.label(12, weight: .medium))
                    .foregroundStyle(AppTheme.textMuted)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text(recommendation.displayTitle.uppercased())
                    .font(.headline(44))
                    .foregroundStyle(AppTheme.text)
                Text(recommendation.heroSubtitle)
                    .font(.headline(34))
                    .foregroundStyle(AppTheme.primary)
            }

            HStack(spacing: 16) {
                SurfaceCard {
                    VStack(alignment: .leading, spacing: 12) {
                        Label("Crossover Edge", systemImage: "chart.line.uptrend.xyaxis")
                            .font(.label(11, weight: .bold))
                            .foregroundStyle(AppTheme.textMuted)
                        Text(recommendation.edgeText)
                            .font(.headline(48))
                            .foregroundStyle(AppTheme.primary)
                        Text("Fair line \(recommendation.fairLine.formattedCompact) versus market \(recommendation.sportsbookLine.formattedCompact).")
                            .font(.system(size: 13, weight: .medium))
                            .foregroundStyle(AppTheme.textMuted)
                    }
                }

                SurfaceCard {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Confidence Score")
                            .font(.label(11, weight: .bold))
                            .foregroundStyle(AppTheme.textMuted)
                        Text(recommendation.confidenceText)
                            .font(.headline(40))
                            .foregroundStyle(AppTheme.text)
                        ConfidenceMeter(probability: recommendation.selectedProbability ?? 0.5)
                    }
                }
            }

            SurfaceCard(background: AppTheme.surface) {
                VStack(alignment: .leading, spacing: 18) {
                    Text("Market Comparison")
                        .font(.label(11, weight: .bold))
                        .foregroundStyle(AppTheme.textMuted)

                    HStack {
                        VStack(alignment: .leading, spacing: 6) {
                            Text("Model Projection")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundStyle(AppTheme.textMuted)
                            Text(recommendation.fairLine.formattedCompact)
                                .font(.headline(30))
                                .foregroundStyle(AppTheme.text)
                        }
                        Spacer()
                        VStack(alignment: .trailing, spacing: 6) {
                            Text("Sportsbook Line")
                                .font(.system(size: 12, weight: .medium))
                                .foregroundStyle(AppTheme.textMuted)
                            Text(recommendation.sportsbookLine.formattedCompact)
                                .font(.headline(30))
                                .foregroundStyle(AppTheme.textMuted)
                        }
                    }

                    HStack {
                        metricRow(title: "Win Probability", value: recommendation.probabilityText, accent: AppTheme.primary)
                        metricRow(title: "Fair Odds", value: (recommendation.fairOdds ?? 0).americanOddsText, accent: AppTheme.text)
                        metricRow(title: "Implied", value: recommendation.marketImpliedText, accent: AppTheme.text)
                    }

                    HStack(spacing: 12) {
                        Button {
                            store.toggleParlay(recommendation)
                        } label: {
                            Text(store.isInParlay(recommendation) ? "Remove from Parlay" : "Add to Parlay")
                                .font(.label(12, weight: .bold))
                                .foregroundStyle(AppTheme.surfaceLowest)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 14)
                                .background(AppTheme.primary)
                                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                        }

                        Button {
                            store.toggleSaved(recommendation)
                        } label: {
                            Text(store.isSaved(recommendation) ? "Saved" : "Save Pick")
                                .font(.label(12, weight: .bold))
                                .foregroundStyle(AppTheme.text)
                                .frame(maxWidth: .infinity)
                                .padding(.vertical, 14)
                                .background(AppTheme.surfaceHighest)
                                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                        }
                    }
                }
            }
        }
    }

    private var rangeAndReasons: some View {
        VStack(spacing: 16) {
            SurfaceCard {
                VStack(alignment: .leading, spacing: 18) {
                    HStack {
                        Text("Likely Range Variance")
                            .font(.headline(20))
                            .foregroundStyle(AppTheme.text)
                        Spacer()
                        Text("PROBABILITY DENSITY")
                            .font(.label(10, weight: .bold))
                            .foregroundStyle(AppTheme.textMuted)
                    }

                    RangeBars(recommendation: recommendation)

                    HStack {
                        Text("\((recommendation.likelyRangeLow ?? recommendation.fairLine - 3).formattedCompact0)")
                            .font(.label(11, weight: .bold))
                            .foregroundStyle(AppTheme.textMuted)
                        Spacer()
                        Text("LIKELY: \(recommendation.likelyRangeText)")
                            .font(.label(11, weight: .bold))
                            .foregroundStyle(AppTheme.text)
                        Spacer()
                        Text("\((recommendation.likelyRangeHigh ?? recommendation.fairLine + 3).formattedCompact0)")
                            .font(.label(11, weight: .bold))
                            .foregroundStyle(AppTheme.textMuted)
                    }
                }
            }

            SurfaceCard {
                VStack(alignment: .leading, spacing: 14) {
                    Text("Why we like this pick")
                        .font(.headline(20))
                        .foregroundStyle(AppTheme.text)

                    ForEach(recommendation.reasons, id: \.self) { reason in
                        VStack(alignment: .leading, spacing: 4) {
                            Text(reason.label)
                                .font(.label(11, weight: .bold))
                                .foregroundStyle(AppTheme.text)
                            Text(reason.detail)
                                .font(.system(size: 14, weight: .medium))
                                .foregroundStyle(AppTheme.textMuted)
                        }
                    }

                    Label("Latest snapshot \(formattedTimestamp(recommendation.dataTimestamp))", systemImage: "checkmark.seal.fill")
                        .font(.label(11, weight: .bold))
                        .foregroundStyle(AppTheme.primary)
                }
            }
        }
    }

    private var milestones: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Milestone Ladder")
                .font(.headline(20))
                .foregroundStyle(AppTheme.text)

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 140), spacing: 12)], spacing: 12) {
                ForEach(recommendation.milestoneCards, id: \.self) { milestone in
                    SurfaceCard(background: AppTheme.surfaceHighest) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("\(milestone.threshold.formattedCompact0)+ \(recommendation.marketLabel.uppercased())")
                                .font(.label(11, weight: .bold))
                                .foregroundStyle(AppTheme.textMuted)
                            HStack(alignment: .bottom) {
                                Text(milestone.probability.formattedPercent(digits: 0))
                                    .font(.headline(28))
                                    .foregroundStyle(AppTheme.text)
                                Spacer()
                                Text(milestone.badgeText)
                                    .font(.label(10, weight: .bold))
                                    .foregroundStyle(milestone.badgeText == "TARGET" ? AppTheme.primary : AppTheme.textMuted)
                            }
                        }
                    }
                }
            }
        }
    }

    private func metricRow(title: String, value: String, accent: Color) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title)
                .font(.system(size: 12, weight: .medium))
                .foregroundStyle(AppTheme.textMuted)
            Text(value)
                .font(.label(18, weight: .bold))
                .foregroundStyle(accent)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func formattedTimestamp(_ value: String) -> String {
        let formatter = ISO8601DateFormatter()
        if let date = formatter.date(from: value) {
            return DateFormatter.timestampDisplay.string(from: date)
        }
        return value
    }
}
