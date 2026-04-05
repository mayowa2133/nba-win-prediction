import SwiftUI
import Charts

struct TrendsView: View {
    @EnvironmentObject private var store: AppStore

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                hero
                content
            }
            .padding(.horizontal, 16)
            .padding(.bottom, 120)
        }
        .background(AppTheme.background.ignoresSafeArea())
        .navigationTitle("Trends")
        .navigationBarTitleDisplayMode(.inline)
        .task {
            if case .idle = store.trendsState {
                await store.reloadTrendsScreen()
            }
        }
    }

    var hero: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("THE TRACK RECORD")
                .font(.headline(38))
                .foregroundStyle(AppTheme.text)
            Text("Performance audit, settlement transparency, and market readiness from the live API.")
                .font(.system(size: 14, weight: .medium))
                .foregroundStyle(AppTheme.textMuted)
        }
    }

    @ViewBuilder
    private var content: some View {
        if store.trends.recentSettlements.isEmpty && store.trends.chartPoints.isEmpty {
            switch store.trendsState {
            case .idle, .loading:
                LoadingCard(title: "Loading live performance data…")
            case let .failed(message):
                MessageCard(
                    title: "Performance Feed Unavailable",
                    message: message,
                    buttonTitle: "Retry"
                ) {
                    Task { await store.reloadTrendsScreen() }
                }
            case .empty:
                MessageCard(
                    title: "No Settled Results",
                    message: "The API has not returned any settled recommendations yet."
                )
                readinessSection
            case .loaded:
                EmptyView()
            }
        } else {
            metrics
            chartSection
            settlementSection
            readinessSection
        }
    }

    private var metrics: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 180), spacing: 16)], spacing: 16) {
            SurfaceCard {
                metricContent(title: "Portfolio ROI", value: store.trends.roi.formattedPercent(digits: 1), accent: AppTheme.primary)
            }
            SurfaceCard {
                metricContent(title: "CLV Edge", value: "\(store.trends.clv >= 0 ? "+" : "")\((store.trends.clv * 100).formatted(.number.precision(.fractionLength(1))))%", accent: AppTheme.text)
            }
            SurfaceCard {
                metricContent(title: "Hit Rate", value: store.trends.hitRate.formattedPercent(digits: 1), accent: AppTheme.text)
            }
        }
    }

    private var chartSection: some View {
        SurfaceCard {
            VStack(alignment: .leading, spacing: 20) {
                SectionTitle(title: "CUMULATIVE PROFIT YIELD", subtitle: "Derived directly from settled recommendation rows.")

                if store.trends.chartPoints.isEmpty {
                    MessageCard(
                        title: "No Trend Series",
                        message: "Settled recommendations exist, but the API did not return any chart points."
                    )
                } else {
                    Chart(store.trends.chartPoints) { point in
                        AreaMark(
                            x: .value("Point", point.label),
                            y: .value("ROI", point.cumulativeRoi)
                        )
                        .foregroundStyle(
                            LinearGradient(colors: [AppTheme.primary.opacity(0.22), AppTheme.primary.opacity(0.02)], startPoint: .top, endPoint: .bottom)
                        )

                        LineMark(
                            x: .value("Point", point.label),
                            y: .value("ROI", point.cumulativeRoi)
                        )
                        .foregroundStyle(AppTheme.primary)
                        .lineStyle(StrokeStyle(lineWidth: 3, lineCap: .round, lineJoin: .round))
                    }
                    .frame(height: 240)
                    .chartYAxis {
                        AxisMarks(position: .leading)
                    }
                    .chartXAxis {
                        AxisMarks(values: store.trends.chartPoints.map(\.label))
                    }
                }
            }
        }
    }

    private var settlementSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Recent Settlement")
                    .font(.headline(22))
                    .foregroundStyle(AppTheme.text)
                Spacer()
                Text("Live settled sample")
                    .font(.label(10, weight: .bold))
                    .foregroundStyle(AppTheme.textMuted)
            }

            ForEach(store.trends.recentSettlements) { recommendation in
                let isWin = recommendation.result?.lowercased() == "win"
                let isLoss = recommendation.result?.lowercased() == "loss"
                let accent = isWin ? AppTheme.primary : (isLoss ? AppTheme.error : AppTheme.textMuted)
                let icon = isWin ? "checkmark.circle.fill" : (isLoss ? "xmark.circle.fill" : "minus.circle.fill")

                SurfaceCard(background: AppTheme.surfaceHigh) {
                    HStack {
                        Image(systemName: icon)
                            .foregroundStyle(accent)
                            .font(.system(size: 20, weight: .bold))
                            .frame(width: 42, height: 42)
                            .background(accent.opacity(0.12))
                            .clipShape(Circle())

                        VStack(alignment: .leading, spacing: 4) {
                            Text("\(recommendation.displayTitle) - \(recommendation.shortMarketLabel)")
                                .font(.label(13, weight: .bold))
                                .foregroundStyle(AppTheme.text)
                            Text("\(recommendation.marketLabel.uppercased()) - \(DateFormatter.shortDisplay.string(from: recommendation.date))")
                                .font(.label(10, weight: .bold))
                                .foregroundStyle(AppTheme.textMuted)
                        }

                        Spacer()

                        VStack(alignment: .trailing, spacing: 4) {
                            Text("\(recommendation.roi ?? 0 >= 0 ? "+" : "")\((recommendation.roi ?? 0).formatted(.number.precision(.fractionLength(2))))u")
                                .font(.label(14, weight: .bold))
                                .foregroundStyle(accent)
                            Text("ODDS: \((recommendation.sportsbookOdds ?? 0).americanOddsText)")
                                .font(.label(10, weight: .bold))
                                .foregroundStyle(AppTheme.textMuted)
                        }
                    }
                }
            }
        }
    }

    @ViewBuilder
    private var readinessSection: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Market Readiness")
                .font(.headline(24))
                .foregroundStyle(AppTheme.text)

            if store.readiness.isEmpty {
                switch store.readinessState {
                case .idle, .loading:
                    LoadingCard(title: "Loading market readiness…")
                case let .failed(message):
                    MessageCard(
                        title: "Readiness Unavailable",
                        message: message
                    )
                case .empty:
                    MessageCard(
                        title: "No Readiness Data",
                        message: "The readiness endpoint returned no market status rows."
                    )
                case .loaded:
                    EmptyView()
                }
            } else {
                ForEach(store.readiness) { readiness in
                    let accent = readiness.status.lowercased() == "production" ? AppTheme.primary : (readiness.status.lowercased() == "beta" ? AppTheme.warning : Color.blue.opacity(0.85))
                    SurfaceCard {
                        VStack(alignment: .leading, spacing: 14) {
                            HStack {
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(readiness.market.replacingOccurrences(of: "_", with: " ").uppercased())
                                        .font(.label(11, weight: .bold))
                                        .foregroundStyle(AppTheme.textMuted)
                                    Text(readiness.label)
                                        .font(.headline(24))
                                        .foregroundStyle(AppTheme.text)
                                }
                                Spacer()
                                StatusBadge(text: readiness.status, accent: accent, fill: accent.opacity(0.14))
                            }

                            Text(readiness.summary)
                                .font(.system(size: 14, weight: .medium))
                                .foregroundStyle(AppTheme.textMuted)

                            HStack {
                                MetricTile(title: "Tier", value: readiness.tier)
                                MetricTile(title: "Status", value: readiness.status.capitalized, accent: accent)
                            }
                        }
                    }
                }
            }
        }
    }

    private func metricContent(title: String, value: String, accent: Color) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(title.uppercased())
                .font(.label(10, weight: .bold))
                .foregroundStyle(AppTheme.textMuted)
            Text(value)
                .font(.headline(40))
                .foregroundStyle(accent)
        }
    }
}
