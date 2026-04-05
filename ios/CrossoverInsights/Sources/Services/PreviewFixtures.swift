import Foundation

enum PreviewFixtures {
    static let recommendations: [Recommendation] = [
        Recommendation(
            id: "rec_lebron_points",
            gameID: "game_lal_gsw",
            player: "LeBron James",
            gameDate: "2026-04-04",
            homeTeam: "Golden State Warriors",
            awayTeam: "Los Angeles Lakers",
            market: "player_points",
            selection: "over",
            sportsbookLine: 25.5,
            sportsbookOdds: -110,
            fairLine: 28.4,
            fairOdds: -180,
            edge: 0.092,
            selectedProbability: 0.642,
            marketImpliedProbability: 0.524,
            confidence: "high",
            status: "production",
            modelVersion: "preview",
            dataTimestamp: "2026-04-04T19:30:00Z",
            publishedLine: 25.5,
            publishedOdds: -110,
            publishedAt: "2026-04-04T19:30:00Z",
            likelyRangeLow: 25,
            likelyRangeHigh: 31,
            likelyRangeConfidence: 0.5,
            mostLikelyMilestone: 25,
            mostLikelyMilestoneProbability: 0.68,
            milestoneProbabilities: [
                MilestoneProbability(threshold: 20, probability: 0.88, fairOdds: -733, lineEquivalent: 19.5),
                MilestoneProbability(threshold: 25, probability: 0.68, fairOdds: -213, lineEquivalent: 24.5),
                MilestoneProbability(threshold: 30, probability: 0.42, fairOdds: 138, lineEquivalent: 29.5),
                MilestoneProbability(threshold: 35, probability: 0.18, fairOdds: 456, lineEquivalent: 34.5),
            ],
            closingLine: nil,
            closingOdds: nil,
            actualValue: nil,
            result: nil,
            clv: nil,
            roi: nil,
            reasons: [
                Reason(label: "Model vs line", detail: "Model projects 28.4 against 25.5, so the current points number still trails the fair price."),
                Reason(label: "Likely range", detail: "The model central range clusters between 25 and 31 points with room for another gear late."),
                Reason(label: "Market status", detail: "Player points is the strongest production market in the stack and carries the best readiness label."),
            ]
        ),
        Recommendation(
            id: "rec_ad_rebounds",
            gameID: "game_lal_gsw",
            player: "Anthony Davis",
            gameDate: "2026-04-04",
            homeTeam: "Golden State Warriors",
            awayTeam: "Los Angeles Lakers",
            market: "player_rebounds",
            selection: "over",
            sportsbookLine: 12.5,
            sportsbookOdds: -115,
            fairLine: 13.8,
            fairOdds: -152,
            edge: 0.071,
            selectedProbability: 0.618,
            marketImpliedProbability: 0.535,
            confidence: "high",
            status: "experimental",
            modelVersion: "preview",
            dataTimestamp: "2026-04-04T19:30:00Z",
            publishedLine: 12.5,
            publishedOdds: -115,
            publishedAt: "2026-04-04T19:30:00Z",
            likelyRangeLow: 11,
            likelyRangeHigh: 15,
            likelyRangeConfidence: 0.5,
            mostLikelyMilestone: 12,
            mostLikelyMilestoneProbability: 0.73,
            milestoneProbabilities: [
                MilestoneProbability(threshold: 10, probability: 0.89, fairOdds: -809, lineEquivalent: 9.5),
                MilestoneProbability(threshold: 12, probability: 0.73, fairOdds: -270, lineEquivalent: 11.5),
                MilestoneProbability(threshold: 14, probability: 0.41, fairOdds: 144, lineEquivalent: 13.5),
                MilestoneProbability(threshold: 16, probability: 0.19, fairOdds: 426, lineEquivalent: 15.5),
            ],
            closingLine: nil,
            closingOdds: nil,
            actualValue: nil,
            result: nil,
            clv: nil,
            roi: nil,
            reasons: [
                Reason(label: "Lineup context", detail: "The glass rate holds when the Lakers lean big, but the market is still marked experimental."),
            ]
        ),
        Recommendation(
            id: "rec_bos_spread",
            gameID: "game_bos_phi",
            player: nil,
            gameDate: "2026-04-04",
            homeTeam: "Philadelphia 76ers",
            awayTeam: "Boston Celtics",
            market: "game_spread",
            selection: "away",
            sportsbookLine: -4.5,
            sportsbookOdds: -110,
            fairLine: -6.1,
            fairOdds: -150,
            edge: 0.066,
            selectedProbability: 0.611,
            marketImpliedProbability: 0.524,
            confidence: "medium",
            status: "production",
            modelVersion: "preview",
            dataTimestamp: "2026-04-04T19:30:00Z",
            publishedLine: -4.5,
            publishedOdds: -110,
            publishedAt: "2026-04-04T19:30:00Z",
            likelyRangeLow: -8,
            likelyRangeHigh: -2,
            likelyRangeConfidence: 0.5,
            mostLikelyMilestone: nil,
            mostLikelyMilestoneProbability: nil,
            milestoneProbabilities: [],
            closingLine: nil,
            closingOdds: nil,
            actualValue: nil,
            result: nil,
            clv: nil,
            roi: nil,
            reasons: [
                Reason(label: "Win probability", detail: "The spread model still prices Boston stronger than the current market number."),
            ]
        ),
        Recommendation(
            id: "rec_tatum_points",
            gameID: "game_bos_phi",
            player: "Jayson Tatum",
            gameDate: "2026-04-04",
            homeTeam: "Philadelphia 76ers",
            awayTeam: "Boston Celtics",
            market: "player_points",
            selection: "over",
            sportsbookLine: 28.5,
            sportsbookOdds: -108,
            fairLine: 30.1,
            fairOdds: -148,
            edge: 0.058,
            selectedProbability: 0.602,
            marketImpliedProbability: 0.519,
            confidence: "medium",
            status: "production",
            modelVersion: "preview",
            dataTimestamp: "2026-04-04T19:30:00Z",
            publishedLine: 28.5,
            publishedOdds: -108,
            publishedAt: "2026-04-04T19:30:00Z",
            likelyRangeLow: 26,
            likelyRangeHigh: 33,
            likelyRangeConfidence: 0.5,
            mostLikelyMilestone: 30,
            mostLikelyMilestoneProbability: 0.52,
            milestoneProbabilities: [
                MilestoneProbability(threshold: 20, probability: 0.87, fairOdds: -669, lineEquivalent: 19.5),
                MilestoneProbability(threshold: 25, probability: 0.69, fairOdds: -223, lineEquivalent: 24.5),
                MilestoneProbability(threshold: 30, probability: 0.52, fairOdds: -108, lineEquivalent: 29.5),
                MilestoneProbability(threshold: 35, probability: 0.27, fairOdds: 270, lineEquivalent: 34.5),
            ],
            closingLine: nil,
            closingOdds: nil,
            actualValue: nil,
            result: nil,
            clv: nil,
            roi: nil,
            reasons: [
                Reason(label: "Model vs line", detail: "The projection still clears the market after accounting for the stronger road context."),
            ]
        ),
        Recommendation(
            id: "rec_okc_total",
            gameID: "game_okc_den",
            player: nil,
            gameDate: "2026-04-04",
            homeTeam: "Denver Nuggets",
            awayTeam: "Oklahoma City Thunder",
            market: "game_total",
            selection: "under",
            sportsbookLine: 234.5,
            sportsbookOdds: -108,
            fairLine: 229.8,
            fairOdds: -134,
            edge: 0.049,
            selectedProbability: 0.573,
            marketImpliedProbability: 0.519,
            confidence: "medium",
            status: "production",
            modelVersion: "preview",
            dataTimestamp: "2026-04-04T19:30:00Z",
            publishedLine: 234.5,
            publishedOdds: -108,
            publishedAt: "2026-04-04T19:30:00Z",
            likelyRangeLow: 226,
            likelyRangeHigh: 233,
            likelyRangeConfidence: 0.5,
            mostLikelyMilestone: nil,
            mostLikelyMilestoneProbability: nil,
            milestoneProbabilities: [],
            closingLine: nil,
            closingOdds: nil,
            actualValue: nil,
            result: nil,
            clv: nil,
            roi: nil,
            reasons: [
                Reason(label: "Game total", detail: "The total still looks inflated versus the pace-adjusted projection, so the under stays playable."),
            ]
        ),
        Recommendation(
            id: "rec_settled_win",
            gameID: "game_mem_mia",
            player: "Ja Morant",
            gameDate: "2026-04-02",
            homeTeam: "Miami Heat",
            awayTeam: "Memphis Grizzlies",
            market: "player_assists",
            selection: "over",
            sportsbookLine: 7.5,
            sportsbookOdds: -110,
            fairLine: 8.2,
            fairOdds: -132,
            edge: 0.041,
            selectedProbability: 0.57,
            marketImpliedProbability: 0.524,
            confidence: "medium",
            status: "production",
            modelVersion: "preview",
            dataTimestamp: "2026-04-02T18:00:00Z",
            publishedLine: 7.5,
            publishedOdds: -110,
            publishedAt: "2026-04-02T18:00:00Z",
            likelyRangeLow: 6,
            likelyRangeHigh: 10,
            likelyRangeConfidence: 0.5,
            mostLikelyMilestone: 8,
            mostLikelyMilestoneProbability: 0.57,
            milestoneProbabilities: [],
            closingLine: 8.0,
            closingOdds: -120,
            actualValue: 9,
            result: "win",
            clv: 0.03,
            roi: 0.91,
            reasons: [
                Reason(label: "Settled", detail: "This preview record exists to drive the performance screen."),
            ]
        ),
    ]

    static let readiness: [MarketReadinessEntry] = [
        MarketReadinessEntry(market: "player_points", status: "production", tier: "A", label: "Production", summary: "Strongest live sample, clearest calibration, and most stable user-facing explanations."),
        MarketReadinessEntry(market: "game_spread", status: "production", tier: "A", label: "Production", summary: "Game-market scoring is ready for discovery, but still benefits from more settled live volume."),
        MarketReadinessEntry(market: "player_rebounds", status: "experimental", tier: "B", label: "Experimental", summary: "The scoring path exists, but readiness remains below the bar for full promotion."),
        MarketReadinessEntry(market: "player_threes", status: "beta", tier: "B", label: "Beta", summary: "Useful in the product, but still accumulating evidence before it becomes a default recommendation class."),
    ]

    static func recommendation(id: String) -> Recommendation? {
        recommendations.first { $0.id == id }
    }

    static func homeResponse(date: String?) -> MobileHomeResponse {
        let availableDates = Array(Set(recommendations.map(\.gameDate))).sorted(by: >)
        let selectedDate = (date.flatMap { requested in
            availableDates.contains(requested) ? requested : nil
        }) ?? availableDates.first ?? ""
        let filtered = recommendations.filter { $0.gameDate == selectedDate && $0.result == nil }
        let groupedGames: [String: [Recommendation]] = Dictionary(grouping: filtered, by: \.gameID)
        let games: [MobileGameSummary] = groupedGames
            .compactMap { gameID, items -> MobileGameSummary? in
                guard let first = items.first else { return nil }
                return MobileGameSummary(
                    id: gameID,
                    gameDate: first.gameDate,
                    commenceTime: gameID == "game_lal_gsw" ? "2026-04-04T22:00:00Z" : "2026-04-04T20:00:00Z",
                    homeTeam: first.homeTeam,
                    awayTeam: first.awayTeam,
                    recommendationCount: items.count,
                    topRecommendation: items.sorted { $0.rankingScore > $1.rankingScore }.first ?? first
                )
            }
            .sorted { $0.topRecommendation.rankingScore > $1.topRecommendation.rankingScore }
        let suggested = Array(filtered.sorted { $0.rankingScore > $1.rankingScore }.prefix(3))
        return MobileHomeResponse(
            selectedDate: selectedDate,
            availableDates: availableDates,
            featuredRecommendations: Array(filtered.sorted { $0.rankingScore > $1.rankingScore }.prefix(6)),
            games: games,
            trendingParlays: [
                MobileParlaySuggestion(
                    id: "preview-two-leg",
                    title: "2-Leg Value",
                    combinedOdds: +245,
                    combinedProbability: 0.42,
                    recommendations: Array(suggested.prefix(2))
                ),
                MobileParlaySuggestion(
                    id: "preview-three-leg",
                    title: "3-Leg Moonshot",
                    combinedOdds: +540,
                    combinedProbability: 0.19,
                    recommendations: suggested
                ),
            ]
        )
    }

    static func gameDetail(for id: String) -> MobileGameDetailResponse {
        let gameRecommendations = recommendations.filter { $0.gameID == id && $0.result == nil }
        let first = gameRecommendations.first ?? recommendations[0]
        return MobileGameDetailResponse(
            id: id,
            gameDate: first.gameDate,
            commenceTime: "2026-04-04T22:00:00Z",
            homeTeam: first.homeTeam,
            awayTeam: first.awayTeam,
            recommendations: gameRecommendations,
            injuries: [
                GameInjury(
                    playerName: "LeBron James",
                    teamAbbrev: "LAL",
                    reportStatus: "Probable",
                    normalizedStatus: "probable",
                    projectedAvailability: "expected to play",
                    rawReason: "Illness",
                    reportedAt: "2026-04-04T17:00:00Z"
                ),
                GameInjury(
                    playerName: "Kevon Looney",
                    teamAbbrev: "GSW",
                    reportStatus: "Out",
                    normalizedStatus: "out",
                    projectedAvailability: "inactive",
                    rawReason: "Hip",
                    reportedAt: "2026-04-04T17:10:00Z"
                ),
            ],
            lineupSummary: [
                TeamLineupSummary(
                    teamAbbrev: "LAL",
                    projectedReturningStarters: 4,
                    projectedReplacements: 1,
                    starters: [
                        LineupStarter(playerName: "LeBron James", projectedPosition: "F", starterProbability: 0.96, injuryStatus: "probable", projectionReason: "recent_starter"),
                        LineupStarter(playerName: "Anthony Davis", projectedPosition: "C", starterProbability: 0.98, injuryStatus: "available", projectionReason: "locked_in_starter"),
                    ]
                ),
                TeamLineupSummary(
                    teamAbbrev: "GSW",
                    projectedReturningStarters: 5,
                    projectedReplacements: 0,
                    starters: [
                        LineupStarter(playerName: "Stephen Curry", projectedPosition: "G", starterProbability: 0.99, injuryStatus: "available", projectionReason: "locked_in_starter"),
                    ]
                ),
            ]
        )
    }

    static let trends = MobileTrendsResponse(
        roi: -0.045,
        clv: 0.03,
        hitRate: 1.0,
        wins: 1,
        losses: 0,
        pushes: 0,
        recentSettlements: recommendations.filter { $0.result != nil },
        chartPoints: [
            TrendChartPoint(gameDate: "2026-04-02", label: "04-02", cumulativeRoi: 0.91),
        ]
    )
}
