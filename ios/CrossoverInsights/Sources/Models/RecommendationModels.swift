import Foundation

struct RecommendationListResponse: Decodable {
    let items: [Recommendation]
}

struct MarketReadinessResponse: Decodable {
    let items: [MarketReadinessEntry]
}

struct MobileHomeResponse: Decodable {
    let selectedDate: String
    let availableDates: [String]
    let featuredRecommendations: [Recommendation]
    let games: [MobileGameSummary]
    let trendingParlays: [MobileParlaySuggestion]

    static let empty = MobileHomeResponse(
        selectedDate: "",
        availableDates: [],
        featuredRecommendations: [],
        games: [],
        trendingParlays: []
    )
}

struct MobileGameSummary: Identifiable, Hashable, Decodable {
    let id: String
    let gameDate: String
    let commenceTime: String?
    let homeTeam: String
    let awayTeam: String
    let recommendationCount: Int
    let topRecommendation: Recommendation

    var date: Date {
        DateFormatter.apiDate.date(from: gameDate) ?? .now
    }
}

struct MobileParlaySuggestion: Identifiable, Hashable, Decodable {
    let id: String
    let title: String
    let combinedOdds: Int
    let combinedProbability: Double
    let recommendations: [Recommendation]
}

struct MobileGameDetailResponse: Identifiable, Hashable, Decodable {
    let id: String
    let gameDate: String
    let commenceTime: String?
    let homeTeam: String
    let awayTeam: String
    let recommendations: [Recommendation]
    let injuries: [GameInjury]
    let lineupSummary: [TeamLineupSummary]
}

struct GameInjury: Hashable, Decodable, Identifiable {
    var id: String { "\(teamAbbrev ?? "UNK")-\(playerName)" }

    let playerName: String
    let teamAbbrev: String?
    let reportStatus: String
    let normalizedStatus: String?
    let projectedAvailability: String?
    let rawReason: String?
    let reportedAt: String?
}

struct TeamLineupSummary: Hashable, Decodable, Identifiable {
    var id: String { teamAbbrev }

    let teamAbbrev: String
    let projectedReturningStarters: Int?
    let projectedReplacements: Int?
    let starters: [LineupStarter]
}

struct LineupStarter: Hashable, Decodable, Identifiable {
    var id: String { playerName }

    let playerName: String
    let projectedPosition: String?
    let starterProbability: Double
    let injuryStatus: String?
    let projectionReason: String
}

struct MobileTrendsResponse: Decodable {
    let roi: Double
    let clv: Double
    let hitRate: Double
    let wins: Int
    let losses: Int
    let pushes: Int
    let recentSettlements: [Recommendation]
    let chartPoints: [TrendChartPoint]

    static let empty = MobileTrendsResponse(
        roi: 0,
        clv: 0,
        hitRate: 0,
        wins: 0,
        losses: 0,
        pushes: 0,
        recentSettlements: [],
        chartPoints: []
    )
}

struct TrendChartPoint: Hashable, Decodable, Identifiable {
    var id: String { gameDate }

    let gameDate: String
    let label: String
    let cumulativeRoi: Double
}

enum RemoteLoadState: Equatable {
    case idle
    case loading
    case loaded
    case empty
    case failed(String)

    var isLoading: Bool {
        if case .loading = self {
            return true
        }
        return false
    }

    var errorMessage: String? {
        if case let .failed(message) = self {
            return message
        }
        return nil
    }
}

struct Recommendation: Identifiable, Hashable, Decodable {
    let id: String
    let gameID: String
    let player: String?
    let gameDate: String
    let homeTeam: String
    let awayTeam: String
    let market: String
    let selection: String
    let sportsbookLine: Double
    let sportsbookOdds: Double?
    let fairLine: Double
    let fairOdds: Int?
    let edge: Double
    let selectedProbability: Double?
    let marketImpliedProbability: Double?
    let confidence: String
    let status: String
    let modelVersion: String
    let dataTimestamp: String
    let publishedLine: Double?
    let publishedOdds: Double?
    let publishedAt: String?
    let likelyRangeLow: Double?
    let likelyRangeHigh: Double?
    let likelyRangeConfidence: Double?
    let mostLikelyMilestone: Double?
    let mostLikelyMilestoneProbability: Double?
    let milestoneProbabilities: [MilestoneProbability]
    let closingLine: Double?
    let closingOdds: Double?
    let actualValue: Double?
    let result: String?
    let clv: Double?
    let roi: Double?
    let reasons: [Reason]

    var displayTitle: String {
        player ?? "\(awayCode) @ \(homeCode)"
    }

    var awayCode: String {
        TeamCode.abbreviation(for: awayTeam)
    }

    var homeCode: String {
        TeamCode.abbreviation(for: homeTeam)
    }

    var marketLabel: String {
        MarketLabel.label(for: market)
    }

    var shortMarketLabel: String {
        if market.hasPrefix("game_") {
            return "\(marketLabel) \(selection.uppercased())"
        }
        return "\(selection.uppercased()) \(formattedSportsbookLine) \(marketLabel)"
    }

    var heroSubtitle: String {
        if market.hasPrefix("game_") {
            return "\(marketLabel.uppercased()) \(selection.uppercased())"
        }
        return "\(selection.uppercased()) \(formattedSportsbookLine) \(marketLabel.uppercased())"
    }

    var formattedSportsbookLine: String {
        sportsbookLine.formattedCompact
    }

    var edgeText: String {
        edge.formattedPercent(digits: 1)
    }

    var probabilityText: String {
        (selectedProbability ?? 0).formattedPercent(digits: 1)
    }

    var marketImpliedText: String {
        (marketImpliedProbability ?? 0).formattedPercent(digits: 1)
    }

    var confidenceText: String {
        confidence.uppercased()
    }

    var statusText: String {
        status.uppercased()
    }

    var projectionText: String {
        if market.hasPrefix("game_") {
            return fairLine.formattedCompact
        }
        return "\(fairLine.formattedCompact) \(marketLabel.uppercased())"
    }

    var milestoneCards: [MilestoneProbability] {
        if !milestoneProbabilities.isEmpty {
            return Array(milestoneProbabilities.prefix(4))
        }
        let step = MarketLabel.milestoneStep(for: market)
        let base = sportsbookLine
        return (0..<4).map { index in
            MilestoneProbability(
                threshold: max(step, base + (Double(index) - 1) * step),
                probability: max(0.18, min(0.88, (selectedProbability ?? 0.55) - Double(index) * 0.16)),
                fairOdds: nil,
                lineEquivalent: nil
            )
        }
    }

    var likelyRangeText: String {
        let low = (likelyRangeLow ?? fairLine - 3).formattedCompact0
        let high = (likelyRangeHigh ?? fairLine + 3).formattedCompact0
        return "\(low) - \(high)"
    }

    var date: Date {
        DateFormatter.apiDate.date(from: gameDate) ?? .now
    }
}

struct Reason: Hashable, Decodable {
    let label: String
    let detail: String
}

struct MilestoneProbability: Hashable, Decodable {
    let threshold: Double
    let probability: Double
    let fairOdds: Int?
    let lineEquivalent: Double?

    var badgeText: String {
        if probability >= 0.8 { return "LOCKED" }
        if probability >= 0.6 { return "TARGET" }
        if probability >= 0.35 { return "STRETCH" }
        return "LOTTO"
    }
}

struct MarketReadinessEntry: Hashable, Decodable, Identifiable {
    var id: String { market }
    let market: String
    let status: String
    let tier: String
    let label: String
    let summary: String
}

struct GameSlate: Identifiable, Hashable {
    let id: String
    let gameDate: String
    let homeTeam: String
    let awayTeam: String
    let recommendations: [Recommendation]

    var topRecommendation: Recommendation {
        recommendations.sorted { $0.rankingScore > $1.rankingScore }.first ?? recommendations[0]
    }

    var date: Date {
        DateFormatter.apiDate.date(from: gameDate) ?? .now
    }
}

enum AppTab: Hashable {
    case home
    case games
    case parlay
    case trends
}

enum MarketFilter: String, CaseIterable, Identifiable {
    case all
    case playerPoints = "player_points"
    case playerRebounds = "player_rebounds"
    case playerAssists = "player_assists"
    case playerThrees = "player_threes"
    case gameSpread = "game_spread"
    case gameTotal = "game_total"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .all: return "All"
        case .playerPoints: return "Points"
        case .playerRebounds: return "Rebounds"
        case .playerAssists: return "Assists"
        case .playerThrees: return "Threes"
        case .gameSpread: return "Spreads"
        case .gameTotal: return "Totals"
        }
    }
}

enum ConfidenceFilter: String, CaseIterable, Identifiable {
    case all
    case low
    case medium
    case high

    var id: String { rawValue }

    var title: String {
        switch self {
        case .all: return "All"
        case .low: return "Low"
        case .medium: return "Med"
        case .high: return "High"
        }
    }
}

extension Recommendation {
    var rankingScore: Double {
        (selectedProbability ?? 0) + edge * 100
    }
}

enum TeamCode {
    static func abbreviation(for team: String) -> String {
        let map: [String: String] = [
            "Atlanta Hawks": "ATL",
            "Boston Celtics": "BOS",
            "Brooklyn Nets": "BKN",
            "Charlotte Hornets": "CHA",
            "Chicago Bulls": "CHI",
            "Cleveland Cavaliers": "CLE",
            "Dallas Mavericks": "DAL",
            "Denver Nuggets": "DEN",
            "Detroit Pistons": "DET",
            "Golden State Warriors": "GSW",
            "Houston Rockets": "HOU",
            "Indiana Pacers": "IND",
            "Los Angeles Clippers": "LAC",
            "Los Angeles Lakers": "LAL",
            "Memphis Grizzlies": "MEM",
            "Miami Heat": "MIA",
            "Milwaukee Bucks": "MIL",
            "Minnesota Timberwolves": "MIN",
            "New Orleans Pelicans": "NOP",
            "New York Knicks": "NYK",
            "Oklahoma City Thunder": "OKC",
            "Orlando Magic": "ORL",
            "Philadelphia 76ers": "PHI",
            "Phoenix Suns": "PHX",
            "Portland Trail Blazers": "POR",
            "Sacramento Kings": "SAC",
            "San Antonio Spurs": "SAS",
            "Toronto Raptors": "TOR",
            "Utah Jazz": "UTA",
            "Washington Wizards": "WAS",
        ]
        return map[team] ?? String(team.prefix(3)).uppercased()
    }
}

enum MarketLabel {
    static func label(for market: String) -> String {
        switch market {
        case "player_points": return "Points"
        case "player_rebounds": return "Rebounds"
        case "player_assists": return "Assists"
        case "player_threes": return "Threes"
        case "player_points_rebounds": return "Pts + Reb"
        case "player_points_assists": return "Pts + Ast"
        case "player_rebounds_assists": return "Reb + Ast"
        case "player_points_rebounds_assists": return "PRA"
        case "game_moneyline": return "Moneyline"
        case "game_spread": return "Spread"
        case "game_total": return "Total"
        default: return market.replacingOccurrences(of: "_", with: " ").capitalized
        }
    }

    static func milestoneStep(for market: String) -> Double {
        if market.contains("threes") { return 1 }
        if market.contains("rebounds") || market.contains("assists") { return 2 }
        return 5
    }
}

extension Double {
    var formattedCompact: String {
        formatted(.number.precision(.fractionLength(1))).replacingOccurrences(of: ".0", with: "")
    }

    var formattedCompact0: String {
        formatted(.number.precision(.fractionLength(0)))
    }

    func formattedPercent(digits: Int) -> String {
        let value = self * 100
        return value.formatted(.number.precision(.fractionLength(digits))) + "%"
    }

    var americanOddsText: String {
        let rounded = Int(self.rounded())
        return rounded > 0 ? "+\(rounded)" : "\(rounded)"
    }
}

extension Int {
    var americanOddsText: String {
        self > 0 ? "+\(self)" : "\(self)"
    }
}

extension DateFormatter {
    static let apiDate: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter
    }()

    static let shortDisplay: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "E, MMM d"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter
    }()

    static let longDisplay: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d"
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter
    }()

    static let timestampDisplay: DateFormatter = {
        let formatter = DateFormatter()
        formatter.dateStyle = .medium
        formatter.timeStyle = .short
        formatter.locale = Locale(identifier: "en_US_POSIX")
        return formatter
    }()
}
