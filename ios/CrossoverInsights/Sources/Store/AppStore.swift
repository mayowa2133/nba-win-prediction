import Foundation
import SwiftUI

@MainActor
final class AppStore: ObservableObject {
    @Published var selectedTab: AppTab = .home
    @Published var selectedDate: String = ""
    @Published var marketFilter: MarketFilter = .all
    @Published var confidenceFilter: ConfidenceFilter = .all

    @Published private(set) var home = MobileHomeResponse.empty
    @Published private(set) var trends = MobileTrendsResponse.empty
    @Published private(set) var readiness: [MarketReadinessEntry] = []
    @Published private(set) var homeState: RemoteLoadState = .idle
    @Published private(set) var trendsState: RemoteLoadState = .idle
    @Published private(set) var readinessState: RemoteLoadState = .idle
    @Published private(set) var gameDetails: [String: MobileGameDetailResponse] = [:]
    @Published private(set) var gameLoadStates: [String: RemoteLoadState] = [:]

    @Published private(set) var parlayIDs: Set<String>
    @Published private(set) var savedIDs: Set<String>
    @Published private(set) var playedIDs: Set<String>

    private let apiClient: APIClient
    private let defaults: UserDefaults
    private var recommendationCache: [String: Recommendation] = [:]
    private var didLoad = false
    private var homeRequestToken = 0

    init(apiClient: APIClient = APIClient(), defaults: UserDefaults = .standard) {
        self.apiClient = apiClient
        self.defaults = defaults
        self.parlayIDs = Set(defaults.stringArray(forKey: StorageKey.parlay.rawValue) ?? [])
        self.savedIDs = Set(defaults.stringArray(forKey: StorageKey.saved.rawValue) ?? [])
        self.playedIDs = Set(defaults.stringArray(forKey: StorageKey.played.rawValue) ?? [])
    }

    func loadIfNeeded() async {
        guard !didLoad else { return }
        didLoad = true
        await reload()
    }

    func reload() async {
        async let homeReload: Void = reloadHome(date: selectedDate.isEmpty ? nil : selectedDate)
        async let readinessReload: Void = reloadReadiness()
        async let trendsReload: Void = reloadTrends()
        _ = await (homeReload, readinessReload, trendsReload)
        await hydratePersistedSelections()
        pruneSelections()
    }

    func selectDate(_ date: String) async {
        selectedDate = date
        await reloadHome(date: date)
        await hydratePersistedSelections()
        pruneSelections()
    }

    func updateMarketFilter(_ filter: MarketFilter) async {
        marketFilter = filter
        await reloadHome(date: selectedDate.isEmpty ? nil : selectedDate)
        await hydratePersistedSelections()
        pruneSelections()
    }

    func updateConfidenceFilter(_ filter: ConfidenceFilter) async {
        confidenceFilter = filter
        await reloadHome(date: selectedDate.isEmpty ? nil : selectedDate)
        await hydratePersistedSelections()
        pruneSelections()
    }

    func reloadGameDetail(_ gameID: String) async {
        let currentState = gameLoadStates[gameID] ?? .idle
        if currentState.isLoading {
            return
        }
        gameLoadStates[gameID] = .loading
        do {
            let detail = try await apiClient.fetchGameDetail(id: gameID)
            gameDetails[gameID] = detail
            ingest(detail.recommendations)
            gameLoadStates[gameID] = detail.recommendations.isEmpty ? .empty : .loaded
        } catch {
            gameLoadStates[gameID] = .failed(message(for: error))
        }
        await hydratePersistedSelections()
        pruneSelections()
    }

    func reloadTrendsScreen() async {
        await reloadTrends()
    }

    var availableDates: [String] {
        home.availableDates
    }

    var edgeRecommendations: [Recommendation] {
        home.featuredRecommendations
    }

    var groupedGames: [MobileGameSummary] {
        home.games
    }

    var allGames: [MobileGameSummary] {
        home.games
    }

    var parlayRecommendations: [Recommendation] {
        parlayIDs.compactMap { recommendationCache[$0] }.sorted { $0.rankingScore > $1.rankingScore }
    }

    var parlaySuggestions: [MobileParlaySuggestion] {
        home.trendingParlays
    }

    var homeTrendingParlays: [MobileParlaySuggestion] {
        home.trendingParlays
    }

    var isDemoModeEnabled: Bool {
        apiClient.configuration.demoModeEnabled
    }

    var apiBaseURL: String? {
        apiClient.configuration.baseURL?.absoluteString
    }

    func addToParlay(_ recommendation: Recommendation) {
        ingest(recommendation)
        guard !parlayIDs.contains(recommendation.id) else { return }
        parlayIDs.insert(recommendation.id)
        persist(parlayIDs, key: .parlay)
    }

    func toggleParlay(_ recommendation: Recommendation) {
        ingest(recommendation)
        toggle(recommendation.id, set: &parlayIDs, key: .parlay)
    }

    func toggleSaved(_ recommendation: Recommendation) {
        ingest(recommendation)
        toggle(recommendation.id, set: &savedIDs, key: .saved)
    }

    func togglePlayed(_ recommendation: Recommendation) {
        ingest(recommendation)
        toggle(recommendation.id, set: &playedIDs, key: .played)
    }

    func isInParlay(_ recommendation: Recommendation) -> Bool {
        parlayIDs.contains(recommendation.id)
    }

    func isSaved(_ recommendation: Recommendation) -> Bool {
        savedIDs.contains(recommendation.id)
    }

    func isPlayed(_ recommendation: Recommendation) -> Bool {
        playedIDs.contains(recommendation.id)
    }

    func game(for id: String) -> MobileGameSummary? {
        home.games.first { $0.id == id }
    }

    func gameDetail(for id: String) -> MobileGameDetailResponse? {
        gameDetails[id]
    }

    func gameLoadState(for id: String) -> RemoteLoadState {
        gameLoadStates[id] ?? .idle
    }

    func recommendation(for id: String) -> Recommendation? {
        recommendationCache[id]
    }

    func combinedParlayOdds(for picks: [Recommendation]) -> Int {
        guard !picks.isEmpty else { return 0 }
        let decimal = picks.reduce(1.0) { partialResult, recommendation in
            partialResult * decimalOdds(from: recommendation.sportsbookOdds ?? 0)
        }
        return americanOdds(from: decimal)
    }

    func combinedProbability(for picks: [Recommendation]) -> Double {
        picks.reduce(1.0) { partialResult, recommendation in
            partialResult * (recommendation.selectedProbability ?? 0.5)
        }
    }

    func payout(for odds: Int, stake: Double) -> Double {
        stake * decimalOdds(from: Double(odds))
    }

    func hasCorrelationWarning(for picks: [Recommendation]) -> Bool {
        let gameIDs = picks.map(\.gameID)
        return Set(gameIDs).count != gameIDs.count
    }

    private var marketQueryValue: String? {
        marketFilter == .all ? nil : marketFilter.rawValue
    }

    private var confidenceQueryValue: String? {
        confidenceFilter == .all ? nil : confidenceFilter.rawValue
    }

    private func reloadHome(date: String?) async {
        homeRequestToken += 1
        let requestToken = homeRequestToken
        homeState = .loading
        do {
            let response = try await apiClient.fetchHome(
                date: date,
                market: marketQueryValue,
                confidence: confidenceQueryValue
            )
            guard requestToken == homeRequestToken else { return }
            home = response
            selectedDate = response.selectedDate
            ingest(response.featuredRecommendations)
            ingest(response.games.map(\.topRecommendation))
            response.trendingParlays.forEach { ingest($0.recommendations) }
            homeState = (response.featuredRecommendations.isEmpty && response.games.isEmpty) ? .empty : .loaded
        } catch {
            guard requestToken == homeRequestToken else { return }
            home = MobileHomeResponse.empty
            homeState = .failed(message(for: error))
        }
    }

    private func reloadReadiness() async {
        readinessState = .loading
        do {
            readiness = try await apiClient.fetchReadiness()
            readinessState = readiness.isEmpty ? .empty : .loaded
        } catch {
            readiness = []
            readinessState = .failed(message(for: error))
        }
    }

    private func reloadTrends() async {
        trendsState = .loading
        do {
            trends = try await apiClient.fetchTrends()
            ingest(trends.recentSettlements)
            trendsState = (trends.recentSettlements.isEmpty && trends.chartPoints.isEmpty) ? .empty : .loaded
        } catch {
            trends = .empty
            trendsState = .failed(message(for: error))
        }
    }

    private func hydratePersistedSelections() async {
        let requestedIDs = parlayIDs.union(savedIDs).union(playedIDs)
        let missingIDs = requestedIDs.subtracting(Set(recommendationCache.keys))
        guard !missingIDs.isEmpty else { return }
        for recommendationID in missingIDs {
            do {
                let recommendation = try await apiClient.fetchRecommendation(id: recommendationID)
                ingest(recommendation)
            } catch {
                continue
            }
        }
    }

    private func ingest(_ items: [Recommendation]) {
        for item in items {
            recommendationCache[item.id] = item
        }
    }

    private func ingest(_ item: Recommendation) {
        recommendationCache[item.id] = item
    }

    private func pruneSelections() {
        let valid = Set(recommendationCache.keys)
        parlayIDs = parlayIDs.intersection(valid)
        savedIDs = savedIDs.intersection(valid)
        playedIDs = playedIDs.intersection(valid)
        persist(parlayIDs, key: .parlay)
        persist(savedIDs, key: .saved)
        persist(playedIDs, key: .played)
    }

    private func toggle(_ id: String, set: inout Set<String>, key: StorageKey) {
        if set.contains(id) {
            set.remove(id)
        } else {
            set.insert(id)
        }
        persist(set, key: key)
    }

    private func persist(_ set: Set<String>, key: StorageKey) {
        defaults.set(Array(set), forKey: key.rawValue)
    }

    private func decimalOdds(from american: Double) -> Double {
        guard american != 0 else { return 1 }
        return american > 0 ? 1 + american / 100 : 1 + 100 / abs(american)
    }

    private func americanOdds(from decimal: Double) -> Int {
        guard decimal > 1 else { return 0 }
        return decimal >= 2 ? Int(((decimal - 1) * 100).rounded()) : Int((-100 / (decimal - 1)).rounded())
    }

    private func message(for error: Error) -> String {
        if let localized = error as? LocalizedError, let description = localized.errorDescription {
            return description
        }
        return error.localizedDescription
    }

    private enum StorageKey: String {
        case parlay = "crossover.parlay"
        case saved = "crossover.saved"
        case played = "crossover.played"
    }
}
