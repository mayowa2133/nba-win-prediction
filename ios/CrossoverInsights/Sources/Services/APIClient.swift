import Foundation

enum APIClientError: LocalizedError {
    case missingBaseURL
    case invalidBaseURL(String)
    case invalidResponse
    case httpStatus(Int)

    var errorDescription: String? {
        switch self {
        case .missingBaseURL:
            return "Set CROSSOVER_API_BASE_URL to a LAN or staging URL before running on a physical iPhone."
        case let .invalidBaseURL(value):
            return "CROSSOVER_API_BASE_URL is invalid: \(value)"
        case .invalidResponse:
            return "The API returned an invalid response."
        case let .httpStatus(statusCode):
            return "The API returned HTTP \(statusCode)."
        }
    }
}

struct AppConfiguration {
    let baseURL: URL?
    let demoModeEnabled: Bool
    let configurationError: APIClientError?

    init(bundle: Bundle = .main, processInfo: ProcessInfo = .processInfo) {
        #if DEBUG
        demoModeEnabled = processInfo.environment["CROSSOVER_DEMO_MODE"] == "1"
        #else
        demoModeEnabled = false
        #endif

        if let env = processInfo.environment["CROSSOVER_API_BASE_URL"]?.trimmingCharacters(in: .whitespacesAndNewlines),
           !env.isEmpty {
            if let url = URL(string: env) {
                baseURL = url
                configurationError = nil
            } else {
                baseURL = nil
                configurationError = .invalidBaseURL(env)
            }
            return
        }

        if let infoValue = bundle.object(forInfoDictionaryKey: "CrossoverAPIBaseURL") as? String {
            let trimmed = infoValue.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty {
                if let url = URL(string: trimmed) {
                    baseURL = url
                    configurationError = nil
                } else {
                    baseURL = nil
                    configurationError = .invalidBaseURL(trimmed)
                }
                return
            }
        }

        #if targetEnvironment(simulator)
        baseURL = URL(string: "http://127.0.0.1:8000")
        configurationError = nil
        #else
        baseURL = nil
        configurationError = .missingBaseURL
        #endif
    }
}

struct APIClient {
    let configuration: AppConfiguration

    init(configuration: AppConfiguration = AppConfiguration()) {
        self.configuration = configuration
    }

    func fetchHome(date: String? = nil, market: String? = nil, confidence: String? = nil) async throws -> MobileHomeResponse {
        if configuration.demoModeEnabled {
            return PreviewFixtures.homeResponse(date: date)
        }
        var queryItems: [URLQueryItem] = []
        if let date, !date.isEmpty {
            queryItems.append(URLQueryItem(name: "date", value: date))
        }
        if let market, !market.isEmpty {
            queryItems.append(URLQueryItem(name: "market", value: market))
        }
        if let confidence, !confidence.isEmpty {
            queryItems.append(URLQueryItem(name: "confidence", value: confidence))
        }
        return try await decode(path: "/v1/mobile/home", queryItems: queryItems)
    }

    func fetchGameDetail(id: String) async throws -> MobileGameDetailResponse {
        if configuration.demoModeEnabled {
            return PreviewFixtures.gameDetail(for: id)
        }
        return try await decode(path: "/v1/mobile/games/\(id)")
    }

    func fetchTrends() async throws -> MobileTrendsResponse {
        if configuration.demoModeEnabled {
            return PreviewFixtures.trends
        }
        return try await decode(path: "/v1/mobile/trends")
    }

    func fetchReadiness() async throws -> [MarketReadinessEntry] {
        if configuration.demoModeEnabled {
            return PreviewFixtures.readiness
        }
        let response: MarketReadinessResponse = try await decode(path: "/v1/markets/readiness")
        return response.items
    }

    func fetchRecommendation(id: String) async throws -> Recommendation {
        if configuration.demoModeEnabled {
            if let item = PreviewFixtures.recommendation(id: id) {
                return item
            }
            throw APIClientError.invalidResponse
        }
        return try await decode(path: "/v1/recommendations/\(id)")
    }

    private func decode<T: Decodable>(path: String, queryItems: [URLQueryItem] = []) async throws -> T {
        if let configurationError = configuration.configurationError {
            throw configurationError
        }
        guard let baseURL = configuration.baseURL else {
            throw APIClientError.missingBaseURL
        }
        guard var components = URLComponents(url: baseURL, resolvingAgainstBaseURL: false) else {
            throw APIClientError.invalidResponse
        }
        let normalizedBasePath = components.path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        let normalizedRequestPath = path.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        components.path = "/" + [normalizedBasePath, normalizedRequestPath]
            .filter { !$0.isEmpty }
            .joined(separator: "/")
        if !queryItems.isEmpty {
            components.queryItems = queryItems
        }
        guard let url = components.url else {
            throw APIClientError.invalidResponse
        }

        let (data, response) = try await URLSession.shared.data(from: url)
        guard let http = response as? HTTPURLResponse else {
            throw APIClientError.invalidResponse
        }
        guard 200..<300 ~= http.statusCode else {
            throw APIClientError.httpStatus(http.statusCode)
        }
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        decoder.dateDecodingStrategy = .iso8601
        return try decoder.decode(T.self, from: data)
    }
}
