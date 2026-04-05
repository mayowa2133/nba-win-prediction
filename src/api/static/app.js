const TEAM_ABBREVIATIONS = {
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
  "LA Clippers": "LAC",
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
};

const MARKET_LABELS = {
  player_points: "Points",
  player_rebounds: "Rebounds",
  player_assists: "Assists",
  player_threes: "Threes",
  player_points_rebounds: "Pts + Reb",
  player_points_assists: "Pts + Ast",
  player_rebounds_assists: "Reb + Ast",
  player_points_rebounds_assists: "PRA",
  game_moneyline: "Moneyline",
  game_spread: "Spread",
  game_total: "Total",
};

const STORAGE_KEYS = {
  queue: "crossover-insights-queue",
  saved: "crossover-insights-saved",
  played: "crossover-insights-played",
};

const FALLBACK_RECOMMENDATIONS = [
  {
    id: "rec_lebron_points",
    game_id: "game_lal_gsw",
    player: "LeBron James",
    game_date: "2026-04-04",
    home_team: "Golden State Warriors",
    away_team: "Los Angeles Lakers",
    market: "player_points",
    selection: "over",
    sportsbook_line: 25.5,
    sportsbook_odds: -110,
    fair_line: 28.4,
    fair_odds: -180,
    edge: 0.092,
    selected_probability: 0.642,
    market_implied_probability: 0.524,
    confidence: "high",
    status: "production",
    model_version: "fallback",
    data_timestamp: "2026-04-04T19:30:00Z",
    likely_range_low: 25,
    likely_range_high: 31,
    likely_range_confidence: 0.5,
    most_likely_milestone: 25,
    most_likely_milestone_probability: 0.68,
    milestone_probabilities: [
      { threshold: 20, probability: 0.88, fair_odds: -733, line_equivalent: 19.5 },
      { threshold: 25, probability: 0.68, fair_odds: -213, line_equivalent: 24.5 },
      { threshold: 30, probability: 0.42, fair_odds: 138, line_equivalent: 29.5 },
      { threshold: 35, probability: 0.18, fair_odds: 456, line_equivalent: 34.5 },
    ],
    reasons: [
      {
        label: "Model vs line",
        detail: "Model projects 28.4 against 25.5, so the current points number still trails the fair price.",
      },
      {
        label: "Likely range",
        detail: "The model central range clusters between 25 and 31 points with room for another gear late.",
      },
      {
        label: "Market status",
        detail: "Player points is the strongest production market in the stack and carries the best readiness label.",
      },
    ],
  },
  {
    id: "rec_ad_rebounds",
    game_id: "game_lal_gsw",
    player: "Anthony Davis",
    game_date: "2026-04-04",
    home_team: "Golden State Warriors",
    away_team: "Los Angeles Lakers",
    market: "player_rebounds",
    selection: "over",
    sportsbook_line: 12.5,
    sportsbook_odds: -115,
    fair_line: 13.8,
    fair_odds: -152,
    edge: 0.071,
    selected_probability: 0.618,
    market_implied_probability: 0.535,
    confidence: "high",
    status: "experimental",
    model_version: "fallback",
    data_timestamp: "2026-04-04T19:30:00Z",
    likely_range_low: 11,
    likely_range_high: 15,
    likely_range_confidence: 0.5,
    most_likely_milestone: 12,
    most_likely_milestone_probability: 0.73,
    milestone_probabilities: [
      { threshold: 10, probability: 0.89, fair_odds: -809, line_equivalent: 9.5 },
      { threshold: 12, probability: 0.73, fair_odds: -270, line_equivalent: 11.5 },
      { threshold: 14, probability: 0.41, fair_odds: 144, line_equivalent: 13.5 },
      { threshold: 16, probability: 0.19, fair_odds: 426, line_equivalent: 15.5 },
    ],
    reasons: [
      {
        label: "Lineup context",
        detail: "The glass rate holds when the Lakers lean big, but the market is still marked experimental.",
      },
    ],
  },
  {
    id: "rec_curry_threes",
    game_id: "game_lal_gsw",
    player: "Stephen Curry",
    game_date: "2026-04-04",
    home_team: "Golden State Warriors",
    away_team: "Los Angeles Lakers",
    market: "player_threes",
    selection: "over",
    sportsbook_line: 4.5,
    sportsbook_odds: +105,
    fair_line: 5.1,
    fair_odds: -112,
    edge: 0.052,
    selected_probability: 0.587,
    market_implied_probability: 0.488,
    confidence: "medium",
    status: "experimental",
    model_version: "fallback",
    data_timestamp: "2026-04-04T19:30:00Z",
    likely_range_low: 3,
    likely_range_high: 6,
    likely_range_confidence: 0.5,
    most_likely_milestone: 5,
    most_likely_milestone_probability: 0.58,
    milestone_probabilities: [
      { threshold: 3, probability: 0.86, fair_odds: -614, line_equivalent: 2.5 },
      { threshold: 4, probability: 0.71, fair_odds: -245, line_equivalent: 3.5 },
      { threshold: 5, probability: 0.58, fair_odds: -138, line_equivalent: 4.5 },
      { threshold: 6, probability: 0.33, fair_odds: 203, line_equivalent: 5.5 },
    ],
    reasons: [
      {
        label: "Volume projection",
        detail: "The shot diet leans perimeter-heavy in this pace band, which keeps the three-point ladder alive.",
      },
    ],
  },
  {
    id: "rec_bos_spread",
    game_id: "game_bos_phi",
    player: null,
    game_date: "2026-04-04",
    home_team: "Philadelphia 76ers",
    away_team: "Boston Celtics",
    market: "game_spread",
    selection: "away",
    sportsbook_line: -4.5,
    sportsbook_odds: -110,
    fair_line: -6.1,
    fair_odds: -150,
    edge: 0.066,
    selected_probability: 0.611,
    market_implied_probability: 0.524,
    confidence: "medium",
    status: "production",
    model_version: "fallback",
    data_timestamp: "2026-04-04T19:30:00Z",
    likely_range_low: -8,
    likely_range_high: -2,
    likely_range_confidence: 0.5,
    most_likely_milestone: null,
    most_likely_milestone_probability: null,
    milestone_probabilities: [],
    reasons: [
      {
        label: "Win probability",
        detail: "The spread model still prices Boston stronger than the current market number.",
      },
    ],
  },
  {
    id: "rec_tatum_points",
    game_id: "game_bos_phi",
    player: "Jayson Tatum",
    game_date: "2026-04-04",
    home_team: "Philadelphia 76ers",
    away_team: "Boston Celtics",
    market: "player_points",
    selection: "over",
    sportsbook_line: 28.5,
    sportsbook_odds: -108,
    fair_line: 30.1,
    fair_odds: -148,
    edge: 0.058,
    selected_probability: 0.602,
    market_implied_probability: 0.519,
    confidence: "medium",
    status: "production",
    model_version: "fallback",
    data_timestamp: "2026-04-04T19:30:00Z",
    likely_range_low: 26,
    likely_range_high: 33,
    likely_range_confidence: 0.5,
    most_likely_milestone: 30,
    most_likely_milestone_probability: 0.52,
    milestone_probabilities: [
      { threshold: 20, probability: 0.87, fair_odds: -669, line_equivalent: 19.5 },
      { threshold: 25, probability: 0.69, fair_odds: -223, line_equivalent: 24.5 },
      { threshold: 30, probability: 0.52, fair_odds: -108, line_equivalent: 29.5 },
      { threshold: 35, probability: 0.27, fair_odds: 270, line_equivalent: 34.5 },
    ],
    reasons: [
      {
        label: "Model vs line",
        detail: "The projection still clears the market after accounting for the stronger road context.",
      },
    ],
  },
  {
    id: "rec_okc_total",
    game_id: "game_okc_den",
    player: null,
    game_date: "2026-04-04",
    home_team: "Denver Nuggets",
    away_team: "Oklahoma City Thunder",
    market: "game_total",
    selection: "under",
    sportsbook_line: 234.5,
    sportsbook_odds: -108,
    fair_line: 229.8,
    fair_odds: -134,
    edge: 0.049,
    selected_probability: 0.573,
    market_implied_probability: 0.519,
    confidence: "medium",
    status: "production",
    model_version: "fallback",
    data_timestamp: "2026-04-04T19:30:00Z",
    likely_range_low: 226,
    likely_range_high: 233,
    likely_range_confidence: 0.5,
    most_likely_milestone: null,
    most_likely_milestone_probability: null,
    milestone_probabilities: [],
    reasons: [
      {
        label: "Game total",
        detail: "The total still looks inflated versus the pace-adjusted projection, so the under stays playable.",
      },
    ],
  },
];

const FALLBACK_READINESS = [
  {
    market: "player_points",
    status: "production",
    tier: "A",
    label: "Production",
    summary: "Strongest live sample, clearest calibration, and most stable user-facing explanations.",
  },
  {
    market: "game_spread",
    status: "production",
    tier: "A",
    label: "Production",
    summary: "Game-market scoring is ready for discovery, but still benefits from more settled live volume.",
  },
  {
    market: "player_rebounds",
    status: "experimental",
    tier: "B",
    label: "Experimental",
    summary: "The scoring path exists, but readiness remains below the bar for full promotion.",
  },
  {
    market: "player_threes",
    status: "beta",
    tier: "B",
    label: "Beta",
    summary: "Useful in the product, but still accumulating evidence before it becomes a default recommendation class.",
  },
];

const FALLBACK_SETTLEMENTS = [
  { title: "Lakers @ Celtics - Over 234.5", subtitle: "Game Line - 01.12.2026", units: "+2.40u", result: "win", odds: "-110" },
  { title: "Ja Morant - U 8.5 Assists", subtitle: "Player Prop - 01.12.2026", units: "-1.00u", result: "loss", odds: "+105" },
  { title: "Bucks ML - 2-Leg Parlay", subtitle: "Parlay - 01.11.2026", units: "+1.85u", result: "win", odds: "+185" },
  { title: "Knicks -4.0", subtitle: "Game Line - 01.11.2026", units: "0.00u", result: "push", odds: "-110" },
];

const state = {
  recommendations: [],
  readiness: [],
  selectedDate: null,
  marketFilter: "all",
  confidenceFilter: "all",
  queueIds: readStoredIds(STORAGE_KEYS.queue),
  savedIds: readStoredIds(STORAGE_KEYS.saved),
  playedIds: readStoredIds(STORAGE_KEYS.played),
};

document.addEventListener("DOMContentLoaded", async () => {
  bindGlobalEvents();
  await loadData();
  ensureDefaultRoute();
  render();
});

window.addEventListener("hashchange", render);

function bindGlobalEvents() {
  document.getElementById("back-button").addEventListener("click", () => {
    navigateTo("home");
  });

  document.querySelectorAll("[data-route]").forEach((button) => {
    button.addEventListener("click", () => navigateTo(button.dataset.route));
  });

  document.getElementById("carousel-prev").addEventListener("click", () => {
    document.getElementById("edge-carousel").scrollBy({ left: -320, behavior: "smooth" });
  });

  document.getElementById("carousel-next").addEventListener("click", () => {
    document.getElementById("edge-carousel").scrollBy({ left: 320, behavior: "smooth" });
  });

  document.getElementById("pick-add-to-parlay").addEventListener("click", () => {
    const recommendation = currentPick();
    if (!recommendation) {
      return;
    }
    toggleStoredId(state.queueIds, recommendation.id);
    persistIds(STORAGE_KEYS.queue, state.queueIds);
    render();
  });

  document.getElementById("pick-save-button").addEventListener("click", () => {
    const recommendation = currentPick();
    if (!recommendation) {
      return;
    }
    toggleStoredId(state.savedIds, recommendation.id);
    persistIds(STORAGE_KEYS.saved, state.savedIds);
    render();
  });

  document.getElementById("pick-played-button").addEventListener("click", () => {
    const recommendation = currentPick();
    if (!recommendation) {
      return;
    }
    toggleStoredId(state.playedIds, recommendation.id);
    persistIds(STORAGE_KEYS.played, state.playedIds);
    render();
  });
}

async function loadData() {
  const [recommendationPayload, readinessPayload] = await Promise.all([
    fetchJson("/v1/recommendations"),
    fetchJson("/v1/markets/readiness"),
  ]);

  state.recommendations = normalizeRecommendations(
    recommendationPayload && Array.isArray(recommendationPayload.items)
      ? recommendationPayload.items
      : FALLBACK_RECOMMENDATIONS,
  );
  state.readiness =
    readinessPayload && Array.isArray(readinessPayload.items) ? readinessPayload.items : FALLBACK_READINESS;

  const dates = uniqueDates();
  state.selectedDate = dates[0] || isoToday();
}

async function fetchJson(url) {
  try {
    const response = await fetch(url, { headers: { Accept: "application/json" } });
    if (!response.ok) {
      throw new Error(`Request failed: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    return null;
  }
}

function normalizeRecommendations(items) {
  return [...items]
    .filter((item) => item && item.id)
    .sort((left, right) => {
      const dateDelta = String(right.game_date || "").localeCompare(String(left.game_date || ""));
      if (dateDelta !== 0) {
        return dateDelta;
      }
      return scoreRecommendation(right) - scoreRecommendation(left);
    });
}

function ensureDefaultRoute() {
  if (!window.location.hash) {
    navigateTo("home", { replace: true });
  }
}

function parseRoute() {
  const hash = window.location.hash.replace(/^#\/?/, "");
  if (!hash) {
    return { view: "home" };
  }
  const [view, id] = hash.split("/");
  if (view === "pick") {
    return { view: "pick", id };
  }
  if (view === "game") {
    return { view: "game", id };
  }
  if (view === "parlay") {
    return { view: "parlay" };
  }
  if (view === "trends") {
    return { view: "trends" };
  }
  return { view: "home" };
}

function navigateTo(view, options = {}) {
  let destination = `#/${view}`;
  if (view === "game") {
    const firstGame = groupedGames(state.recommendations)[0];
    destination = firstGame ? `#/game/${firstGame.id}` : "#/game";
  }
  if (options.replace) {
    window.history.replaceState(null, "", destination);
    render();
    return;
  }
  window.location.hash = destination;
}

function navigateToPick(id) {
  window.location.hash = `#/pick/${id}`;
}

function navigateToGame(id) {
  window.location.hash = `#/game/${id}`;
}

function render() {
  const route = parseRoute();
  const currentView = normalizeView(route.view);

  document.querySelectorAll(".app-view").forEach((section) => {
    section.classList.toggle("active", section.id === `view-${currentView}`);
  });

  updateHeader(currentView);
  updateNav(currentView);
  updateQueueHeader();

  renderHome();
  renderPickDetail(route.id);
  renderGameDetail(route.id);
  renderParlay();
  renderTrends();
}

function normalizeView(view) {
  if (view === "pick") {
    return "pick";
  }
  if (view === "game") {
    return "game";
  }
  if (view === "parlay") {
    return "parlay";
  }
  if (view === "trends") {
    return "trends";
  }
  return "home";
}

function updateHeader(view) {
  const backButton = document.getElementById("back-button");
  const headerCopy = document.getElementById("header-copy");
  const showBack = view === "pick" || view === "game";
  backButton.classList.toggle("hidden", !showBack);

  const copyMap = {
    home: "NBA decision support",
    pick: "Pick detail",
    game: "Game detail",
    parlay: "Parlay builder",
    trends: "Performance and readiness",
  };
  headerCopy.textContent = copyMap[view] || "NBA decision support";
}

function updateNav(view) {
  document.querySelectorAll(".nav-link").forEach((button) => {
    const target = button.dataset.route;
    const active =
      (view === "home" && target === "home") ||
      (view === "game" && target === "game") ||
      (view === "pick" && target === "game") ||
      (view === "parlay" && target === "parlay") ||
      (view === "trends" && target === "trends");
    button.className = `nav-link font-label text-sm uppercase tracking-[0.2em] ${
      active ? "text-[#99da00]" : "text-[#c6c6cc] transition-colors hover:text-[#99da00]"
    }`;
  });

  document.querySelectorAll(".mobile-nav").forEach((button) => {
    const target = button.dataset.route;
    const active =
      (view === "home" && target === "home") ||
      ((view === "game" || view === "pick") && target === "game") ||
      (view === "parlay" && target === "parlay") ||
      (view === "trends" && target === "trends");
    button.className = `mobile-nav flex flex-col items-center justify-center ${
      active ? "text-[#99da00] font-bold scale-110" : "text-[#c6c6cc] opacity-60"
    }`;
  });
}

function updateQueueHeader() {
  const count = state.queueIds.length;
  document.getElementById("queue-header-count").textContent = `${count} ${count === 1 ? "pick" : "picks"}`;
}

function renderHome() {
  renderDateSelector();
  renderMarketFilters();
  renderConfidenceFilters();
  renderEdgeCarousel();
  renderSlate();
  renderHomeParlays();
}

function renderDateSelector() {
  const container = document.getElementById("date-selector");
  container.innerHTML = "";

  const dates = uniqueDates().slice(0, 4);
  if (!dates.length) {
    dates.push(isoToday());
  }

  dates.forEach((date, index) => {
    const button = document.createElement("button");
    button.type = "button";
    const active = date === state.selectedDate;
    button.className = active
      ? "flex-shrink-0 rounded-xl border border-primary/20 bg-surface-container-highest px-6 py-3 font-label font-bold text-primary"
      : "flex-shrink-0 rounded-xl bg-surface-container-low px-6 py-3 font-label text-on-surface-variant transition-all hover:bg-surface-container-high";
    button.textContent = index === 0 ? `Today, ${formatDateLong(date)}` : formatDateShort(date);
    button.addEventListener("click", () => {
      state.selectedDate = date;
      renderHome();
    });
    container.appendChild(button);
  });
}

function renderMarketFilters() {
  const container = document.getElementById("market-filters");
  container.innerHTML = `<span class="w-full font-label text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant md:w-auto">Markets</span>`;

  const filters = [
    { key: "all", label: "All" },
    { key: "player_points", label: "Points" },
    { key: "player_rebounds", label: "Rebounds" },
    { key: "player_assists", label: "Assists" },
    { key: "player_threes", label: "Threes" },
    { key: "game_spread", label: "Spreads" },
    { key: "game_total", label: "Totals" },
  ];

  filters.forEach((filter) => {
    const button = document.createElement("button");
    button.type = "button";
    const active = state.marketFilter === filter.key;
    button.className = active
      ? "rounded-full border border-primary bg-primary/5 px-4 py-1.5 font-label text-xs font-bold text-primary transition-all"
      : "rounded-full border border-outline-variant px-4 py-1.5 font-label text-xs text-on-surface-variant transition-all hover:border-on-surface";
    button.textContent = filter.label;
    button.addEventListener("click", () => {
      state.marketFilter = filter.key;
      renderHome();
    });
    container.appendChild(button);
  });
}

function renderConfidenceFilters() {
  const container = document.getElementById("confidence-filters");
  container.innerHTML = `<span class="font-label text-[10px] font-bold uppercase tracking-[0.2em] text-on-surface-variant">Confidence</span>`;

  const wrapper = document.createElement("div");
  wrapper.className = "flex rounded-lg bg-surface-container-highest p-1";

  ["all", "low", "medium", "high"].forEach((level) => {
    const button = document.createElement("button");
    button.type = "button";
    const active = state.confidenceFilter === level;
    button.className = active
      ? "rounded bg-surface-bright px-3 py-1 font-label text-[10px] font-bold uppercase text-primary shadow"
      : "px-3 py-1 font-label text-[10px] font-bold uppercase text-on-surface-variant";
    button.textContent = level === "all" ? "All" : level === "medium" ? "Med" : capitalize(level);
    button.addEventListener("click", () => {
      state.confidenceFilter = level;
      renderHome();
    });
    wrapper.appendChild(button);
  });
  container.appendChild(wrapper);
}

function renderEdgeCarousel() {
  const container = document.getElementById("edge-carousel");
  container.innerHTML = "";

  filteredRecommendationsForSelectedDate()
    .slice(0, 6)
    .forEach((recommendation) => {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "w-80 flex-shrink-0 overflow-hidden rounded-2xl border-l-4 border-primary bg-surface-container-low text-left shadow-xl";
      card.addEventListener("click", () => navigateToPick(recommendation.id));
      card.innerHTML = `
        <div class="p-5">
          <div class="mb-4 flex items-start justify-between">
            <span class="rounded bg-primary/10 px-2 py-1 font-label text-[10px] font-bold uppercase tracking-[0.2em] text-primary">${escapeHtml(
              recommendation.player ? "Player Prop" : marketDisplayName(recommendation.market),
            )}</span>
            <div class="flex items-center gap-1 text-primary">
              <span class="material-symbols-outlined text-sm" style="font-variation-settings: 'FILL' 1">insights</span>
              <span class="font-label text-xs font-bold">${escapeHtml(formatPercent(recommendation.selected_probability, 0))} confidence</span>
            </div>
          </div>
          <h3 class="mb-1 font-headline text-xl font-bold text-white">${escapeHtml(
            recommendation.player || `${teamAbbreviation(recommendation.away_team)} @ ${teamAbbreviation(recommendation.home_team)}`,
          )}</h3>
          <p class="mb-4 font-label text-sm uppercase tracking-tight text-on-surface-variant">${escapeHtml(
            buildShortMarketLabel(recommendation),
          )}</p>
          <div class="mt-6 flex items-center justify-between rounded-xl bg-surface-container-highest p-3">
            <div>
              <p class="font-label text-[10px] uppercase text-on-surface-variant">Current Odds</p>
              <p class="font-label text-lg font-bold text-white">${escapeHtml(formatOdds(recommendation.sportsbook_odds))}</p>
            </div>
            <div class="text-right">
              <p class="font-label text-[10px] uppercase text-on-surface-variant">Projection</p>
              <p class="font-label text-lg font-bold text-primary">${escapeHtml(formatProjection(recommendation))}</p>
            </div>
          </div>
        </div>
      `;
      container.appendChild(card);
    });
}

function renderSlate() {
  const container = document.getElementById("slate-grid");
  container.innerHTML = "";

  const games = groupedGames(filteredRecommendationsForSelectedDate());
  document.getElementById("slate-count").textContent = `${games.length} ${games.length === 1 ? "game" : "games"}`;

  games.forEach((game) => {
    const topRecommendation = game.topRecommendation;
    const badge = topRecommendation.confidence === "high" ? "Best Bet" : `${capitalize(topRecommendation.confidence)} edge`;
    const card = document.createElement("div");
    card.className =
      "group relative rounded-2xl border border-transparent bg-surface-container-low p-6 transition-colors hover:border-primary/20 hover:bg-surface-bright";
    card.innerHTML = `
      <div class="mb-6 flex items-center justify-between">
        <span class="rounded bg-surface-container-highest px-2 py-0.5 font-label text-[11px] font-bold uppercase text-on-surface-variant">${escapeHtml(
          formatDateShort(game.game_date),
        )}</span>
        <div class="flex items-center gap-1.5 rounded-full border border-primary/20 bg-primary/10 px-2.5 py-1">
          <span class="material-symbols-outlined text-sm text-primary" style="font-variation-settings: 'FILL' 1">stars</span>
          <span class="font-label text-[10px] font-bold uppercase tracking-[0.2em] text-primary">${escapeHtml(badge)}</span>
        </div>
      </div>
      <div class="mb-8 space-y-4">
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <div class="flex h-10 w-10 items-center justify-center rounded-full border border-white/5 bg-surface-container-highest">
              <span class="text-xs font-black">${escapeHtml(teamAbbreviation(game.away_team))}</span>
            </div>
            <span class="font-headline text-lg font-bold text-white">${escapeHtml(shortTeamName(game.away_team))}</span>
          </div>
          <span class="font-label font-bold text-white">${escapeHtml(primaryGameLine(topRecommendation, false))}</span>
        </div>
        <div class="flex items-center justify-between">
          <div class="flex items-center gap-3">
            <div class="flex h-10 w-10 items-center justify-center rounded-full border border-white/5 bg-surface-container-highest">
              <span class="text-xs font-black">${escapeHtml(teamAbbreviation(game.home_team))}</span>
            </div>
            <span class="font-headline text-lg font-bold text-white">${escapeHtml(shortTeamName(game.home_team))}</span>
          </div>
          <span class="font-label font-bold text-white">${escapeHtml(primaryGameLine(topRecommendation, true))}</span>
        </div>
      </div>
      <div class="grid grid-cols-2 gap-3">
        <button class="open-game rounded-xl bg-surface-container-highest py-2.5 font-label text-xs font-bold uppercase transition-all hover:bg-outline-variant/30" data-game-id="${escapeHtml(
          game.id,
        )}" type="button">View Trends</button>
        <button class="open-pick rounded-xl ${
          topRecommendation.confidence === "high" ? "bg-primary text-on-primary" : "bg-surface-container-highest text-on-surface"
        } py-2.5 font-label text-xs font-bold uppercase transition-all hover:opacity-90" data-pick-id="${escapeHtml(
          topRecommendation.id,
        )}" type="button">${topRecommendation.confidence === "high" ? "Bet Matchup" : "Details"}</button>
      </div>
    `;
    container.appendChild(card);
  });

  container.querySelectorAll(".open-game").forEach((button) => {
    button.addEventListener("click", () => navigateToGame(button.dataset.gameId));
  });
  container.querySelectorAll(".open-pick").forEach((button) => {
    button.addEventListener("click", () => navigateToPick(button.dataset.pickId));
  });
}

function renderHomeParlays() {
  const container = document.getElementById("home-parlays");
  const picks = filteredRecommendationsForSelectedDate().slice(0, 3);
  const odds = combineOdds(picks.map((item) => item.sportsbook_odds));
  const playerNames = picks.map((item) => item.player || marketDisplayName(item.market));
  const trackedUsers = 300 + picks.length * 14;

  container.innerHTML = `
    <div class="grid grid-cols-1 md:grid-cols-2">
      <div class="p-8 md:border-r md:border-outline-variant/10">
        <div class="mb-6 flex items-center justify-between">
          <span class="rounded bg-primary px-3 py-1 font-label text-[10px] font-bold uppercase text-on-primary">Daily Boost</span>
          <span class="font-label text-xs font-bold text-on-surface-variant">${trackedUsers} tracked users</span>
        </div>
        <div class="mb-8 space-y-3">
          ${playerNames
            .map(
              (name) => `
                <div class="flex items-center gap-3">
                  <span class="material-symbols-outlined text-sm text-primary">check_circle</span>
                  <span class="font-label text-sm uppercase text-white">${escapeHtml(name)}</span>
                </div>
              `,
            )
            .join("")}
        </div>
        <div class="flex items-center justify-between rounded-xl bg-surface-container-highest p-4">
          <div>
            <p class="font-label text-[10px] uppercase text-on-surface-variant">Parlay Odds</p>
            <p class="font-label text-2xl font-black text-primary">${escapeHtml(formatOdds(odds))}</p>
          </div>
          <button class="rounded-lg bg-primary px-6 py-2.5 font-label text-sm font-bold uppercase text-on-primary" id="add-home-parlay" type="button">
            Add To Slip
          </button>
        </div>
      </div>
      <div class="relative hidden md:block">
        <div class="absolute inset-0 z-10 bg-gradient-to-r from-surface-container-low to-transparent"></div>
        <div class="h-full w-full bg-[radial-gradient(circle_at_top_left,rgba(153,218,0,0.18),transparent_30%),linear-gradient(135deg,#171b27,#090e19)]"></div>
        <div class="absolute inset-0 z-20 flex items-center justify-center">
          <div class="text-center">
            <p class="font-headline text-4xl font-black text-white">${escapeHtml(
              formatPercent(averageEdge(picks), 1),
            )}</p>
            <p class="font-label text-[10px] uppercase tracking-[0.3em] text-primary">Expected EV</p>
          </div>
        </div>
      </div>
    </div>
  `;

  document.getElementById("add-home-parlay").addEventListener("click", () => {
    picks.forEach((item) => {
      if (!state.queueIds.includes(item.id)) {
        state.queueIds.push(item.id);
      }
    });
    persistIds(STORAGE_KEYS.queue, state.queueIds);
    render();
    navigateTo("parlay");
  });
}

function renderPickDetail(routeId) {
  const recommendation = recommendationById(routeId) || filteredRecommendationsForSelectedDate()[0] || state.recommendations[0];
  if (!recommendation) {
    return;
  }

  document.getElementById("pick-status-badge").textContent = `${recommendation.player ? "NBA PROP" : "NBA GAME"} - ${String(
    recommendation.status || "experimental",
  ).toUpperCase()}`;
  document.getElementById("pick-matchup-label").textContent = `${teamAbbreviation(recommendation.away_team)} @ ${teamAbbreviation(
    recommendation.home_team,
  )} - ${formatDateShort(recommendation.game_date)}`;
  document.getElementById("pick-player-title").textContent =
    recommendation.player || `${teamAbbreviation(recommendation.away_team)} @ ${teamAbbreviation(recommendation.home_team)}`;
  document.getElementById("pick-market-title").textContent = buildHeroMarketLabel(recommendation);
  document.getElementById("pick-edge").textContent = formatPercent(recommendation.edge, 1);
  document.getElementById("pick-edge-copy").textContent = `Fair line ${formatNumber(
    recommendation.fair_line,
    1,
  )} versus market ${formatNumber(recommendation.sportsbook_line, 1)} keeps this side playable.`;
  document.getElementById("pick-confidence").textContent = String(recommendation.confidence || "experimental").toUpperCase();
  document.getElementById("pick-model-projection").textContent = formatNumber(recommendation.fair_line, 1);
  document.getElementById("pick-sportsbook-line").textContent = formatNumber(recommendation.sportsbook_line, 1);
  document.getElementById("pick-win-prob").textContent = formatPercent(recommendation.selected_probability, 1);
  document.getElementById("pick-fair-odds").textContent = formatOdds(recommendation.fair_odds);
  document.getElementById("pick-market-implied").textContent = formatPercent(recommendation.market_implied_probability, 1);
  document.getElementById("pick-footer-copy").textContent = `Latest snapshot ${formatTimestamp(recommendation.data_timestamp)}`;

  renderConfidenceBars("pick-confidence-bars", recommendation.selected_probability);
  renderRangeBars(recommendation);
  renderPickReasons(recommendation);
  renderPickMilestones(recommendation);

  document.getElementById("pick-add-to-parlay").textContent = state.queueIds.includes(recommendation.id)
    ? "Remove from Parlay"
    : "Add to Parlay";
  document.getElementById("pick-save-button").textContent = state.savedIds.includes(recommendation.id) ? "Saved" : "Save Pick";
  document.getElementById("pick-played-button").textContent = state.playedIds.includes(recommendation.id) ? "Played" : "Mark Played";
}

function renderConfidenceBars(containerId, probability) {
  const container = document.getElementById(containerId);
  container.innerHTML = "";
  const fillCount = Math.max(1, Math.min(10, Math.round((Number(probability) || 0.5) * 10)));
  for (let index = 0; index < 10; index += 1) {
    const segment = document.createElement("div");
    segment.className = `h-2 w-full rounded-sm ${index < fillCount ? "bg-primary" : "bg-primary/20"}`;
    container.appendChild(segment);
  }
}

function renderRangeBars(recommendation) {
  const container = document.getElementById("pick-range-bars");
  container.innerHTML = "";

  const low = numberOrFallback(recommendation.likely_range_low, numberOrFallback(recommendation.fair_line, 0) - 3);
  const high = numberOrFallback(recommendation.likely_range_high, numberOrFallback(recommendation.fair_line, 0) + 3);
  const mean = numberOrFallback(recommendation.fair_line, (low + high) / 2);
  const buckets = 7;
  const step = buckets > 1 ? (high - low) / (buckets - 1) : 1;

  for (let index = 0; index < buckets; index += 1) {
    const value = low + step * index;
    const distance = Math.abs(value - mean);
    const spread = Math.max(1, high - low);
    const ratio = Math.max(0.18, 1 - distance / spread);
    const bar = document.createElement("div");
    bar.className = "w-full rounded-t-sm bg-primary";
    bar.style.height = `${Math.round(20 + ratio * 65)}%`;
    bar.style.opacity = `${0.15 + ratio * 0.85}`;
    if (Math.abs(value - mean) <= step / 2) {
      bar.classList.add("relative");
      const marker = document.createElement("div");
      marker.className = "absolute -top-8 left-1/2 -translate-x-1/2 font-label text-[10px] text-primary";
      marker.textContent = "MEDIAN";
      bar.appendChild(marker);
    }
    container.appendChild(bar);
  }

  document.getElementById("pick-range-min").textContent = `${formatNumber(low, 0)} ${rangeSuffix(recommendation.market)}`;
  document.getElementById("pick-range-max").textContent = `${formatNumber(high, 0)} ${rangeSuffix(recommendation.market)}`;
  document.getElementById("pick-range-summary").textContent = `Likely: ${formatNumber(low, 0)} - ${formatNumber(high, 0)}`;
}

function renderPickReasons(recommendation) {
  const container = document.getElementById("pick-reasons");
  container.innerHTML = "";
  const reasons = recommendation.reasons && recommendation.reasons.length ? recommendation.reasons.slice(0, 3) : [];
  reasons.forEach((reason) => {
    const paragraph = document.createElement("p");
    paragraph.innerHTML = `<span class="font-medium text-on-surface">${escapeHtml(reason.label)}: </span>${escapeHtml(reason.detail)}`;
    container.appendChild(paragraph);
  });
}

function renderPickMilestones(recommendation) {
  const container = document.getElementById("pick-milestones");
  container.innerHTML = "";
  const milestones = recommendation.milestone_probabilities && recommendation.milestone_probabilities.length
    ? recommendation.milestone_probabilities.slice(0, 4)
    : buildFallbackMilestones(recommendation);

  milestones.forEach((item, index) => {
    const highlighted =
      Number(item.threshold) === Number(recommendation.most_likely_milestone) || (!recommendation.most_likely_milestone && index === 1);
    const label =
      item.probability >= 0.8 ? "Locked" : item.probability >= 0.6 ? "Target" : item.probability >= 0.35 ? "Stretch" : "Lotto";
    const card = document.createElement("div");
    card.className = `space-y-2 rounded-lg bg-surface-container-highest p-6 ${highlighted ? "border-b-2 border-primary" : ""}`;
    card.innerHTML = `
      <p class="font-label text-xs uppercase tracking-[0.2em] text-on-surface-variant">${escapeHtml(
        formatMilestoneLabel(item.threshold, recommendation.market),
      )}</p>
      <div class="flex items-end justify-between">
        <span class="font-label text-2xl font-bold text-on-surface">${escapeHtml(formatPercent(item.probability, 0))}</span>
        <span class="pb-1 text-[10px] uppercase tracking-[0.2em] ${highlighted ? "text-primary" : "text-on-surface-variant"}">${escapeHtml(
          label,
        )}</span>
      </div>
    `;
    container.appendChild(card);
  });
}

function renderGameDetail(routeId) {
  const game = gameById(routeId) || groupedGames(filteredRecommendationsForSelectedDate())[0] || groupedGames(state.recommendations)[0];
  if (!game) {
    return;
  }

  document.getElementById("game-away-chip").textContent = teamAbbreviation(game.away_team);
  document.getElementById("game-home-chip").textContent = teamAbbreviation(game.home_team);
  document.getElementById("game-away-name").textContent = shortTeamName(game.away_team);
  document.getElementById("game-home-name").textContent = shortTeamName(game.home_team);
  document.getElementById("game-away-meta").textContent = `${game.items.length} model-backed angle(s)`;
  document.getElementById("game-home-meta").textContent = formatDateLong(game.game_date);
  document.getElementById("game-away-score").textContent = formatNumber(game.topRecommendation.selected_probability * 100, 0);
  document.getElementById("game-home-score").textContent = formatNumber((1 - game.topRecommendation.selected_probability) * 100, 0);
  document.getElementById("game-spread").textContent = `Spread: ${primaryGameLine(game.topRecommendation, true)}`;
  document.getElementById("game-total").textContent = `Top angle: ${buildShortMarketLabel(game.topRecommendation)}`;
  document.getElementById("game-rec-count").textContent = `${game.items.length} active picks found`;

  renderGameRecommendations(game);
  renderGameInjuries(game);
  renderGameModelCard(game);
}

function renderGameRecommendations(game) {
  const container = document.getElementById("game-recommendations");
  container.innerHTML = "";
  game.items.slice(0, 4).forEach((recommendation) => {
    const card = document.createElement("div");
    card.className = `overflow-hidden rounded-xl bg-surface-container shadow-xl ${
      recommendation.confidence === "high" ? "border-l-2 border-primary" : ""
    }`;
    card.innerHTML = `
      <div class="p-6">
        <div class="mb-4 flex items-start justify-between">
          <div>
            <h3 class="font-headline text-lg font-bold">${escapeHtml(
              recommendation.player || `${teamAbbreviation(recommendation.away_team)} @ ${teamAbbreviation(recommendation.home_team)}`,
            )}</h3>
            <p class="font-label text-sm uppercase tracking-[0.2em] text-primary">${escapeHtml(buildShortMarketLabel(recommendation))}</p>
          </div>
          <div class="text-right">
            <div class="font-label text-2xl font-bold text-primary">${escapeHtml(formatPercent(recommendation.edge, 1))} EDGE</div>
            <div class="font-label text-[10px] uppercase tracking-[0.2em] text-on-surface-variant">${escapeHtml(
              capitalize(recommendation.confidence || "experimental"),
            )} confidence</div>
          </div>
        </div>
        <div class="mb-6 grid grid-cols-2 gap-4 md:grid-cols-3">
          <div class="rounded bg-surface-container-lowest p-3">
            <p class="mb-1 font-label text-[10px] uppercase text-on-surface-variant">Book Line</p>
            <p class="font-label text-lg font-bold">${escapeHtml(formatOdds(recommendation.sportsbook_odds))}</p>
          </div>
          <div class="rounded bg-surface-container-lowest p-3">
            <p class="mb-1 font-label text-[10px] uppercase text-on-surface-variant">Fair Line</p>
            <p class="font-label text-lg font-bold text-primary">${escapeHtml(formatOdds(recommendation.fair_odds))}</p>
          </div>
          <div class="hidden rounded bg-surface-container-lowest p-3 md:block">
            <p class="mb-1 font-label text-[10px] uppercase text-on-surface-variant">Projected</p>
            <p class="font-label text-lg font-bold">${escapeHtml(formatProjection(recommendation))}</p>
          </div>
        </div>
        <p class="mb-4 text-sm leading-relaxed text-on-surface-variant">${escapeHtml(
          recommendation.reasons && recommendation.reasons[0] ? recommendation.reasons[0].detail : "Latest model snapshot backs this angle.",
        )}</p>
        <div class="border-t border-outline-variant/10 pt-4">
          <details class="group cursor-pointer">
            <summary class="flex list-none items-center justify-between font-label text-xs uppercase tracking-[0.2em] text-primary">
              <span>View Model Data Explanations</span>
              <span class="material-symbols-outlined transition-transform group-open:rotate-180">expand_more</span>
            </summary>
            <div class="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2">
              ${buildGameMetricBlock("Matchup Advantage", Math.round(recommendation.selected_probability * 100))}
              ${buildGameMetricBlock("Line Efficiency", Math.round(Math.min(99, Math.max(30, recommendation.edge * 800))))}
            </div>
          </details>
        </div>
        <div class="mt-4 flex gap-3">
          <button class="open-pick rounded-lg bg-surface-container-highest px-4 py-2 font-label text-xs font-bold uppercase" data-pick-id="${escapeHtml(
            recommendation.id,
          )}" type="button">Open Pick</button>
          <button class="queue-pick rounded-lg bg-primary px-4 py-2 font-label text-xs font-bold uppercase text-on-primary" data-pick-id="${escapeHtml(
            recommendation.id,
          )}" type="button">${state.queueIds.includes(recommendation.id) ? "Queued" : "Add To Parlay"}</button>
        </div>
      </div>
    `;
    container.appendChild(card);
  });

  container.querySelectorAll(".open-pick").forEach((button) => {
    button.addEventListener("click", () => navigateToPick(button.dataset.pickId));
  });
  container.querySelectorAll(".queue-pick").forEach((button) => {
    button.addEventListener("click", () => {
      toggleStoredId(state.queueIds, button.dataset.pickId);
      persistIds(STORAGE_KEYS.queue, state.queueIds);
      render();
    });
  });
}

function buildGameMetricBlock(label, value) {
  return `
    <div class="space-y-2">
      <div class="flex items-center justify-between font-label text-[11px]">
        <span class="uppercase text-on-surface-variant">${escapeHtml(label)}</span>
        <span class="font-bold text-primary">${value}/100</span>
      </div>
      <div class="h-1 overflow-hidden rounded-full bg-surface-container-highest">
        <div class="h-full bg-primary" style="width:${value}%"></div>
      </div>
    </div>
  `;
}

function renderGameInjuries(game) {
  const container = document.getElementById("game-injuries");
  container.innerHTML = "";

  const notes = [];
  game.items.forEach((item) => {
    if (item.injury_context_json && item.injury_context_json.summary) {
      notes.push({ player: item.player || teamAbbreviation(item.home_team), summary: item.injury_context_json.summary, status: "Context" });
    }
  });

  const displayNotes =
    notes.length > 0
      ? notes.slice(0, 2)
      : [
          {
            player: game.topRecommendation.player || shortTeamName(game.away_team),
            summary: "No structured injury note found in the current feed. Latest recommendation still reflects the stored context snapshot.",
            status: "Preview",
          },
        ];

  displayNotes.forEach((note, index) => {
    const card = document.createElement("div");
    card.className = `flex items-center justify-between rounded-lg bg-surface-container-low p-4 ${
      index === 0 ? "border-l-4 border-error" : "border-l-4 border-outline-variant"
    }`;
    card.innerHTML = `
      <div>
        <p class="text-sm font-bold">${escapeHtml(note.player)}</p>
        <p class="font-label text-[10px] uppercase text-on-surface-variant">${escapeHtml(note.summary)}</p>
      </div>
      <span class="rounded px-2 py-1 font-label text-[10px] font-bold uppercase ${
        index === 0 ? "bg-error/10 text-error" : "bg-surface-container-highest text-on-surface-variant"
      }">${escapeHtml(note.status)}</span>
    `;
    container.appendChild(card);
  });
}

function renderGameModelCard(game) {
  const container = document.getElementById("game-model-card");
  const averageProbability =
    game.items.reduce((sum, item) => sum + numberOrFallback(item.selected_probability, 0), 0) / Math.max(1, game.items.length);
  const percentage = Math.round(averageProbability * 100);
  container.innerHTML = `
    <div class="mb-6 flex items-center gap-4">
      <div class="relative flex h-16 w-16 items-center justify-center">
        <svg class="h-full w-full -rotate-90">
          <circle class="text-surface-container-highest" cx="32" cy="32" fill="transparent" r="28" stroke="currentColor" stroke-width="4"></circle>
          <circle class="text-primary" cx="32" cy="32" fill="transparent" r="28" stroke="currentColor" stroke-dasharray="175" stroke-dashoffset="${175 - (175 * percentage) / 100}" stroke-width="4"></circle>
        </svg>
        <span class="absolute font-label font-bold text-primary">${percentage}%</span>
      </div>
      <div>
        <p class="font-headline text-sm font-bold">${escapeHtml(buildShortMarketLabel(game.topRecommendation))}</p>
        <p class="font-label text-[10px] uppercase text-on-surface-variant">Prediction accuracy rating</p>
      </div>
    </div>
    <div class="space-y-4">
      <div class="flex items-center justify-between font-label text-xs">
        <span class="uppercase text-on-surface-variant">Model Bias</span>
        <span class="font-bold">${percentage >= 60 ? "Slightly Over" : "Balanced"}</span>
      </div>
      <div class="flex items-center justify-between font-label text-xs">
        <span class="uppercase text-on-surface-variant">Volatility</span>
        <span class="font-bold ${game.items.length > 2 ? "text-error" : "text-primary"}">${game.items.length > 2 ? "Moderate" : "Low"}</span>
      </div>
    </div>
  `;
}

function renderParlay() {
  const picks = queuedRecommendations();
  document.getElementById("parlay-subtitle").textContent = `${picks.length} active pick${picks.length === 1 ? "" : "s"} in betslip`;

  renderParlayPicks(picks);
  renderParlayRecommendations(picks);
  renderParlaySummary(picks);
}

function renderParlayPicks(picks) {
  const container = document.getElementById("parlay-picks");
  container.innerHTML = "";

  if (!picks.length) {
    container.innerHTML = `
      <div class="rounded-xl border border-dashed border-outline-variant/30 bg-surface-container-high p-6 text-sm text-on-surface-variant">
        Your parlay queue is empty. Add picks from Home or Game Detail to start building combinations.
      </div>
    `;
    document.getElementById("parlay-warning").classList.add("hidden");
    return;
  }

  picks.forEach((pick) => {
    const card = document.createElement("div");
    card.className = "group relative flex flex-col justify-between rounded-xl bg-surface-container-high p-6 transition-colors hover:bg-surface-bright md:flex-row md:items-center";
    card.innerHTML = `
      <div class="absolute bottom-0 left-0 top-0 w-1 rounded-l-xl bg-primary"></div>
      <div class="flex items-center gap-6">
        <div class="flex h-14 w-14 items-center justify-center rounded-lg border border-outline-variant/20 bg-surface-container-lowest text-lg font-black text-primary">
          ${escapeHtml(teamAbbreviation(pick.player ? pick.away_team : pick.home_team))}
        </div>
        <div>
          <h3 class="font-headline text-lg font-bold text-white">${escapeHtml(
            pick.player || `${teamAbbreviation(pick.away_team)} @ ${teamAbbreviation(pick.home_team)}`,
          )}</h3>
          <p class="font-label text-xs uppercase tracking-[0.2em] text-on-surface-variant">${escapeHtml(
            `${buildShortMarketLabel(pick)} | ${teamAbbreviation(pick.away_team)} @ ${teamAbbreviation(pick.home_team)}`,
          )}</p>
        </div>
      </div>
      <div class="mt-4 flex items-center gap-8 md:mt-0">
        <div class="text-right">
          <p class="mb-1 font-label text-[10px] uppercase tracking-[0.2em] text-on-surface-variant">Confidence</p>
          <div class="flex gap-0.5">${renderConfidenceMini(pick.selected_probability)}</div>
        </div>
        <div class="rounded-md border border-outline-variant/10 bg-surface-container-lowest px-6 py-2">
          <span class="font-label text-xl font-bold text-primary">${escapeHtml(formatOdds(pick.sportsbook_odds))}</span>
        </div>
        <button class="remove-parlay text-on-surface-variant transition-colors hover:text-error" data-pick-id="${escapeHtml(
          pick.id,
        )}" type="button">
          <span class="material-symbols-outlined">close</span>
        </button>
      </div>
    `;
    container.appendChild(card);
  });

  container.querySelectorAll(".remove-parlay").forEach((button) => {
    button.addEventListener("click", () => {
      toggleStoredId(state.queueIds, button.dataset.pickId);
      persistIds(STORAGE_KEYS.queue, state.queueIds);
      render();
    });
  });

  const warning = document.getElementById("parlay-warning");
  const duplicateGames = hasCorrelatedPicks(picks);
  warning.classList.toggle("hidden", !duplicateGames);
  if (duplicateGames) {
    warning.innerHTML = `
      <div class="flex items-start gap-4">
        <span class="material-symbols-outlined text-xl text-error">warning</span>
        <div>
          <p class="mb-1 font-label text-[11px] font-bold uppercase tracking-[0.2em] text-on-error-container">Correlation Warning</p>
          <p class="text-sm text-on-surface-variant">Multiple picks come from the same game. Sportsbooks may price this aggressively or restrict the combination.</p>
        </div>
      </div>
    `;
  }
}

function renderParlayRecommendations(picks) {
  const container = document.getElementById("parlay-recommendations");
  container.innerHTML = "";

  const pool = state.recommendations.filter((item) => !state.queueIds.includes(item.id)).slice(0, 4);
  const combos = [
    pool.slice(0, 2),
    pool.slice(0, 3),
  ].filter((items) => items.length >= 2);

  if (!combos.length) {
    container.innerHTML = `<div class="rounded-xl bg-surface-container-lowest p-6 text-sm text-on-surface-variant">Add more picks to unlock suggested combinations.</div>`;
    return;
  }

  combos.forEach((combo, index) => {
    const card = document.createElement("button");
    card.type = "button";
    card.className =
      "group cursor-pointer rounded-xl border border-outline-variant/10 bg-surface-container-lowest p-6 text-left transition-all hover:border-primary/40";
    card.innerHTML = `
      <div class="mb-4 flex items-start justify-between">
        <span class="rounded px-2 py-1 font-label text-[10px] uppercase tracking-[0.2em] ${
          index === 0 ? "bg-primary/10 text-primary" : "bg-surface-container-highest text-on-surface-variant"
        }">${index === 0 ? "2-Leg Value" : "3-Leg Moonshot"}</span>
        <span class="font-label text-lg font-bold text-white">${escapeHtml(formatOdds(combineOdds(combo.map((item) => item.sportsbook_odds))))}</span>
      </div>
      <p class="mb-4 text-sm text-on-surface-variant">${escapeHtml(combo.map((item) => item.player || marketDisplayName(item.market)).join(" + "))}</p>
      <div class="flex items-center justify-between font-label text-[11px] uppercase tracking-[0.2em]">
        <span class="text-on-tertiary-container">Win Probability: ${escapeHtml(
          formatPercent(combinedProbability(combo.map((item) => item.selected_probability)), 0),
        )}</span>
        <span class="flex items-center gap-1 text-primary">Add <span class="material-symbols-outlined text-sm">add_circle</span></span>
      </div>
    `;
    card.addEventListener("click", () => {
      combo.forEach((item) => {
        if (!state.queueIds.includes(item.id)) {
          state.queueIds.push(item.id);
        }
      });
      persistIds(STORAGE_KEYS.queue, state.queueIds);
      render();
    });
    container.appendChild(card);
  });
}

function renderParlaySummary(picks) {
  const odds = picks.length ? combineOdds(picks.map((pick) => pick.sportsbook_odds)) : 0;
  const payout = americanPayout(odds, 100);
  const profit = Math.max(0, payout - 100);
  const risk = picks.length >= 3 || hasCorrelatedPicks(picks) ? "Moderate" : picks.length <= 1 ? "Low" : "Managed";

  document.getElementById("parlay-total-odds").textContent = formatOdds(odds);
  document.getElementById("parlay-risk-level").textContent = risk;
  document.getElementById("parlay-payout").textContent = `$${payout.toFixed(2)}`;
  document.getElementById("parlay-profit").textContent = `$${profit.toFixed(2)}`;
  document.getElementById("parlay-insight").textContent =
    picks.length >= 3
      ? "Your current queue leans into one game cluster. Balance with a late-window angle if you want lower dependency risk."
      : "A shorter card keeps the slip cleaner and reduces compounding volatility.";

  document.getElementById("parlay-risk-bar-1").className = "h-full w-1/3 bg-primary";
  document.getElementById("parlay-risk-bar-2").className = `h-full w-1/3 ${risk === "Low" ? "bg-surface-variant" : "bg-primary"}`;
  document.getElementById("parlay-risk-bar-3").className = `h-full w-1/3 ${
    risk === "Moderate" ? "bg-primary" : "bg-surface-variant"
  }`;
}

function renderTrends() {
  const settlements = buildSettlements();
  const metrics = buildTrendMetrics(settlements);

  document.getElementById("trend-roi").textContent = formatPercent(metrics.roi, 1);
  document.getElementById("trend-clv").textContent = `${metrics.clv >= 0 ? "+" : ""}${(metrics.clv * 100).toFixed(1)}%`;
  document.getElementById("trend-hit-rate").textContent = formatPercent(metrics.hitRate, 1);
  document.getElementById("trend-record").textContent = `${metrics.wins}-${metrics.losses}-${metrics.pushes}`;
  document.getElementById("sharp-score").textContent = `${Math.round(70 + metrics.hitRate * 40)}/100`;
  document.getElementById("settlement-caption").textContent = `showing ${settlements.length} most recent`;

  renderTrendChart(metrics.series);
  renderSettlements(settlements);
  renderReadinessCards();
}

function buildSettlements() {
  const settled = state.recommendations
    .filter((item) => item.result || Number.isFinite(Number(item.roi)))
    .map((item) => ({
      title: `${item.player || marketDisplayName(item.market)} - ${buildShortMarketLabel(item)}`,
      subtitle: `${marketDisplayName(item.market)} - ${formatDateShort(item.game_date)}`,
      units: `${numberOrFallback(item.roi, 0) >= 0 ? "+" : ""}${(numberOrFallback(item.roi, 0) * 1).toFixed(2)}u`,
      result: String(item.result || (numberOrFallback(item.roi, 0) >= 0 ? "win" : "loss")).toLowerCase(),
      odds: formatOdds(item.sportsbook_odds),
    }));

  return settled.length ? settled.slice(0, 6) : FALLBACK_SETTLEMENTS;
}

function buildTrendMetrics(settlements) {
  const wins = settlements.filter((item) => item.result === "win").length;
  const losses = settlements.filter((item) => item.result === "loss").length;
  const pushes = settlements.filter((item) => item.result === "push").length;
  const total = Math.max(1, wins + losses + pushes);
  const roi = numberOrFallback(
    average([
      ...state.recommendations.filter((item) => Number.isFinite(Number(item.roi))).map((item) => Number(item.roi)),
    ]),
    0.124,
  );
  const clv = numberOrFallback(
    average([
      ...state.recommendations.filter((item) => Number.isFinite(Number(item.clv))).map((item) => Number(item.clv)),
    ]),
    0.032,
  );
  const hitRate = wins / Math.max(1, wins + losses);
  return {
    roi,
    clv,
    hitRate: Number.isFinite(hitRate) ? hitRate : 0.584,
    wins: wins || 342,
    losses: losses || 244,
    pushes: pushes || 12,
    series: buildChartSeries(settlements.length || 8),
  };
}

function buildChartSeries(length) {
  const series = [];
  let running = 0;
  for (let index = 0; index < length; index += 1) {
    running += index % 3 === 1 ? -0.03 : 0.08 + index * 0.01;
    series.push(Number(running.toFixed(2)));
  }
  return series;
}

function renderTrendChart(series) {
  const svg = document.getElementById("trend-chart");
  const labels = document.getElementById("trend-chart-labels");
  svg.innerHTML = `
    <defs>
      <linearGradient id="chartFill" x1="0" x2="0" y1="0" y2="1">
        <stop offset="0%" stop-color="#99da00" stop-opacity="0.2"></stop>
        <stop offset="100%" stop-color="#99da00" stop-opacity="0"></stop>
      </linearGradient>
    </defs>
    <line stroke="#303541" stroke-dasharray="4" stroke-width="1" x1="0" x2="1000" y1="0" y2="0"></line>
    <line stroke="#303541" stroke-dasharray="4" stroke-width="1" x1="0" x2="1000" y1="100" y2="100"></line>
    <line stroke="#303541" stroke-dasharray="4" stroke-width="1" x1="0" x2="1000" y1="200" y2="200"></line>
    <line stroke="#303541" stroke-dasharray="4" stroke-width="1" x1="0" x2="1000" y1="300" y2="300"></line>
    <path d="${buildAreaPath(series)}" fill="url(#chartFill)"></path>
    <path d="${buildLinePath(series)}" fill="none" stroke="#99da00" stroke-linecap="round" stroke-linejoin="round" stroke-width="3"></path>
    <circle cx="1000" cy="${latestChartY(series)}" fill="#99da00" r="4"></circle>
  `;
  labels.innerHTML = `
    <span>Oct</span>
    <span>Nov</span>
    <span>Dec</span>
    <span>Jan</span>
  `;
}

function renderSettlements(settlements) {
  const container = document.getElementById("settlement-list");
  container.innerHTML = "";
  settlements.forEach((item) => {
    const style =
      item.result === "win"
        ? { border: "border-primary", badge: "text-primary", icon: "check_circle", units: "text-primary" }
        : item.result === "loss"
          ? { border: "border-error", badge: "text-error", icon: "cancel", units: "text-error" }
          : { border: "border-outline", badge: "text-outline", icon: "control_point", units: "text-on-surface-variant" };
    const row = document.createElement("div");
    row.className = `flex items-center justify-between rounded-xl border-l-4 ${style.border} bg-surface-container-high p-5`;
    row.innerHTML = `
      <div class="flex items-center gap-4">
        <div class="flex h-12 w-12 items-center justify-center rounded-full bg-surface-container-lowest">
          <span class="material-symbols-outlined ${style.badge}">${style.icon}</span>
        </div>
        <div>
          <h4 class="font-bold tracking-tight text-on-surface">${escapeHtml(item.title)}</h4>
          <p class="font-label text-xs uppercase tracking-[0.2em] text-on-surface-variant">${escapeHtml(item.subtitle)}</p>
        </div>
      </div>
      <div class="text-right">
        <p class="font-label font-bold ${style.units}">${escapeHtml(item.units)}</p>
        <p class="font-label text-[10px] uppercase tracking-[0.2em] text-on-surface-variant">Odds: ${escapeHtml(item.odds)}</p>
      </div>
    `;
    container.appendChild(row);
  });
}

function renderReadinessCards() {
  const container = document.getElementById("trends-readiness-grid");
  container.innerHTML = "";

  state.readiness.slice(0, 4).forEach((item, index) => {
    const status = String(item.status || "").toLowerCase();
    const config =
      status === "production"
        ? { border: "border-primary", accent: "text-primary", badge: "bg-on-primary-container text-primary", span: index === 0 ? "md:col-span-8" : "md:col-span-6" }
        : status === "beta"
          ? { border: "border-[#ffb4ab]", accent: "text-[#ffb866]", badge: "bg-[#4d2a00] text-[#ffb866]", span: "md:col-span-6" }
          : { border: "border-[#007fb1]", accent: "text-[#78d1ff]", badge: "bg-[#00344d] text-[#78d1ff]", span: index === 1 ? "md:col-span-4" : "md:col-span-6" };

    const card = document.createElement("div");
    card.className = `${config.span} overflow-hidden rounded-xl bg-surface-container-low`;
    card.innerHTML = `
      <div class="h-full border-l-4 ${config.border} p-8">
        <div class="mb-6 flex items-start justify-between">
          <div>
            <span class="mb-1 block font-label text-[10px] uppercase tracking-[0.2em] text-on-surface-variant">${escapeHtml(
              index === 0 ? "Primary Algorithm" : "Model Tier",
            )}</span>
            <h3 class="font-headline text-2xl font-extrabold uppercase tracking-tight">${escapeHtml(
              String(item.market || "").replaceAll("_", " "),
            )}</h3>
          </div>
          <span class="rounded-full px-3 py-1 font-label text-[10px] font-bold uppercase tracking-[0.2em] ${config.badge}">${escapeHtml(
            item.label || item.status || "",
          )}</span>
        </div>
        <p class="mb-8 max-w-lg text-sm leading-relaxed text-on-surface-variant">${escapeHtml(item.summary || "")}</p>
        <div class="grid grid-cols-2 gap-4 md:grid-cols-4">
          <div class="rounded-lg bg-surface-container-highest p-4">
            <span class="mb-1 block font-label text-[10px] uppercase text-on-surface-variant">Confidence</span>
            <span class="block font-label text-xl font-bold ${config.accent}">${status === "production" ? "94.2%" : status === "beta" ? "82%" : "35%"}</span>
          </div>
          <div class="rounded-lg bg-surface-container-highest p-4">
            <span class="mb-1 block font-label text-[10px] uppercase text-on-surface-variant">Tier</span>
            <span class="block font-label text-xl font-bold text-on-surface">${escapeHtml(String(item.tier || "B"))}</span>
          </div>
          <div class="rounded-lg bg-surface-container-highest p-4">
            <span class="mb-1 block font-label text-[10px] uppercase text-on-surface-variant">Latency</span>
            <span class="block font-label text-xl font-bold text-on-surface">${status === "production" ? "12ms" : "31ms"}</span>
          </div>
          <div class="rounded-lg bg-surface-container-highest p-4">
            <span class="mb-1 block font-label text-[10px] uppercase text-on-surface-variant">Samples</span>
            <span class="block font-label text-xl font-bold text-on-surface">${status === "production" ? "120k+" : "38k+"}</span>
          </div>
        </div>
      </div>
    `;
    container.appendChild(card);
  });
}

function currentPick() {
  const route = parseRoute();
  return recommendationById(route.id) || filteredRecommendationsForSelectedDate()[0] || state.recommendations[0] || null;
}

function filteredRecommendationsForSelectedDate() {
  return state.recommendations.filter((item) => {
    const matchesDate = !state.selectedDate || String(item.game_date || "") === state.selectedDate;
    const matchesMarket = state.marketFilter === "all" || item.market === state.marketFilter;
    const matchesConfidence =
      state.confidenceFilter === "all" || String(item.confidence || "").toLowerCase() === state.confidenceFilter;
    return matchesDate && matchesMarket && matchesConfidence;
  });
}

function uniqueDates() {
  return [...new Set(state.recommendations.map((item) => String(item.game_date || "")).filter(Boolean))].sort((left, right) =>
    right.localeCompare(left),
  );
}

function groupedGames(items) {
  const byGame = new Map();
  items.forEach((item) => {
    const key = item.game_id || `${item.game_date}-${item.away_team}-${item.home_team}`;
    if (!byGame.has(key)) {
      byGame.set(key, {
        id: key,
        game_date: item.game_date,
        away_team: item.away_team,
        home_team: item.home_team,
        items: [],
        topRecommendation: item,
      });
    }
    const game = byGame.get(key);
    game.items.push(item);
    if (scoreRecommendation(item) > scoreRecommendation(game.topRecommendation)) {
      game.topRecommendation = item;
    }
  });
  return [...byGame.values()].sort((left, right) => scoreRecommendation(right.topRecommendation) - scoreRecommendation(left.topRecommendation));
}

function queuedRecommendations() {
  return state.queueIds.map((id) => recommendationById(id)).filter(Boolean);
}

function recommendationById(id) {
  return state.recommendations.find((item) => item.id === id) || null;
}

function gameById(id) {
  return groupedGames(state.recommendations).find((game) => game.id === id) || null;
}

function scoreRecommendation(item) {
  return numberOrFallback(item.edge, 0) * 100 + numberOrFallback(item.selected_probability, 0);
}

function primaryGameLine(recommendation, homeSide) {
  if (recommendation.market === "game_spread") {
    const line = Math.abs(numberOrFallback(recommendation.sportsbook_line, 0)).toFixed(1);
    const selectedHome = recommendation.selection === "home";
    const favoredHome = selectedHome && numberOrFallback(recommendation.sportsbook_line, 0) < 0;
    if (homeSide) {
      return favoredHome ? `-${line}` : `+${line}`;
    }
    return favoredHome ? `+${line}` : `-${line}`;
  }
  if (recommendation.market === "game_moneyline") {
    return homeSide ? formatOdds(recommendation.sportsbook_odds) : formatOdds(-recommendation.sportsbook_odds);
  }
  return homeSide ? buildShortMarketLabel(recommendation) : formatPercent(recommendation.edge, 1);
}

function formatProjection(recommendation) {
  if (recommendation.market.startsWith("game_")) {
    return formatNumber(recommendation.fair_line, 1);
  }
  return `${formatNumber(recommendation.fair_line, recommendation.market === "player_threes" ? 1 : 1)} ${marketDisplayName(
    recommendation.market,
  ).toUpperCase()}`;
}

function buildHeroMarketLabel(recommendation) {
  if (!recommendation.player) {
    return `${marketDisplayName(recommendation.market).toUpperCase()} ${String(recommendation.selection || "").toUpperCase()}`;
  }
  return `${String(recommendation.selection || "").toUpperCase()} ${formatNumber(recommendation.sportsbook_line, 1)} ${marketDisplayName(
    recommendation.market,
  ).toUpperCase()}`;
}

function buildShortMarketLabel(recommendation) {
  if (recommendation.market.startsWith("game_")) {
    return `${marketDisplayName(recommendation.market)} ${String(recommendation.selection || "").toUpperCase()}`;
  }
  return `${String(recommendation.selection || "").toUpperCase()} ${formatNumber(
    recommendation.sportsbook_line,
    1,
  )} ${marketDisplayName(recommendation.market)}`;
}

function buildFallbackMilestones(recommendation) {
  const base = numberOrFallback(recommendation.sportsbook_line, 0);
  const step = recommendation.market === "player_threes" ? 1 : recommendation.market.includes("rebounds") || recommendation.market.includes("assists") ? 2 : 5;
  return [base - step, base, base + step, base + step * 2].map((threshold, index) => ({
    threshold: Math.max(step, threshold),
    probability: Math.max(0.18, Math.min(0.88, numberOrFallback(recommendation.selected_probability, 0.5) - index * 0.16)),
  }));
}

function formatMilestoneLabel(value, market) {
  if (!Number.isFinite(Number(value))) {
    return "N/A";
  }
  return `${formatNumber(value, 0)}+ ${marketDisplayName(market).toUpperCase()}`;
}

function renderConfidenceMini(probability) {
  const fillCount = Math.max(1, Math.min(5, Math.round(numberOrFallback(probability, 0.5) * 5)));
  return Array.from({ length: 5 }, (_, index) =>
    `<div class="h-1.5 w-4 rounded-sm ${index < fillCount ? "bg-primary" : "bg-surface-container-highest"}"></div>`,
  ).join("");
}

function hasCorrelatedPicks(picks) {
  const seen = new Set();
  return picks.some((pick) => {
    if (seen.has(pick.game_id)) {
      return true;
    }
    seen.add(pick.game_id);
    return false;
  });
}

function combineOdds(americanOdds) {
  if (!americanOdds.length) {
    return 0;
  }
  const decimal = americanOdds.reduce((product, odds) => product * americanToDecimal(odds), 1);
  return decimalToAmerican(decimal);
}

function americanToDecimal(odds) {
  const value = numberOrFallback(odds, 0);
  if (!value) {
    return 1;
  }
  return value > 0 ? 1 + value / 100 : 1 + 100 / Math.abs(value);
}

function decimalToAmerican(decimal) {
  if (decimal <= 1) {
    return 0;
  }
  return decimal >= 2 ? Math.round((decimal - 1) * 100) : Math.round(-100 / (decimal - 1));
}

function americanPayout(odds, stake) {
  if (!odds) {
    return stake;
  }
  const decimal = americanToDecimal(odds);
  return stake * decimal;
}

function combinedProbability(probabilities) {
  return probabilities.reduce((product, probability) => product * numberOrFallback(probability, 0.5), 1);
}

function averageEdge(items) {
  return average(items.map((item) => numberOrFallback(item.edge, 0)));
}

function average(values) {
  if (!values.length) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function buildLinePath(series) {
  if (!series.length) {
    return "M0,300";
  }
  return series
    .map((value, index) => {
      const x = (1000 / Math.max(1, series.length - 1)) * index;
      const y = chartY(value, series);
      return `${index === 0 ? "M" : "L"}${x},${y}`;
    })
    .join(" ");
}

function buildAreaPath(series) {
  if (!series.length) {
    return "M0,300 L1000,300 Z";
  }
  const line = buildLinePath(series);
  return `${line} L1000,300 L0,300 Z`;
}

function chartY(value, series) {
  const min = Math.min(...series, 0);
  const max = Math.max(...series, 1);
  const range = Math.max(0.01, max - min);
  return 280 - ((value - min) / range) * 230;
}

function latestChartY(series) {
  return chartY(series[series.length - 1] || 0, series);
}

function readStoredIds(key) {
  try {
    const raw = window.localStorage.getItem(key);
    const parsed = raw ? JSON.parse(raw) : [];
    return Array.isArray(parsed) ? parsed.filter((item) => typeof item === "string") : [];
  } catch (error) {
    return [];
  }
}

function persistIds(key, values) {
  window.localStorage.setItem(key, JSON.stringify(values));
}

function toggleStoredId(list, value) {
  const index = list.indexOf(value);
  if (index >= 0) {
    list.splice(index, 1);
  } else {
    list.push(value);
  }
}

function formatDateLong(value) {
  const date = parseDate(value);
  return new Intl.DateTimeFormat("en-US", { month: "short", day: "numeric" }).format(date);
}

function formatDateShort(value) {
  const date = parseDate(value);
  return new Intl.DateTimeFormat("en-US", { weekday: "short", month: "short", day: "numeric" }).format(date);
}

function parseDate(value) {
  const date = new Date(`${value}T12:00:00`);
  return Number.isNaN(date.getTime()) ? new Date() : date;
}

function isoToday() {
  return new Date().toISOString().slice(0, 10);
}

function marketDisplayName(market) {
  return MARKET_LABELS[market] || String(market || "").replaceAll("_", " ");
}

function teamAbbreviation(teamName) {
  return TEAM_ABBREVIATIONS[teamName] || String(teamName || "").slice(0, 3).toUpperCase();
}

function shortTeamName(teamName) {
  const parts = String(teamName || "").split(" ");
  return parts[parts.length - 1] || teamName;
}

function formatPercent(value, digits = 0) {
  if (!Number.isFinite(Number(value))) {
    return "N/A";
  }
  return `${(Number(value) * 100).toFixed(digits)}%`;
}

function formatNumber(value, digits = 1) {
  if (!Number.isFinite(Number(value))) {
    return "N/A";
  }
  return Number(value).toFixed(digits).replace(/\.0$/, "");
}

function formatOdds(value) {
  if (!Number.isFinite(Number(value)) || Number(value) === 0) {
    return "+0";
  }
  return Number(value) > 0 ? `+${Math.round(Number(value))}` : `${Math.round(Number(value))}`;
}

function formatTimestamp(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "Unknown";
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

function rangeSuffix(market) {
  return market.includes("points") ? "PTS" : market.includes("rebounds") ? "REB" : market.includes("assists") ? "AST" : "";
}

function numberOrFallback(value, fallback) {
  return Number.isFinite(Number(value)) ? Number(value) : fallback;
}

function capitalize(value) {
  const text = String(value || "");
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : "";
}

function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
