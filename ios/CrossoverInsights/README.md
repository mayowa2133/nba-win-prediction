# Crossover Insights iOS App

Native SwiftUI prototype for the NBA recommendation product.

## Run

1. `cd ios/CrossoverInsights`
2. `xcodegen generate`
3. Open `CrossoverInsights.xcodeproj`
4. Select an iPhone simulator such as `iPhone 17`, `iPhone 17 Pro`, or `iPhone 16e`
5. Run the `CrossoverInsights` scheme

## API

The app now expects live API data at runtime.

Configuration order:

1. `CROSSOVER_API_BASE_URL` environment variable in the Xcode scheme
2. `CrossoverAPIBaseURL` Info.plist value from the project build setting
3. Simulator-only fallback to `http://127.0.0.1:8000`

For a physical iPhone, do not use `127.0.0.1`. Start the FastAPI app on your Mac with:

- `uvicorn src.api.app:app --host 0.0.0.0 --port 8000`

Then point the app to your Mac's LAN IP:

- `CROSSOVER_API_BASE_URL=http://192.168.x.x:8000`

Runtime failures now show explicit loading, empty, and error states instead of silently falling back to preview fixtures.

Optional debug-only demo mode:

- `CROSSOVER_DEMO_MODE=1`

## Troubleshooting

- If Xcode says `Signing for "CrossoverInsights" requires a development team`, you are building to a physical iPhone. Either:
  - switch the run destination to an iPhone simulator, or
  - open `Signing & Capabilities` and choose your Apple development team
- If a specific simulator name fails, check Xcode's destination list. Simulator availability depends on the installed runtime set.
- If the app shows a base URL error on a physical iPhone, set `CROSSOVER_API_BASE_URL` to a reachable LAN or staging host.
