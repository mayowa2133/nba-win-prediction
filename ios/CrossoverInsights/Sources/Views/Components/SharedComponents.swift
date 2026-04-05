import SwiftUI

struct SurfaceCard<Content: View>: View {
    let background: Color
    let content: Content

    init(background: Color = AppTheme.surfaceLow, @ViewBuilder content: () -> Content) {
        self.background = background
        self.content = content()
    }

    var body: some View {
        content
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(background)
            .clipShape(RoundedRectangle(cornerRadius: AppTheme.cardRadius, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: AppTheme.cardRadius, style: .continuous)
                    .stroke(AppTheme.outline.opacity(0.22), lineWidth: 1)
            )
    }
}

struct SectionTitle: View {
    let title: String
    let subtitle: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title)
                .font(.headline(24))
                .foregroundStyle(AppTheme.text)
            if let subtitle {
                Text(subtitle)
                    .font(.system(size: 13, weight: .medium))
                    .foregroundStyle(AppTheme.textMuted)
            }
        }
    }
}

struct StatusBadge: View {
    let text: String
    let accent: Color
    let fill: Color

    var body: some View {
        Text(text.uppercased())
            .font(.label(10, weight: .bold))
            .tracking(1.4)
            .foregroundStyle(accent)
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
            .background(fill)
            .clipShape(Capsule())
    }
}

struct ConfidenceMeter: View {
    let probability: Double
    let segments: Int
    let compact: Bool

    init(probability: Double, segments: Int = 10, compact: Bool = false) {
        self.probability = probability
        self.segments = segments
        self.compact = compact
    }

    var body: some View {
        let count = max(1, min(segments, Int((probability * Double(segments)).rounded())))
        HStack(spacing: compact ? 2 : 4) {
            ForEach(0..<segments, id: \.self) { index in
                RoundedRectangle(cornerRadius: compact ? 2 : 3, style: .continuous)
                    .fill(index < count ? AppTheme.primary : AppTheme.primary.opacity(0.18))
                    .frame(width: compact ? 14 : nil, height: compact ? 6 : 8)
            }
        }
    }
}

struct MetricTile: View {
    let title: String
    let value: String
    var accent: Color = AppTheme.text

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title.uppercased())
                .font(.label(10, weight: .bold))
                .foregroundStyle(AppTheme.textMuted)
            Text(value)
                .font(.label(20, weight: .bold))
                .foregroundStyle(accent)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
        .background(AppTheme.surfaceHighest)
        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
    }
}

struct TeamBubble: View {
    let code: String

    var body: some View {
        ZStack {
            Circle()
                .fill(AppTheme.surfaceHighest)
            Text(code)
                .font(.label(14, weight: .black))
                .foregroundStyle(AppTheme.text)
        }
        .frame(width: 42, height: 42)
    }
}

struct RangeBars: View {
    let recommendation: Recommendation

    var body: some View {
        let low = recommendation.likelyRangeLow ?? recommendation.fairLine - 3
        let high = recommendation.likelyRangeHigh ?? recommendation.fairLine + 3
        let mid = recommendation.fairLine
        HStack(alignment: .bottom, spacing: 4) {
            ForEach(0..<7, id: \.self) { index in
                let value = low + ((high - low) / 6.0) * Double(index)
                let ratio = max(0.18, 1 - abs(value - mid) / max(1, high - low))
                RoundedRectangle(cornerRadius: 4, style: .continuous)
                    .fill(AppTheme.primary.opacity(0.15 + ratio * 0.85))
                    .frame(maxWidth: .infinity)
                    .frame(height: 36 + ratio * 110)
                    .overlay(alignment: .top) {
                        if abs(value - mid) < ((high - low) / 8.0) {
                            Text("MEDIAN")
                                .font(.label(10, weight: .bold))
                                .foregroundStyle(AppTheme.primary)
                                .offset(y: -24)
                        }
                    }
            }
        }
        .frame(height: 170, alignment: .bottom)
    }
}

struct LoadingCard: View {
    let title: String

    var body: some View {
        SurfaceCard {
            HStack(spacing: 12) {
                ProgressView()
                    .tint(AppTheme.primary)
                Text(title)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(AppTheme.textMuted)
            }
        }
    }
}

struct MessageCard: View {
    let title: String
    let message: String
    let buttonTitle: String?
    let action: (() -> Void)?

    init(title: String, message: String, buttonTitle: String? = nil, action: (() -> Void)? = nil) {
        self.title = title
        self.message = message
        self.buttonTitle = buttonTitle
        self.action = action
    }

    var body: some View {
        SurfaceCard {
            VStack(alignment: .leading, spacing: 12) {
                Text(title)
                    .font(.headline(20))
                    .foregroundStyle(AppTheme.text)
                Text(message)
                    .font(.system(size: 14, weight: .medium))
                    .foregroundStyle(AppTheme.textMuted)
                if let buttonTitle, let action {
                    Button(action: action) {
                        Text(buttonTitle)
                            .font(.label(12, weight: .bold))
                            .foregroundStyle(AppTheme.surfaceLowest)
                            .padding(.horizontal, 16)
                            .padding(.vertical, 10)
                            .background(AppTheme.primary)
                            .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                    }
                }
            }
        }
    }
}
