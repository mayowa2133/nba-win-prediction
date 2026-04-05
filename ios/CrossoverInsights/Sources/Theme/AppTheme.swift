import SwiftUI

enum AppTheme {
    static let background = Color(hex: "#0E131E")
    static let surface = Color(hex: "#1B1F2B")
    static let surfaceLow = Color(hex: "#171B27")
    static let surfaceLowest = Color(hex: "#090E19")
    static let surfaceHigh = Color(hex: "#252A36")
    static let surfaceHighest = Color(hex: "#303541")
    static let surfaceBright = Color(hex: "#343946")
    static let primary = Color(hex: "#99DA00")
    static let primaryDeep = Color(hex: "#5F8900")
    static let text = Color(hex: "#DEE2F2")
    static let textMuted = Color(hex: "#C6C6CC")
    static let outline = Color(hex: "#45464C")
    static let error = Color(hex: "#FFB4AB")
    static let warning = Color(hex: "#FFB866")

    static let cardRadius: CGFloat = 22
}

extension Color {
    init(hex: String) {
        let hexString = hex.replacingOccurrences(of: "#", with: "")
        let value = UInt64(hexString, radix: 16) ?? 0
        let r = Double((value & 0xFF0000) >> 16) / 255
        let g = Double((value & 0x00FF00) >> 8) / 255
        let b = Double(value & 0x0000FF) / 255
        self.init(red: r, green: g, blue: b)
    }
}

extension Font {
    static func headline(_ size: CGFloat) -> Font {
        .system(size: size, weight: .black, design: .rounded)
    }

    static func label(_ size: CGFloat, weight: Font.Weight = .semibold) -> Font {
        .system(size: size, weight: weight, design: .rounded)
    }
}
