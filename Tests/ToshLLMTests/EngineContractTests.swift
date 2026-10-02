import XCTest
@testable import ToshLLM

/// Two contracts that the app's own display depended on and did not hold.
///
/// The context meter compared a prompt against the window the settings asked
/// for, while the engine may have been given a different one — the router
/// rewrites the context per model, so it could report 6k of 16384k against an
/// 8K model and then blame the attachments for the overflow.
///
/// And the extra-arguments field is appended last, where llama.cpp takes the
/// last value, so it could decide who reaches the server while every label in
/// the app said otherwise.
final class EngineContractTests: XCTestCase {

    // MARK: - The context the engine reports

    private func decodeProps(_ json: String) throws -> ModelCapabilitiesService.Props {
        try JSONDecoder().decode(ModelCapabilitiesService.Props.self, from: Data(json.utf8))
    }

    /// Both spellings are in the wild, and which one arrives depends on the
    /// build. The settings block wins where both are present.
    func testContextIsReadFromEitherLocation() throws {
        let nested = try decodeProps("""
        {"default_generation_settings": {"n_ctx": 32768}}
        """)
        XCTAssertEqual(nested.contextTokens, 32768)

        let topLevel = try decodeProps(#"{"n_ctx": 8192}"#)
        XCTAssertEqual(topLevel.contextTokens, 8192)

        let both = try decodeProps("""
        {"n_ctx": 4096, "default_generation_settings": {"n_ctx": 65536}}
        """)
        XCTAssertEqual(both.contextTokens, 65536)
    }

    /// A context written as 16384.0 is the same context, not a decoding failure.
    func testAFractionalContextIsStillAContext() throws {
        let props = try decodeProps(#"{"n_ctx": 16384.0}"#)
        XCTAssertEqual(props.contextTokens, 16384)
    }

    /// One value of an unexpected type in the engine's own settings object used to
    /// fail the decode of the whole dictionary, and since that is `Props.init` it
    /// took the modalities down too — the app then reported no context *and* no
    /// capabilities for a model it could otherwise read fine.
    func testOneUnusableValueDoesNotCostTheOthersTheirContext() throws {
        let props = try decodeProps("""
        {"default_generation_settings": {"n_ctx": 32768, "stopping_criteria": "\\\\n"}}
        """)
        XCTAssertEqual(props.contextTokens, 32768, "a string sibling must not cost the context")

        let withModalities = try decodeProps("""
        {"modalities": {"vision": true, "audio": false, "video": false},
         "default_generation_settings": {"n_ctx": 16384, "template": "chatml"}}
        """)
        XCTAssertEqual(withModalities.contextTokens, 16384)
        XCTAssertEqual(withModalities.modalities?.vision, true)
    }

    /// A settings object that is not an object at all must not be fatal either.
    func testASettingsBlockOfTheWrongShapeLeavesTheContextAtTheTopLevel() throws {
        let props = try decodeProps(#"{"n_ctx": 8192, "default_generation_settings": 5}"#)
        XCTAssertEqual(props.contextTokens, 8192)
    }

    func testAnAbsentOrUnusableContextIsNilRatherThanZero() throws {
        XCTAssertNil(try decodeProps("{}").contextTokens)
        XCTAssertNil(try decodeProps(#"{"n_ctx": null}"#).contextTokens)
        XCTAssertThrowsError(try decodeProps(#"{"n_ctx": \"lots\"}"#))
    }

    // MARK: - Extra arguments cannot quietly change who reaches the server

    private func settings(extraArgs: String, apiKeyEnabled: Bool = true,
                          localNetworkDiscovery: Bool = false) -> ServerSettings {
        var s = ServerSettings(serverBinary: "/usr/bin/true", modelPath: "/tmp/m.gguf",
                               port: 8080, ngl: 99, ncmoe: 24, ctx: 16384, threads: 6,
                               flashAttn: "auto", noMmap: true, jinja: true,
                               vramReserveMB: 1024, gpuIndex: -1, extraArgs: extraArgs,
                               cacheTypeK: "f16", cacheTypeV: "f16", mlock: false)
        s.apiKeyEnabled = apiKeyEnabled
        s.localNetworkDiscovery = localNetworkDiscovery
        return s
    }

    /// The last value wins on the engine side, so the canonical one is restated
    /// at the end. Otherwise `--host 0.0.0.0` in the extra arguments exposed the
    /// server to the network while the card said "This Mac only".
    func testTheListenAddressIsRestatedWhenExtraArgumentsTryToChangeIt() {
        for extra in ["--host 0.0.0.0", "--host=0.0.0.0"] {
            let args = settings(extraArgs: extra).arguments
            XCTAssertEqual(args[args.lastIndex(of: "--host")! + 1], "127.0.0.1",
                           "\"\(extra)\" reached the engine's argument list unchanged")
            // And the app's own value is the final one, not the user's.
            XCTAssertEqual(args.lastIndex(of: "--host"), args.count - 2, "\"\(extra)\" won")
        }
    }

    func testTheListenAddressStillFollowsTheSettingWhenNothingOverridesIt() {
        var s = settings(extraArgs: "--no-warmup -np 2")
        s.localNetworkDiscovery = true
        let args = s.arguments
        let index = args.firstIndex(of: "--host")!
        XCTAssertEqual(args[index + 1], "0.0.0.0")
        XCTAssertTrue(args.contains("--no-warmup"))
        XCTAssertTrue(args.contains("-np"))
    }

    /// The key in the Keychain is what the app shows; it has to be what the
    /// engine gets.
    func testTheApiKeyIsRestatedWhenExtraArgumentsTryToReplaceIt() throws {
        let key = Keychain.apiKey()
        defer { _ = try? Keychain.delete("api-key") }
        for extra in ["--api-key=someone-elses", "--api-key someone-elses"] {
            let args = settings(extraArgs: extra).arguments
            var seen: [String] = []
            var index = 0
            while index < args.count {
                if args[index] == "--api-key", index + 1 < args.count { seen.append(args[index + 1]) }
                if args[index].hasPrefix("--api-key=") { seen.append(String(args[index].dropFirst("--api-key=".count))) }
                index += 1
            }
            // The engine takes the last value for a scalar option, so the
            // Keychain's key has to be the last one — the user's stays in the list
            // but never wins.
            XCTAssertEqual(seen.last, key, "\"\(extra)\" left \(seen.last ?? "nothing") in effect")
            XCTAssertEqual(seen.lastIndex(of: key), seen.count - 1)
        }
    }

    func testThePortIsRestatedToo() {
        var s = settings(extraArgs: "--port 9999")
        s.port = 8081
        let args = s.arguments
        XCTAssertEqual(args[args.firstIndex(of: "--port")! + 1], "8081")
        XCTAssertEqual(args.lastIndex(of: "--port"), args.count - 2)
    }

    /// A flag value that looks like another flag cannot be told apart from the
    /// flag itself, and does not need to be: restating the canonical value is the
    /// direction this function always wants. What must hold is that the canonical
    /// value is the one the engine ends up with.
    func testAFlagValueThatLooksLikeAFlagStillLeavesTheCanonicalHost() {
        let args = settings(extraArgs: "--override-kv --host=0.0.0.0").arguments
        XCTAssertTrue(args.contains("--override-kv"), "the user's own flag still applies")
        XCTAssertEqual(args[args.lastIndex(of: "--host")! + 1], "127.0.0.1")
    }

    /// With the feature off there is no key to restate, so a key typed into the
    /// field is the user's own way of turning authentication on.
    func testAManualKeySurvivesWhenTheFeatureIsOff() {
        let args = settings(extraArgs: "--api-key=typed-by-hand", apiKeyEnabled: false).arguments
        XCTAssertTrue(args.contains("--api-key=typed-by-hand"))
        XCTAssertNil(args.firstIndex(of: "--api-key"), "nothing to restate when the switch is off")
    }
}