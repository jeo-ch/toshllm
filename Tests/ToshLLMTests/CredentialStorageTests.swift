import XCTest
@testable import ToshLLM

/// Two credentials, both long-lived, both previously readable by any process
/// running as this user. The API key was in the Keychain but could not be changed
/// without leaving the app; the archive hook's bearer token was in a plist.
final class CredentialStorageTests: XCTestCase {

    override func tearDown() {
        // Neither item belongs to this machine; do not leave it behind.
        Keychain.delete("api-key")
        Keychain.delete("memoryArchiveHookSecret")
        UserDefaults.standard.removeObject(forKey: SettingsKeys.memoryArchiveHookSecret)
        super.tearDown()
    }

    // MARK: - The API key can be replaced

    func testTheApiKeyIsStableUntilItIsDeliberatelyReplaced() throws {
        let first = Keychain.apiKey()
        XCTAssertEqual(Keychain.apiKey(), first, "the key must not change just because it was read")
        XCTAssertEqual(first.count, 32)

        let second = Keychain.regenerateAPIKey()
        XCTAssertNotEqual(second, first, "a regenerated key has to actually differ")
        XCTAssertEqual(second.count, 32)
        XCTAssertEqual(Keychain.apiKey(), second, "the new key is the one in use afterwards")
    }

    /// Without a rotate the only recovery from a leaked key was deleting it by hand
    /// in Keychain Access, after which the next launch silently minted a
    /// replacement and every configured client started answering 401.
    func testRegeneratingTwiceProducesDistinctKeys() {
        XCTAssertNotEqual(Keychain.regenerateAPIKey(), Keychain.regenerateAPIKey())
    }

    // MARK: - The archive hook's bearer token is not in a plist

    func testTheArchiveSecretRoundTripsThroughTheKeychain() {
        XCTAssertEqual(MemoryArchiveHook.storeSecret("  bearer-token-xyz  "), "bearer-token-xyz",
                       "surrounding whitespace is trimmed on the way in")
        XCTAssertEqual(MemoryArchiveHook.currentSecret(), "bearer-token-xyz")
    }

    func testStoringTheArchiveSecretLeavesNothingInUserDefaults() {
        MemoryArchiveHook.storeSecret("bearer-token-xyz")
        XCTAssertNil(UserDefaults.standard.string(forKey: SettingsKeys.memoryArchiveHookSecret),
                     "a bearer token in a plist is readable by any process as this user")
        XCTAssertEqual(Keychain.get("memoryArchiveHookSecret"), "bearer-token-xyz")
    }

    /// A token typed before the move still works, and is taken off the plist on
    /// first read so nobody has to retype it and nothing is left behind.
    func testALegacySecretInUserDefaultsIsMigratedOnce() throws {
        UserDefaults.standard.removeObject(forKey: SettingsKeys.memoryArchiveHookSecret)
        Keychain.delete("memoryArchiveHookSecret")
        UserDefaults.standard.set("legacy-token", forKey: SettingsKeys.memoryArchiveHookSecret)

        XCTAssertEqual(MemoryArchiveHook.currentSecret(), "legacy-token")
        XCTAssertNil(UserDefaults.standard.string(forKey: SettingsKeys.memoryArchiveHookSecret),
                     "the migration has to remove the plaintext copy")
        XCTAssertEqual(Keychain.get("memoryArchiveHookSecret"), "legacy-token")
    }

    /// Once migrated, the Keychain is the only source: changing the plist behind
    /// the app's back must not take effect.
    func testTheKeychainWinsOverThePlist() {
        MemoryArchiveHook.storeSecret("from-keychain")
        UserDefaults.standard.set("from-plist", forKey: SettingsKeys.memoryArchiveHookSecret)
        XCTAssertEqual(MemoryArchiveHook.currentSecret(), "from-keychain")
        _ = MemoryArchiveHook.currentSecret()   // the read also tidies the plist up
        XCTAssertNil(UserDefaults.standard.string(forKey: SettingsKeys.memoryArchiveHookSecret))
    }

    func testClearingTheArchiveSecretRemovesItEverywhere() {
        MemoryArchiveHook.storeSecret("bearer-token-xyz")
        MemoryArchiveHook.storeSecret("   ")
        XCTAssertNil(Keychain.get("memoryArchiveHookSecret"))
        XCTAssertNil(UserDefaults.standard.string(forKey: SettingsKeys.memoryArchiveHookSecret))
        XCTAssertEqual(MemoryArchiveHook.currentSecret(), "")
    }

    func testAnEmptySecretIsTreatedAsAbsent() {
        Keychain.delete("memoryArchiveHookSecret")
        UserDefaults.standard.removeObject(forKey: SettingsKeys.memoryArchiveHookSecret)
        XCTAssertEqual(MemoryArchiveHook.currentSecret(), "")
        MemoryArchiveHook.storeSecret("")
        XCTAssertEqual(MemoryArchiveHook.currentSecret(), "")
    }
}