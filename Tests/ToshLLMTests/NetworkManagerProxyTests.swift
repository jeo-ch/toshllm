// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class NetworkManagerProxyTests: XCTestCase {
    private var saved: [(String, Any?)] = []

    override func setUp() {
        super.setUp()
        let defaults = UserDefaults.standard
        saved = [
            (SettingsKeys.proxyEnabled, defaults.object(forKey: SettingsKeys.proxyEnabled)),
            (SettingsKeys.proxyHost, defaults.object(forKey: SettingsKeys.proxyHost)),
            (SettingsKeys.proxyPort, defaults.object(forKey: SettingsKeys.proxyPort)),
        ]
    }

    override func tearDown() {
        let defaults = UserDefaults.standard
        for (key, value) in saved {
            if let value {
                defaults.set(value, forKey: key)
            } else {
                defaults.removeObject(forKey: key)
            }
        }
        saved = []
        super.tearDown()
    }

    func testProxyConfigurationExemptsLoopbackAndLocalHosts() throws {
        let defaults = UserDefaults.standard
        defaults.set(true, forKey: SettingsKeys.proxyEnabled)
        defaults.set("proxy.example", forKey: SettingsKeys.proxyHost)
        defaults.set(8080, forKey: SettingsKeys.proxyPort)

        let config = NetworkManager.makeConfiguration()
        let dictionary = try XCTUnwrap(config.connectionProxyDictionary)
        let exceptions = dictionary[AnyHashable(kCFNetworkProxiesExceptionsList as String)] as? [String]

        XCTAssertEqual(Set(exceptions ?? []),
                       Set(["localhost", "127.0.0.1", "::1", "*.local"]))
        XCTAssertEqual(dictionary[AnyHashable(kCFNetworkProxiesHTTPProxy as String)] as? String,
                       "proxy.example")
        XCTAssertEqual(dictionary[AnyHashable(kCFNetworkProxiesHTTPPort as String)] as? Int, 8080)
    }

    func testDisabledProxyLeavesTheSessionWithoutAProxyDictionary() {
        UserDefaults.standard.set(false, forKey: SettingsKeys.proxyEnabled)
        XCTAssertNil(NetworkManager.makeConfiguration().connectionProxyDictionary)
    }
}
