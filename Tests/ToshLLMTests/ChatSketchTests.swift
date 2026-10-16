// ToshLLM - run LLMs locally on Intel Macs with AMD GPUs
// Copyright (C) 2026 Engelbert Delgado <engeldlgado@gmail.com>
// SPDX-License-Identifier: GPL-3.0-or-later

import XCTest
@testable import ToshLLM

final class ChatSketchTests: XCTestCase {
    func testAnAgentsFigureIsReadAndMeasured() throws {
        let sketch = try XCTUnwrap(ChatSketch(["title": "Tract", "unit": "ft", "closed": true,
                                               "points": [[0, 0], [737.85, -426], [613.56, -649.46], [-123.25, -224.06], [0.06, -0.06]]]))
        XCTAssertEqual(sketch.unit, "ft")
        XCTAssertEqual(sketch.gap, hypot(0.06, 0.06), accuracy: 1e-9)
        XCTAssertEqual(sketch.scaleLength, 200)
        XCTAssertNil(sketch.corners)
    }

    func testWhatCannotBeDrawnIsRefused() {
        XCTAssertNil(ChatSketch(["points": [[0, 0]]]), "one point is not a figure")
        XCTAssertNil(ChatSketch(["points": [[0, 0], ["a", 1]]]))
        XCTAssertNil(ChatSketch(["points": [[0, 0], [1, 1]], "corners": [[0]]]))
        XCTAssertNil(ChatSketch(["points": Array(repeating: [0, 0], count: ChatSketch.maximumPoints + 1)]))
    }

    func testAFigureExportsForCADAndAsCorners() {
        let open = ChatSketch(title: nil, unit: "ft", points: [[0, 0], [100, 0], [100, 100], [1, 2]], closed: true)
        let dxf = open.dxf().components(separatedBy: "\n")
        XCTAssertEqual(dxf.filter { $0 == "VERTEX" }.count, 4)
        XCTAssertEqual(dxf.filter { $0 == "TEXT" }.count, 3, "the end that does not close is not a corner")
        XCTAssertTrue(dxf.contains("MISCLOSURE"))
        XCTAssertEqual(Array(dxf.suffix(3)), ["0", "EOF", ""])
        XCTAssertEqual(open.csv(), "point,east_ft,north_ft\n1,0.000,0.000\n2,100.000,0.000\n3,100.000,100.000\n")
        let shut = ChatSketch(title: nil, unit: "ft", points: [[0, 0], [100, 0], [100, 100], [0, 0]], closed: true)
        XCTAssertFalse(shut.dxf().contains("MISCLOSURE"))
    }

    func testAMapExportsOneFilePerKindOfGeometry() throws {
        let collection: [String: Any] = ["type": "FeatureCollection", "features": [
            ["type": "Feature", "geometry": ["type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]], "properties": ["name": "tract"]],
            ["type": "Feature", "geometry": ["type": "Point", "coordinates": [0, 0]], "properties": ["name": "corner 1"]],
            ["type": "Feature", "geometry": ["type": "LineString", "coordinates": [[0, 0], [1, 1]]], "properties": [:]]]]
        let points: [String: Any] = ["type": "FeatureCollection", "features": [
            ["type": "Feature", "geometry": ["type": "Point", "coordinates": [2, 2]], "properties": [:]]]]
        let map = ChatMap(layers: [
            ChatMap.Layer(title: "Tract: 5 acres", geojson: try JSONSerialization.data(withJSONObject: collection), complete: true),
            ChatMap.Layer(title: "Wells", geojson: try JSONSerialization.data(withJSONObject: points), complete: true)])
        let files = map.exportFiles()
        XCTAssertEqual(files.map(\.name), ["Tract- 5 acres - polygons.geojson", "Tract- 5 acres - lines.geojson",
                                            "Tract- 5 acres - points.geojson", "Wells.geojson"])
        let polygons = try XCTUnwrap(JSONSerialization.jsonObject(with: files[0].data) as? [String: Any])
        let features = try XCTUnwrap(polygons["features"] as? [[String: Any]])
        XCTAssertEqual(features.count, 1)
        XCTAssertEqual((features[0]["properties"] as? [String: Any])?["layer"] as? String, "Tract: 5 acres")
    }

    func testAFigureWithAnOriginExportsAtItsCoordinates() throws {
        let sketch = try XCTUnwrap(ChatSketch(["points": [[0, 0], [0, 100], [100, 100], [0, 0]], "unit": "ft", "closed": true,
                                               "origin": [2077075.85, 303260.44],
                                               "plane": "NAD27 Texas North Central (4202), US ft"]))
        XCTAssertEqual(sketch.plane, "NAD27 Texas North Central (4202), US ft")
        let csv = sketch.csv().split(separator: "\n")
        XCTAssertEqual(csv.first, "point,x_ft,y_ft")
        XCTAssertEqual(csv[2], "2,2077075.850,303360.440")
        XCTAssertTrue(sketch.dxf().contains("10\n2077175.850\n20\n303360.440"))
        // without an origin a figure stays in its own plane
        let plain = try XCTUnwrap(ChatSketch(["points": [[0, 0], [0, 100], [0, 0]], "unit": "ft", "plane": "ignored"]))
        XCTAssertNil(plain.plane)
        XCTAssertEqual(plain.csv().split(separator: "\n").first, "point,east_ft,north_ft")
    }

    func testTheFigureStaysWithTheAnswerItCameWith() throws {
        var message = ChatMessage(role: "assistant", content: "It closes.")
        message.sketches = [ChatSketch(title: "Tract 1", unit: "ft", points: [[0, 0], [10, 0], [10, 10], [0, 0]], closed: true),
                            ChatSketch(title: "Tract 2", unit: "ft", points: [[0, 0], [5, 0], [5, 5], [0, 0]], closed: true)]
        let saved = try JSONDecoder().decode(ChatMessage.self, from: JSONEncoder().encode(message))
        XCTAssertEqual(saved.sketches, message.sketches)
        let older = try JSONDecoder().decode(ChatMessage.self, from: JSONEncoder().encode(ChatMessage(role: "assistant", content: "x")))
        XCTAssertNil(older.sketches)
    }
}
