// XCTraceExportTests.swift - Parsing `xctrace export` XML
// Copyright 2026 Vincent Gourbin

import Testing
import Foundation
@testable import MLXProfiler

@Suite("xctrace export parsing")
struct XCTraceExportTests {

    /// Trimmed from a real `Metal System Trace` export, keeping every structural
    /// feature that makes the format awkward: back-references by `id`/`ref`,
    /// a reference to an element nested inside an earlier row, and a second
    /// `<node>` whose rows rely on the schema declared in the first.
    private static let sample = """
    <?xml version="1.0"?>
    <trace-query-result>
    <node xpath='//trace-toc[1]/run[1]/data[1]/table[109]'>\
    <schema name="metal-gpu-intervals">\
    <col><mnemonic>start</mnemonic><name>Creation</name></col>\
    <col><mnemonic>duration</mnemonic><name>Duration</name></col>\
    <col><mnemonic>channel-name</mnemonic><name>Channel Name</name></col>\
    <col><mnemonic>event-label</mnemonic><name>Label</name></col>\
    <col><mnemonic>process</mnemonic><name>Process</name></col>\
    <col><mnemonic>cmdbuffer-id</mnemonic><name>Command Buffer Id</name></col>\
    </schema>\
    <row>\
    <start-time id="1" fmt="00:00.100">100000000</start-time>\
    <duration id="2" fmt="1.00 ms">1000000</duration>\
    <gpu-channel-name id="3" fmt="Compute">Compute</gpu-channel-name>\
    <formatted-label id="4" fmt="Command Buffer 0:Compute Command 0 ( qwen38 (2297) )">\
    <string id="5" fmt="Command Buffer 0:Compute Command 0">Command Buffer 0:Compute Command 0</string>\
    <process id="6" fmt="qwen38 (2297)"><pid id="7" fmt="2297">2297</pid></process>\
    </formatted-label>\
    <process ref="6"/>\
    <metal-command-buffer-id id="8" fmt="0xaa">170</metal-command-buffer-id>\
    </row>\
    </node>
    <node xpath='//trace-toc[1]/run[1]/data[1]/table[110]'>\
    <row>\
    <start-time id="9" fmt="00:00.102">102000000</start-time>\
    <duration ref="2"/>\
    <gpu-channel-name ref="3"/>\
    <formatted-label id="10" fmt="Command Buffer 1:Compute Command 3 ( other (99) )">\
    <string id="11" fmt="Command Buffer 1:Compute Command 3">Command Buffer 1:Compute Command 3</string>\
    <process id="12" fmt="other (99)"><pid id="13" fmt="99">99</pid></process>\
    </formatted-label>\
    <process ref="12"/>\
    <metal-command-buffer-id id="14" fmt="0xbb">187</metal-command-buffer-id>\
    </row>\
    </node>
    </trace-query-result>
    """

    private func parseSample() throws -> XCTraceTable {
        let tables = try XCTraceExport.parse(data: Data(Self.sample.utf8))
        return try #require(tables.first { $0.schema == "metal-gpu-intervals" })
    }

    @Test func testSchemaAndRowsAreRecovered() throws {
        let table = try parseSample()
        #expect(table.columns == ["start", "duration", "channel-name", "event-label", "process", "cmdbuffer-id"])
        // Rows from the second node count too, even though it declares no schema.
        #expect(table.rows.count == 2)
    }

    @Test func testColumnsAreAddressedByMnemonic() throws {
        let table = try parseSample()
        let row = table.rows[0]
        #expect(table.value("start", in: row)?.integerValue == 100_000_000)
        #expect(table.value("duration", in: row)?.integerValue == 1_000_000)
        #expect(table.value("channel-name", in: row)?.fmt == "Compute")
        #expect(table.value("cmdbuffer-id", in: row)?.fmt == "0xaa")
        #expect(table.value("nonexistent", in: row) == nil)
    }

    /// The second row reuses `duration` and `channel-name` by reference only.
    @Test func testBackReferencesResolve() throws {
        let table = try parseSample()
        let second = table.rows[1]
        #expect(table.value("duration", in: second)?.integerValue == 1_000_000)
        #expect(table.value("channel-name", in: second)?.fmt == "Compute")
        #expect(table.value("start", in: second)?.integerValue == 102_000_000)
    }

    /// `<process ref="6"/>` points at a `<process>` defined *inside* the label
    /// of the same row, so interning has to reach nested elements.
    @Test func testReferenceToNestedElementResolves() throws {
        let table = try parseSample()
        let pid = table.value("process", in: table.rows[0])?
            .firstDescendant(named: "pid")?.integerValue
        #expect(pid == 2297)

        let otherPid = table.value("process", in: table.rows[1])?
            .firstDescendant(named: "pid")?.integerValue
        #expect(otherPid == 99)
    }

    @Test func testMalformedInputThrows() {
        #expect(throws: (any Error).self) {
            try XCTraceExport.parse(data: Data("<trace-query-result><node>".utf8))
        }
    }
}

@Suite("GPU interval aggregation")
struct GPUIntervalAggregationTests {

    private func interval(_ start: UInt64, _ duration: UInt64, label: String = "Command Buffer 0:Compute Command 0")
        -> GPUKernelInterval {
        GPUKernelInterval(startUs: start, durationUs: duration, label: label,
                          channel: "Compute", commandBufferId: nil)
    }

    /// Overlapping and nested encoders must count once. Summing durations instead
    /// is what reported the Flash-Next bench at 85.4 % GPU-busy when the union
    /// says 75.7 %.
    @Test func testUnionCountsOverlapOnce() {
        let overlapping = [interval(0, 100), interval(50, 100), interval(120, 30)]
        #expect(MetalSystemTrace.unionDuration(of: overlapping) == 150)

        let nested = [interval(0, 100), interval(10, 20), interval(40, 10)]
        #expect(MetalSystemTrace.unionDuration(of: nested) == 100)
    }

    @Test func testUnionHandlesGapsAndUnsortedInput() {
        let disjoint = [interval(200, 50), interval(0, 100)]
        #expect(MetalSystemTrace.unionDuration(of: disjoint) == 150)
        #expect(MetalSystemTrace.unionDuration(of: []) == 0)
        #expect(MetalSystemTrace.unionDuration(of: [interval(5, 7)]) == 7)
    }

    /// Counters differ between encoders of the same kind; the family must not.
    @Test func testFamilyStripsCounters() {
        #expect(MetalSystemTrace.family(of: "Command Buffer 12:Compute Command 0") == "Compute Command")
        #expect(MetalSystemTrace.family(of: "Command Buffer 3:Compute Command 41") == "Compute Command")
        #expect(MetalSystemTrace.family(of: "Command Buffer 0:Blit Command 0 ( app (12) )") == "Blit Command")
        #expect(MetalSystemTrace.family(of: "") == "unnamed")
    }
}
