// XCTraceExport.swift - Parser for `xctrace export` XML
// Copyright 2026 Vincent Gourbin
//
// The format has three properties worth knowing before reading this file:
//
//   1. A row's children are its columns, positionally, matching the <col> list
//      in the table's <schema>. Columns are not named in the row itself.
//   2. Any element may carry an `id`, and any later element may replace itself
//      with `ref="<id>"` to reuse that value. References are global to the
//      document and can point at elements nested deep inside an earlier row, so
//      every element with an id has to be interned, at every depth.
//   3. A <schema> is emitted once and then applies to the rows of the following
//      <node>s until the next <schema>, which is why the schema is carried
//      forward rather than looked up per node.

import Foundation

/// One element of an xctrace export row.
public final class XCTraceNode {
    public let element: String
    /// The `fmt` attribute — the display string Instruments would show.
    public let fmt: String?
    public internal(set) var text: String = ""
    public internal(set) var children: [XCTraceNode] = []

    internal let id: String?
    internal let ref: String?

    init(element: String, fmt: String? = nil, id: String? = nil, ref: String? = nil) {
        self.element = element
        self.fmt = fmt
        self.id = id
        self.ref = ref
    }

    /// Depth-first search for the first descendant (or self) with this element name.
    public func firstDescendant(named name: String) -> XCTraceNode? {
        if element == name { return self }
        for child in children {
            if let found = child.firstDescendant(named: name) { return found }
        }
        return nil
    }

    /// Text parsed as an integer — xctrace writes raw nanoseconds and ids here,
    /// with the human-readable form in `fmt`.
    public var integerValue: Int64? { Int64(text.trimmingCharacters(in: .whitespacesAndNewlines)) }
}

/// One table from an export: a schema plus its rows.
public struct XCTraceTable {
    public let schema: String
    /// Column mnemonics, in the order row children appear.
    public let columns: [String]
    /// Each row's top-level column values, positionally aligned with ``columns``.
    public let rows: [[XCTraceNode]]

    /// The value in `column` for `row`, or nil if this table has no such column.
    public func value(_ column: String, in row: [XCTraceNode]) -> XCTraceNode? {
        guard let index = columns.firstIndex(of: column), index < row.count else { return nil }
        return row[index]
    }
}

public enum XCTraceExportError: Error, CustomStringConvertible {
    case malformedXML(String)

    public var description: String {
        switch self {
        case .malformedXML(let detail): return "Could not parse xctrace export: \(detail)"
        }
    }
}

public enum XCTraceExport {

    /// Parses an `xctrace export` document into its tables.
    public static func parse(data: Data) throws -> [XCTraceTable] {
        let parser = XMLParser(data: data)
        let delegate = Delegate()
        parser.delegate = delegate
        guard parser.parse() else {
            throw XCTraceExportError.malformedXML(
                parser.parserError?.localizedDescription ?? "unknown error")
        }
        delegate.flushTable()
        return delegate.tables
    }

    // MARK: - SAX delegate

    private final class Delegate: NSObject, XMLParserDelegate {
        var tables: [XCTraceTable] = []

        private var schemaName: String?
        private var columns: [String] = []
        private var rows: [[XCTraceNode]] = []

        private var inSchema = false
        private var inColumn = false
        private var capturingMnemonic = false
        private var mnemonicBuffer = ""

        private var interned: [String: XCTraceNode] = [:]
        private var stack: [XCTraceNode] = []
        private var inRow: Bool { !stack.isEmpty }

        /// Emits the table accumulated so far, if it has any rows.
        func flushTable() {
            if let schemaName, !rows.isEmpty {
                tables.append(XCTraceTable(schema: schemaName, columns: columns, rows: rows))
            }
            rows = []
        }

        func parser(
            _ parser: XMLParser, didStartElement elementName: String,
            namespaceURI: String?, qualifiedName: String?, attributes: [String: String] = [:]
        ) {
            switch elementName {
            case "schema":
                // A new schema closes out whatever rows belonged to the old one.
                flushTable()
                schemaName = attributes["name"]
                columns = []
                inSchema = true

            case "col" where inSchema:
                inColumn = true

            case "mnemonic" where inColumn:
                capturingMnemonic = true
                mnemonicBuffer = ""

            case "row":
                stack = [XCTraceNode(element: "row")]

            default:
                guard inRow else { return }
                stack.append(XCTraceNode(
                    element: elementName, fmt: attributes["fmt"],
                    id: attributes["id"], ref: attributes["ref"]))
            }
        }

        func parser(_ parser: XMLParser, foundCharacters string: String) {
            if capturingMnemonic { mnemonicBuffer += string }
            else if let top = stack.last, stack.count > 1 { top.text += string }
        }

        func parser(
            _ parser: XMLParser, didEndElement elementName: String,
            namespaceURI: String?, qualifiedName: String?
        ) {
            switch elementName {
            case "schema":
                inSchema = false

            case "col" where inSchema:
                inColumn = false

            case "mnemonic" where capturingMnemonic:
                capturingMnemonic = false
                columns.append(mnemonicBuffer.trimmingCharacters(in: .whitespacesAndNewlines))

            case "row":
                guard let root = stack.first else { return }
                rows.append(root.children)
                stack = []

            default:
                guard stack.count > 1 else { return }
                let node = stack.removeLast()
                if let id = node.id { interned[id] = node }
                // A `ref` element carries no content of its own; it stands in for
                // the interned value. If the id is unknown, keep the empty node so
                // column positions stay aligned.
                let resolved = node.ref.flatMap { interned[$0] } ?? node
                stack.last?.children.append(resolved)
            }
        }
    }
}
