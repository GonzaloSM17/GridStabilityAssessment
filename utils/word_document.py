"""Small dependency-free writer for executive Word summary documents."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile


_WORD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_PACKAGE_RELATIONSHIPS = (
    "http://schemas.openxmlformats.org/package/2006/relationships"
)

ET.register_namespace("w", _WORD)


def write_compliance_summary_document(
    path: str | Path,
    *,
    title: str,
    subtitle: str,
    rows: tuple[tuple[str, str, int, float], ...],
    total_cases: int,
    notes: tuple[str, ...],
) -> Path:
    """Write a compact Spanish compliance summary as a standards-based DOCX."""
    output_path = Path(path)
    document = ET.Element(_tag("document"))
    body = ET.SubElement(document, _tag("body"))
    _append_paragraph(body, title, style="Title")
    _append_paragraph(body, subtitle, style="Subtitle")
    _append_paragraph(body, f"Casos analizados: {total_cases}", style="Normal")
    _append_table(
        body,
        ("Criterio", "Clasificación", "N.º de casos", "Porcentaje"),
        tuple(
            (criterion, classification, str(count), f"{percentage * 100:.1f}%")
            for criterion, classification, count, percentage in rows
        ),
    )
    _append_paragraph(body, "Notas de lectura", style="Heading1")
    for note in notes:
        _append_paragraph(body, note, style="Normal", bullet=True)
    _append_section_properties(body)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", _content_types_xml())
        archive.writestr("_rels/.rels", _root_relationships_xml())
        archive.writestr(
            "word/document.xml",
            ET.tostring(document, encoding="utf-8", xml_declaration=True),
        )
        archive.writestr("word/styles.xml", _styles_xml())
        archive.writestr("word/_rels/document.xml.rels", _document_relationships_xml())
    return output_path


def _tag(name: str) -> str:
    return f"{{{_WORD}}}{name}"


def _append_paragraph(parent, text: str, *, style: str, bullet: bool = False) -> None:
    paragraph = ET.SubElement(parent, _tag("p"))
    properties = ET.SubElement(paragraph, _tag("pPr"))
    ET.SubElement(properties, _tag("pStyle"), {_tag("val"): style})
    if bullet:
        ET.SubElement(
            properties,
            _tag("ind"),
            {_tag("left"): "420", _tag("hanging"): "210"},
        )
    run = ET.SubElement(paragraph, _tag("r"))
    text_element = ET.SubElement(run, _tag("t"))
    text_element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text_element.text = ("• " if bullet else "") + text


def _append_table(parent, headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> None:
    table = ET.SubElement(parent, _tag("tbl"))
    properties = ET.SubElement(table, _tag("tblPr"))
    ET.SubElement(properties, _tag("tblW"), {_tag("w"): "0", _tag("type"): "auto"})
    borders = ET.SubElement(properties, _tag("tblBorders"))
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        ET.SubElement(
            borders,
            _tag(edge),
            {_tag("val"): "single", _tag("sz"): "4", _tag("color"): "D9E2F3"},
        )
    grid = ET.SubElement(table, _tag("tblGrid"))
    for width in (3000, 4600, 1500, 1500):
        ET.SubElement(grid, _tag("gridCol"), {_tag("w"): str(width)})
    _append_table_row(table, headers, is_header=True)
    for row in rows:
        _append_table_row(table, row, is_header=False)


def _append_table_row(parent, values: tuple[str, ...], *, is_header: bool) -> None:
    row = ET.SubElement(parent, _tag("tr"))
    for index, value in enumerate(values):
        cell = ET.SubElement(row, _tag("tc"))
        properties = ET.SubElement(cell, _tag("tcPr"))
        if is_header:
            ET.SubElement(properties, _tag("shd"), {_tag("fill"): "4F81BD"})
        elif index == 0:
            ET.SubElement(properties, _tag("shd"), {_tag("fill"): "DCE6F1"})
        paragraph = ET.SubElement(cell, _tag("p"))
        paragraph_properties = ET.SubElement(paragraph, _tag("pPr"))
        if index >= 2:
            ET.SubElement(paragraph_properties, _tag("jc"), {_tag("val"): "center"})
        run = ET.SubElement(paragraph, _tag("r"))
        run_properties = ET.SubElement(run, _tag("rPr"))
        if is_header:
            ET.SubElement(run_properties, _tag("b"))
            ET.SubElement(run_properties, _tag("color"), {_tag("val"): "FFFFFF"})
        text_element = ET.SubElement(run, _tag("t"))
        text_element.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
        text_element.text = value


def _append_section_properties(parent) -> None:
    properties = ET.SubElement(parent, _tag("sectPr"))
    ET.SubElement(properties, _tag("pgSz"), {_tag("w"): "11906", _tag("h"): "16838"})
    ET.SubElement(
        properties,
        _tag("pgMar"),
        {
            _tag("top"): "1134",
            _tag("right"): "1134",
            _tag("bottom"): "1134",
            _tag("left"): "1134",
        },
    )


def _content_types_xml() -> bytes:
    return b'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>'''


def _root_relationships_xml() -> bytes:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{_PACKAGE_RELATIONSHIPS}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>'''.encode("utf-8")


def _document_relationships_xml() -> bytes:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="{_PACKAGE_RELATIONSHIPS}">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''.encode("utf-8")


def _styles_xml() -> bytes:
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="{_WORD}">
  <w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/></w:rPr></w:rPrDefault></w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
  <w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:rPr><w:sz w:val="32"/><w:color w:val="1F4E79"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:rPr><w:sz w:val="20"/><w:color w:val="595959"/></w:rPr></w:style>
  <w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:rPr><w:sz w:val="24"/><w:color w:val="1F4E79"/></w:rPr></w:style>
</w:styles>'''.encode("utf-8")
