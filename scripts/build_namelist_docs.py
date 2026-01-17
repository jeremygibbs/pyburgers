from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "pyburgers" / "schema_namelist.json"
NAMELIST_PATH = ROOT / "namelist.json"
OUTPUT_PATH = ROOT / "docs" / "namelist.md"

SCHEMA_REL = "pyburgers/schema_namelist.json"
NAMELIST_REL = "namelist.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").rstrip()


def describe_constraints(schema: dict) -> str:
    constraints = []
    for key in ("minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"):
        if key in schema:
            constraints.append(f"{key}: {schema[key]}")
    return "; ".join(constraints) if constraints else "none"


def extract_options(schema: dict) -> str:
    if "oneOf" in schema:
        options = []
        for entry in schema["oneOf"]:
            if "const" in entry:
                value = entry["const"]
                description = entry.get("description")
                if description:
                    options.append(f"{value}: {description}")
                else:
                    options.append(str(value))
            elif "enum" in entry:
                for value in entry["enum"]:
                    description = entry.get("description")
                    if description:
                        options.append(f"{value}: {description}")
                    else:
                        options.append(str(value))
        return "; ".join(options) if options else "none"
    if "enum" in schema:
        return ", ".join(str(value) for value in schema["enum"])
    return "none"


def walk_schema(schema: dict, prefix: str = "", parent_required: bool = True):
    properties = schema.get("properties", {})
    required_props = set(schema.get("required", []))

    for name, subschema in properties.items():
        path = f"{prefix}.{name}" if prefix else name
        is_required = parent_required and (name in required_props)
        if subschema.get("type") == "object" and "properties" in subschema:
            yield from walk_schema(subschema, path, is_required)
        else:
            yield {
                "path": path,
                "type": subschema.get("type", "any"),
                "required": is_required,
                "constraints": describe_constraints(subschema),
                "description": subschema.get("description", "none"),
                "options": extract_options(subschema),
            }


def group_entries(entries: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for entry in entries:
        section = entry["path"].split(".", 1)[0]
        grouped.setdefault(section, []).append(entry)
    return grouped


def format_table(entries: list[dict]) -> str:
    lines = [
        "| Key | Type | Required | Description | Options | Constraints |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        required = "yes" if entry["required"] else "no"
        lines.append(
            "| {path} | {type} | {required} | {description} | {options} | {constraints} |".format(
                path=entry["path"],
                type=entry["type"],
                required=required,
                description=entry["description"],
                options=entry["options"],
                constraints=entry["constraints"],
            )
        )
    return "\n".join(lines)


def build_markdown(schema: dict, example_text: str) -> str:
    required_sections = schema.get("required", [])
    entries = list(walk_schema(schema))
    grouped = group_entries(entries)

    lines = [
        "# Namelist",
        "",
        "This page documents the namelist configuration format and available options.",
        "",
        "## Source files",
        "",
        f"- `{NAMELIST_REL}`: example configuration",
        f"- `{SCHEMA_REL}`: JSON schema used for validation",
        "",
        "## Example",
        "",
        "```json",
        example_text,
        "```",
        "",
        "## Reference",
        "",
        "Required top-level sections: "
        + ", ".join(f"`{section}`" for section in required_sections)
        + ".",
        "",
    ]

    for section in required_sections:
        section_entries = grouped.get(section, [])
        if not section_entries:
            continue
        lines.extend([f"### {section}", "", format_table(section_entries), ""])

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    schema = load_json(SCHEMA_PATH)
    example_text = read_text(NAMELIST_PATH)
    markdown = build_markdown(schema, example_text)
    OUTPUT_PATH.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
