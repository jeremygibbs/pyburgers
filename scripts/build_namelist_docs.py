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
        return "<br>".join(options) if options else "none"
    if "enum" in schema:
        return "<br>".join(str(value) for value in schema["enum"])
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


def format_table(entries: list[dict]) -> str:
    lines = [
        "| Key | Description | Options | Required |",
        "| --- | --- | --- | --- |",
    ]
    for entry in entries:
        required = "yes" if entry["required"] else "no"
        key_cell = entry.get("key", entry["path"])
        lines.append(
            "| {path} | {description} | {options} | {required} |".format(
                path=key_cell,
                description=entry["description"],
                options=entry["options"],
                required=required,
            )
        )
    return "\n".join(lines)


def group_entries_by_subsection(entries: list[dict], section: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    order: list[str] = []
    for entry in entries:
        parts = entry["path"].split(".")
        if len(parts) >= 3:
            subgroup = f"{section}.{parts[1]}"
            entry_key = ".".join(parts[2:])
        else:
            subgroup = section
            entry_key = parts[1] if len(parts) > 1 else entry["path"]

        if subgroup not in grouped:
            grouped[subgroup] = []
            order.append(subgroup)

        entry_with_key = dict(entry)
        entry_with_key["key"] = entry_key
        grouped[subgroup].append(entry_with_key)

    return {name: grouped[name] for name in order}


def build_markdown(schema: dict, example_text: str) -> str:
    required_sections = schema.get("required", [])
    entries = list(walk_schema(schema))
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
        section_entries = [entry for entry in entries if entry["path"].startswith(f"{section}.")]
        if not section_entries:
            continue
        lines.append(f"### {section}")
        lines.append("")

        grouped_sections = group_entries_by_subsection(section_entries, section)
        if section in grouped_sections:
            lines.extend([format_table(grouped_sections[section]), ""])

        for subgroup, subgroup_entries in grouped_sections.items():
            if subgroup == section:
                continue
            lines.extend([f"#### {subgroup}", "", format_table(subgroup_entries), ""])

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    schema = load_json(SCHEMA_PATH)
    example_text = read_text(NAMELIST_PATH)
    markdown = build_markdown(schema, example_text)
    OUTPUT_PATH.write_text(markdown, encoding="utf-8")


if __name__ == "__main__":
    main()
