"""Inspect logical FileGDB layers and cross-reference supplied FGDC XML."""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pyogrio

GDB = Path("data/interim/silvis_wui_2020/CONUS_WUI_block_1990_2020_change_v4_gcs_na83.gdb")
METADATA = Path("data/raw/silvis_wui_2020/CONUS_WUI_block_1990_2020_change_v4_metadata.xml")


def metadata(root: Path) -> dict:
    tree = ET.parse(root / METADATA)
    attributes = {}
    for attr in tree.findall(".//attr"):
        label = attr.findtext("attrlabl")
        attributes[label] = {
            "definition": attr.findtext("attrdef"),
            "source": attr.findtext("attrdefs"),
            "domains": [
                {child.tag: child.text for child in domain}
                for domain in attr.findall("attrdomv/edom")
            ],
            "unenumerated_domain": attr.findtext("attrdomv/udom"),
        }
    # Keep year mentions with XML paths: publication/lineage years are not observations.
    mentions = []

    def visit(node, path):
        text = node.text or ""
        if node.tag in {"pubdate", "begdate", "enddate", "procdate", "metd"}:
            text = text[:4]
        years = sorted(set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text)))
        if years:
            mentions.append({"xml_path": path, "years": years, "text": node.text.strip()})
        for child in node:
            visit(child, f"{path}/{child.tag}")

    visit(tree.getroot(), "metadata")
    return {
        "source": METADATA.as_posix(), "attributes": attributes,
        "year_mentions": mentions,
        "process_steps": [p.text for p in tree.findall(".//procdesc")],
        "documented_bounds": {
            n.tag: float(n.text) for n in tree.findall("./idinfo/spdom/bounding/*")
        },
    }


def inventory(root: Path, definitions: dict) -> tuple[list[dict], list[dict]]:
    layers, fields = [], []
    lookup = {name.casefold(): name for name in definitions}
    for name, _ in pyogrio.list_layers(root / GDB):
        info = pyogrio.read_info(
            root / GDB, layer=name, force_feature_count=False, force_total_bounds=False,
        )
        layers.append({
            key: info[key] for key in (
                "layer_name", "crs", "geometry_type", "fid_column", "geometry_name",
                "features", "total_bounds", "driver", "capabilities",
            )
        })
        for field, dtype, ogr, subtype in zip(
            info["fields"], info["dtypes"], info["ogr_types"], info["ogr_subtypes"], strict=True,
        ):
            match = lookup.get(field.casefold())
            years = sorted(set(re.findall(r"(?:19|20)\d{2}", field)))
            fields.append({
                "layer": name, "field": field, "dtype": dtype,
                "ogr_type": ogr, "ogr_subtype": subtype,
                "schema_years": ";".join(years), "metadata_label": match or "",
                "definition": definitions.get(match, {}).get("definition", ""),
            })
    return layers, fields
