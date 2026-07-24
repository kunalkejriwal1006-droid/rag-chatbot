"""Per-document metadata. Hand-curated per file since guessing product names
from messy PDF text isn't reliable; new PDFs fall back to heuristics + a UIN
regex. All documents here are Bajaj Allianz Two-Wheeler motor products, so
insurer/lob/vehicle_type stay constant.
"""
import re

INSURER = "Bajaj Allianz General Insurance (Bajaj General Insurance Ltd.)"
LOB = "motor"
VEHICLE_TYPE = "Two Wheeler"

UIN_RE = re.compile(r"UIN[:\s]*([A-Za-z0-9/\.\-]{8,40})")

# brochure/policy-wording pairs for the same product share a UIN
# (CO_1/CO_2, CO_8/CO_9)
DOCUMENT_METADATA = {
    "BAJAJ ALLIANZ GENERAL INSURANCE CO_1.pdf": {
        "product": "Two Wheeler Package Policy",
        "document_type": "Brochure",
        "uin": "IRDAN113RP0026V01200102",
    },
    "BAJAJ ALLIANZ GENERAL INSURANCE CO_2.pdf": {
        "product": "Two Wheeler Package Policy",
        "document_type": "Policy Wording",
        "uin": "IRDAN113RP0026V01200102",
    },
    "BAJAJ ALLIANZ GENERAL INSURANCE CO_3.pdf": {
        "product": "Two Wheeler Policy - Bundled",
        "document_type": "Policy Wording",
        "uin": "IRDAN113RP0008V01201819",
    },
    "BAJAJ ALLIANZ GENERAL INSURANCE CO_4.pdf": {
        "product": "Liability Only Policy",
        "document_type": "Policy Wording",
        "uin": None,
    },
    "BAJAJ ALLIANZ GENERAL INSURANCE CO_5.pdf": {
        "product": "Liability Only Policy for Two Wheelers - 5 Years",
        "document_type": "Policy Wording",
        "uin": "IRDAN113RP0004V01201819",
    },
    "BAJAJ ALLIANZ GENERAL INSURANCE CO_6.pdf": {
        "product": "Two Wheeler Package Policy - 5 Years",
        "document_type": "Policy Wording",
        "uin": "IRDAN113RPMT0018V01202425",
    },
    "BAJAJ ALLIANZ GENERAL INSURANCE CO_7.pdf": {
        "product": "Standalone Own Damage Cover for Two Wheeler - Long Term",
        "document_type": "Policy Wording",
        "uin": "IRDAN113RP0002V02201920",
    },
    "BAJAJ ALLIANZ GENERAL INSURANCE CO_8.pdf": {
        "product": "Long Term Two Wheeler Package Policy",
        "document_type": "Brochure",
        "uin": "IRDAN113RP0008V01201617",
    },
    "BAJAJ ALLIANZ GENERAL INSURANCE CO_9.pdf": {
        "product": "Long Term Two Wheeler Package Policy",
        "document_type": "Policy Wording",
        "uin": "IRDAN113RP0008V01201617",
    },
    "BAJAJ ALLIANZ GENERAL INSURANCE CO_10.pdf": {
        "product": "Long Term Two Wheeler Insurance Policy (Liability Only)",
        "document_type": "Policy Wording",
        "uin": "BAL-MT-P15-18-V01-14-15",
    },
    "BAJAJ ALLIANZ GENERAL INSURANCE CO_11.pdf": {
        "product": "Long Term Two Wheeler Package Policy - Add-on Covers",
        "document_type": "Endorsement Library",
        "uin": None,
    },
    "BAJAJ ALLIANZ GENERAL INSURANCE CO_12.pdf": {
        "product": "Two Wheeler Package Policy - Add-on Covers",
        "document_type": "Endorsement Library",
        "uin": None,
    },
    "BAJAJ ALLIANZ GENERAL INSURANCE CO_13.pdf": {
        "product": "Two Wheeler Policy - Bundled - Add-on Covers",
        "document_type": "Endorsement Library",
        "uin": None,
    },
}


def get_base_metadata(filename: str, sample_text: str = "") -> dict:
    """Return document-level metadata for a PDF filename, falling back to
    heuristics for files not in DOCUMENT_METADATA."""
    override = DOCUMENT_METADATA.get(filename)
    uin = None
    if sample_text:
        m = UIN_RE.search(sample_text)
        if m:
            uin = m.group(1)

    if override:
        product = override["product"]
        document_type = override["document_type"]
        uin = override["uin"] or uin
    else:
        product = filename.rsplit(".", 1)[0]
        lower = sample_text.lower()
        if "endorsement" in lower or "add-on" in lower or "add on cover" in lower:
            document_type = "Endorsement Library"
        elif "policy wording" in lower or "whereas the insured" in lower:
            document_type = "Policy Wording"
        else:
            document_type = "Brochure"

    return {
        "insurer": INSURER,
        "lob": LOB,
        "vehicle_type": VEHICLE_TYPE,
        "product": product,
        "document_type": document_type,
        "uin": uin,
        "source_file": filename,
    }
