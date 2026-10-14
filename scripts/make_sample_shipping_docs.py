"""Generate fictional sample shipping documents for the extraction demo.

    python scripts/make_sample_shipping_docs.py

All companies, vessels and container numbers are fictional. ``bl_chemicals_errors``
deliberately contains a bad container check digit and an overweight container.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.extraction.iso6346 import check_digit  # noqa: E402

_spec = importlib.util.spec_from_file_location("mkpdf", ROOT / "scripts" / "make_sample_pdf.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
text_to_pdf = _mod.text_to_pdf


def container(prefix: str) -> str:
    return f"{prefix}{check_digit(prefix)}"


C1, C2, C3, C4, C5 = (container(p) for p in ("QZXU261004", "QZXU261005", "QZXU261006",
                                              "QZXU261007", "QZXU261008"))  # fmt: skip
BAD = "QZXU7001125"  # correct check digit would be 0

DOCS = {
    "bl_resin_import.pdf": f"""BILL OF LADING (NON-NEGOTIABLE COPY)
FICTIONAL SAMPLE DOCUMENT FOR DEMONSTRATION

B/L No: SPLU2610045
Date of Issue: 2026-10-02
Shipper: Lianhe Polymer Trading Co., Ltd., Nansha District, Guangzhou, China
Consignee: Selangor Plastics Manufacturing Sdn Bhd, Section 26, Shah Alam, Selangor, Malaysia
Notify Party: Muara Logistics Sdn Bhd, Pulau Indah, Port Klang
Vessel / Voyage: MV Straits Pioneer / 2641E
Port of Loading: Nansha, Guangzhou (CNNSA)
Port of Discharge: Port Klang - Westport (MYPKG)

Container No: {C1}, {C2}
Container Size: 2 x 40' HC
Seal No: SL882101, SL882102
Packages: 1,680 BAGS (25 KG) ON 42 PALLETS
Description of Goods: POLYPROPYLENE RESIN HOMOPOLYMER, INJECTION GRADE
HS Code: 3902.10
Gross Weight: 41,800.000 KGS
Measurement: 108.000 CBM
Freight: PREPAID
""",
    "bl_chemicals_errors.pdf": f"""BILL OF LADING
FICTIONAL SAMPLE DOCUMENT FOR DEMONSTRATION

B/L No: SPLU2610311
Date of Issue: 2026-10-05
Shipper: Formosa Fine Acids Co., Ltd., Kaohsiung, Taiwan
Consignee: Johor Specialty Chemicals Sdn Bhd, Pasir Gudang Industrial Estate, Johor, Malaysia
Notify Party: Same as consignee
Vessel / Voyage: MV Lotus Bay / 118S
Port of Loading: Kaohsiung (TWKHH)
Port of Discharge: Port Klang - Northport (MYPKG)

Container No: {BAD}
Container Size: 1 x 20' GP
Packages: 80 IBC TOTES
Description of Goods: ACETIC ACID, GLACIAL. UN 2789, CLASS 8 (3), PG II. DANGEROUS GOODS
HS Code: 2915.21
Gross Weight: 24,500 KGS
Freight: COLLECT
""",
    "invoice_cpo_export.pdf": f"""COMMERCIAL INVOICE
FICTIONAL SAMPLE DOCUMENT FOR DEMONSTRATION

Invoice No: PSR-INV-2026-0912
Invoice Date: 2026-10-08
Seller: Perkasa Sawit Refinery Sdn Bhd, Jalan Kapar, Klang, Selangor, Malaysia
Buyer: Rotterdam Edible Oils B.V., Botlek, Rotterdam, Netherlands
Incoterms: FOB Port Klang
Country of Origin: Malaysia
Port of Loading: Port Klang - Westport (MYPKG)
Port of Discharge: Rotterdam (NLRTM)

Container No: {C3}, {C4}, {C5}
Container Size: 3 x 20' (FLEXITANK)
Description of Goods: RBD PALM OLEIN, IV 56 MIN, MSPO CERTIFIED
HS Code: 1511.90
Quantity: 63.000 MT
Unit Price: USD 1,085.00 / MT
Total Amount: USD 68,355.00
Gross Weight: 63,450 KGS
Net Weight: 63,000 KGS
""",
}


def main() -> None:
    out = ROOT / "data" / "samples"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in DOCS.items():
        (out / name).write_bytes(text_to_pdf(text))
        print(f"Wrote {out / name}")


if __name__ == "__main__":
    main()
