"""Human-reviewed list of source documents for the PH recycling/waste-management corpus.

Every entry was checked with `curl -A "<browser UA>" -I <url>` on 2026-08-20 and
returned HTTP 200. Government hosting paths shift over time — re-verify before
re-ingesting if fetch.py starts failing.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceDoc:
    doc_id: str
    title: str
    url: str
    doc_type: str  # statute | irr | framework | advisory
    fmt: str  # html | pdf
    jurisdiction: str  # national | lgu:<name>
    language: str
    milestone: str  # which build milestone first ingests this doc


SOURCES: list[SourceDoc] = [
    SourceDoc(
        doc_id="ra9003",
        title="RA 9003 - Ecological Solid Waste Management Act of 2000",
        url="https://lawphil.net/statutes/repacts/ra2001/ra_9003_2001.html",
        doc_type="statute",
        fmt="html",
        jurisdiction="national",
        language="en",
        milestone="m1",
    ),
    SourceDoc(
        doc_id="ra9003-irr",
        title="DENR DAO 2001-34 - Implementing Rules and Regulations of RA 9003",
        url="https://nswmc.emb.gov.ph/wp-content/uploads/2025/04/RA-9003-IRR.pdf",
        doc_type="irr",
        fmt="pdf",
        jurisdiction="national",
        language="en",
        milestone="m1",
    ),
    SourceDoc(
        doc_id="nswmc-framework",
        title="National Solid Waste Management Framework",
        url="https://nswmc.emb.gov.ph/wp-content/uploads/2017/11/NSWMC-FRAMEWORK-PDF.pdf",
        doc_type="framework",
        fmt="pdf",
        jurisdiction="national",
        language="en",
        milestone="m2",
    ),
    SourceDoc(
        doc_id="eswm-households",
        title="Ecological Solid Waste Management for Households",
        url="https://nswmc.emb.gov.ph/wp-content/uploads/2016/06/6.ESWM-for-HH.pdf",
        doc_type="advisory",
        fmt="pdf",
        jurisdiction="national",
        language="en",
        milestone="m2",
    ),
    SourceDoc(
        doc_id="swm-made-easy",
        title="Solid Waste Management Made Easy (NSWMC Guidebook)",
        url="https://nswmc.emb.gov.ph/wp-content/uploads/2018/03/guide_book_bluebook.pdf",
        doc_type="advisory",
        fmt="pdf",
        jurisdiction="national",
        language="en",
        milestone="m2",
    ),
]


def sources_for_milestone(milestone: str) -> list[SourceDoc]:
    order = {"m1": 1, "m2": 2}
    return [s for s in SOURCES if order[s.milestone] <= order[milestone]]
