"""Parse lawphil.net-style statute HTML into normalized page records."""

from bs4 import BeautifulSoup


def parse_html(raw_bytes: bytes, source_url: str) -> list[dict]:
    """Returns a list of {raw_text, page_number, source_url}.

    lawphil.net pages have no pagination, so this yields a single "page" (0)
    containing the full body text with boilerplate (nav/scripts) stripped.
    """
    soup = BeautifulSoup(raw_bytes, "html.parser")

    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    text = soup.get_text("\n")
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    body = "\n".join(lines)

    return [{"raw_text": body, "page_number": 0, "source_url": source_url}]
