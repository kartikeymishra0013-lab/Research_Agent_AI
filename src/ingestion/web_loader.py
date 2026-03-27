"""
Web URL loader using httpx + BeautifulSoup for HTML parsing.
Cleans boilerplate and extracts main article content.
"""
from __future__ import annotations

from urllib.parse import urlparse

from src.ingestion.base_loader import BaseLoader, Document
from src.utils.logger import get_logger

logger = get_logger(__name__)


class WebLoader(BaseLoader):
    """
    Loads and parses web pages, extracting clean text content.

    Uses trafilatura for article-focused extraction with
    BeautifulSoup as fallback for general HTML parsing.
    """

    def __init__(
        self,
        timeout: int = 30,
        user_agent: str = "ScientificDocumentPipeline/1.0",
        use_trafilatura: bool = True,
    ):
        self.timeout = timeout
        self.user_agent = user_agent
        self.use_trafilatura = use_trafilatura

    def load(self, source: str) -> Document:
        if not (source.startswith("http://") or source.startswith("https://")):
            raise ValueError(f"Not a valid URL: {source}")

        logger.info(f"Loading URL: {source}")
        html = self._fetch(source)

        if self.use_trafilatura:
            try:
                content, metadata = self._extract_with_trafilatura(html, source)
            except ImportError:
                logger.warning("trafilatura not available, using BeautifulSoup")
                content, metadata = self._extract_with_bs4(html, source)
        else:
            content, metadata = self._extract_with_bs4(html, source)

        parsed = urlparse(source)
        metadata.update({
            "url": source,
            "domain": parsed.netloc,
        })

        logger.info(f"Extracted {len(content)} chars from {source}")
        return Document(
            content=content,
            source=source,
            doc_type="url",
            metadata=metadata,
        )

    def _fetch(self, url: str) -> str:
        import httpx
        headers = {"User-Agent": self.user_agent}
        response = httpx.get(url, headers=headers, timeout=self.timeout, follow_redirects=True)
        response.raise_for_status()
        return response.text

    def _extract_with_trafilatura(self, html: str, url: str) -> tuple[str, dict]:
        import trafilatura
        from trafilatura.settings import use_config

        config = use_config()
        config.set("DEFAULT", "EXTRACTION_TIMEOUT", "0")

        result = trafilatura.extract(html, url=url, include_comments=False,
                                     include_tables=True, output_format="txt",
                                     config=config)
        meta_result = trafilatura.extract_metadata(html, default_url=url)

        content = result or ""
        metadata = {}
        if meta_result:
            metadata = {
                "title": meta_result.title or "",
                "author": meta_result.author or "",
                "date": meta_result.date or "",
                "description": meta_result.description or "",
            }
        return content, metadata

    def _extract_with_bs4(self, html: str, url: str) -> tuple[str, dict]:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "advertisement", "iframe"]):
            tag.decompose()

        title = soup.find("title")
        title_text = title.get_text(strip=True) if title else ""

        # Try to get main content area
        main = (soup.find("main") or soup.find("article") or
                soup.find("div", class_=lambda c: c and "content" in c.lower()) or
                soup.body)

        content = main.get_text(separator="\n", strip=True) if main else ""

        metadata = {"title": title_text}
        return content, metadata
