from pydantic import BaseModel, HttpUrl
from typing import List, Tuple, Optional
from urllib.parse import urlparse


class PageContent(BaseModel):
    url: str
    content: str
    title: str
    keywords: List[str]
    summary: str
    chunks: Optional[List[str]] = None

    def keywords_as_csv(self) -> str:
        """
        Joins the keywords into a CSV string.
        """
        return ",".join(self.keywords)

    def extract_url_hierarchy_and_subdomain(self) -> Tuple[List[str], Optional[str]]:
        """
        Extracts the URL hierarchy and identifies the subdomain
        """
        parsed_url = urlparse(self.url)
        path_segments = [segment for segment in parsed_url.path.split("/") if segment]
        subdomain = parsed_url.hostname.split(".")[0] if parsed_url.hostname else None
        return path_segments, subdomain
