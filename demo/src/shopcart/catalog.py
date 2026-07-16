"""Catalog support for the demo cart."""

from dataclasses import dataclass


@dataclass(frozen=True)
class CatalogItem:
    name: str
    price: float


CATALOG_CACHE: dict[str, CatalogItem] = {}


class Catalog:
    """Load catalog entries and retain them in a module-level cache."""

    def load(self, name: str, price: float) -> CatalogItem:
        item = CatalogItem(name=name, price=price)
        CATALOG_CACHE[name] = item
        return item

    def read_cached(self, name: str) -> CatalogItem:
        return CATALOG_CACHE[name]
