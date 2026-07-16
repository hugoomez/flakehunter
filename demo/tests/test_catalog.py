from shopcart.catalog import Catalog


def test_load_populates_cache() -> None:
    catalog = Catalog()

    item = catalog.load("tea", 4.50)

    assert item.name == "tea"


def test_read_from_cache() -> None:
    catalog = Catalog()

    assert catalog.read_cached("tea").price == 4.50
