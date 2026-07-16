from shopcart.cart import SHARED_CART


def test_add_discount_to_shared_cart() -> None:
    SHARED_CART.add_discount(0.10)

    assert len(SHARED_CART.discounts) == 1


def test_discount_count() -> None:
    assert len(SHARED_CART.discounts) == 0
