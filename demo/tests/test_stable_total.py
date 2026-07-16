from shopcart.cart import Cart


def test_total_is_correct() -> None:
    cart = Cart()
    cart.add_item(10.00)
    cart.add_item(5.00)
    cart.add_discount(0.10)

    assert cart.total() == 13.50
