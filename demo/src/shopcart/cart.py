"""Cart support for the demo cart."""

from dataclasses import dataclass, field


@dataclass
class Cart:
    """A minimal cart that tracks item prices and percentage discounts."""

    items: list[float] = field(default_factory=list)
    discounts: list[float] = field(default_factory=list)

    def add_item(self, price: float) -> None:
        self.items.append(price)

    def add_discount(self, percentage: float) -> None:
        self.discounts.append(percentage)

    def total(self) -> float:
        subtotal = sum(self.items)
        discount = sum(self.discounts)
        return round(subtotal * (1 - discount), 2)


SHARED_CART = Cart()
