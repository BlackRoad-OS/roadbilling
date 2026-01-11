"""
RoadBilling - Billing & Subscription System for BlackRoad
Usage-based billing, subscriptions, invoicing, and Stripe integration.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional
import hashlib
import json
import logging
import threading
import uuid

logger = logging.getLogger(__name__)


class BillingCycle(str, Enum):
    """Billing cycle periods."""
    MONTHLY = "monthly"
    YEARLY = "yearly"
    WEEKLY = "weekly"
    DAILY = "daily"


class SubscriptionStatus(str, Enum):
    """Subscription status."""
    ACTIVE = "active"
    TRIALING = "trialing"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    PAUSED = "paused"


class InvoiceStatus(str, Enum):
    """Invoice status."""
    DRAFT = "draft"
    OPEN = "open"
    PAID = "paid"
    VOID = "void"
    UNCOLLECTIBLE = "uncollectible"


class PricingType(str, Enum):
    """Pricing models."""
    FLAT = "flat"
    PER_UNIT = "per_unit"
    TIERED = "tiered"
    VOLUME = "volume"


@dataclass
class PricingTier:
    """A pricing tier."""
    up_to: Optional[int]  # None = unlimited
    flat_amount: Decimal = Decimal("0")
    unit_amount: Decimal = Decimal("0")


@dataclass
class Price:
    """A price configuration."""
    id: str
    product_id: str
    amount: Decimal
    currency: str = "usd"
    pricing_type: PricingType = PricingType.FLAT
    billing_cycle: BillingCycle = BillingCycle.MONTHLY
    tiers: List[PricingTier] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Product:
    """A billable product."""
    id: str
    name: str
    description: str = ""
    prices: List[Price] = field(default_factory=list)
    features: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    active: bool = True


@dataclass
class UsageRecord:
    """A usage record."""
    id: str
    subscription_id: str
    quantity: int
    timestamp: datetime
    idempotency_key: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Subscription:
    """A customer subscription."""
    id: str
    customer_id: str
    price_id: str
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    current_period_start: datetime = field(default_factory=datetime.now)
    current_period_end: datetime = field(default_factory=datetime.now)
    trial_end: Optional[datetime] = None
    cancel_at_period_end: bool = False
    canceled_at: Optional[datetime] = None
    quantity: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)
    usage_records: List[UsageRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "price_id": self.price_id,
            "status": self.status.value,
            "current_period_start": self.current_period_start.isoformat(),
            "current_period_end": self.current_period_end.isoformat(),
            "quantity": self.quantity
        }


@dataclass
class InvoiceItem:
    """An invoice line item."""
    description: str
    quantity: int
    unit_amount: Decimal
    amount: Decimal
    product_id: Optional[str] = None


@dataclass
class Invoice:
    """An invoice."""
    id: str
    customer_id: str
    subscription_id: Optional[str] = None
    status: InvoiceStatus = InvoiceStatus.DRAFT
    items: List[InvoiceItem] = field(default_factory=list)
    subtotal: Decimal = Decimal("0")
    tax: Decimal = Decimal("0")
    total: Decimal = Decimal("0")
    currency: str = "usd"
    created_at: datetime = field(default_factory=datetime.now)
    due_date: Optional[datetime] = None
    paid_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "status": self.status.value,
            "subtotal": str(self.subtotal),
            "tax": str(self.tax),
            "total": str(self.total),
            "currency": self.currency,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class Customer:
    """A billing customer."""
    id: str
    email: str
    name: Optional[str] = None
    payment_method_id: Optional[str] = None
    default_currency: str = "usd"
    balance: Decimal = Decimal("0")
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)


class BillingStore:
    """Store for billing data."""

    def __init__(self):
        self.products: Dict[str, Product] = {}
        self.prices: Dict[str, Price] = {}
        self.customers: Dict[str, Customer] = {}
        self.subscriptions: Dict[str, Subscription] = {}
        self.invoices: Dict[str, Invoice] = {}
        self._lock = threading.Lock()

    def save_product(self, product: Product) -> None:
        with self._lock:
            self.products[product.id] = product
            for price in product.prices:
                self.prices[price.id] = price

    def get_product(self, product_id: str) -> Optional[Product]:
        return self.products.get(product_id)

    def save_customer(self, customer: Customer) -> None:
        with self._lock:
            self.customers[customer.id] = customer

    def get_customer(self, customer_id: str) -> Optional[Customer]:
        return self.customers.get(customer_id)

    def save_subscription(self, sub: Subscription) -> None:
        with self._lock:
            self.subscriptions[sub.id] = sub

    def get_subscription(self, sub_id: str) -> Optional[Subscription]:
        return self.subscriptions.get(sub_id)

    def get_customer_subscriptions(self, customer_id: str) -> List[Subscription]:
        return [s for s in self.subscriptions.values() if s.customer_id == customer_id]

    def save_invoice(self, invoice: Invoice) -> None:
        with self._lock:
            self.invoices[invoice.id] = invoice

    def get_invoice(self, invoice_id: str) -> Optional[Invoice]:
        return self.invoices.get(invoice_id)

    def get_customer_invoices(self, customer_id: str) -> List[Invoice]:
        return sorted(
            [i for i in self.invoices.values() if i.customer_id == customer_id],
            key=lambda i: i.created_at,
            reverse=True
        )


class PricingCalculator:
    """Calculate prices based on usage."""

    def calculate(self, price: Price, quantity: int) -> Decimal:
        """Calculate total amount."""
        if price.pricing_type == PricingType.FLAT:
            return price.amount * quantity

        elif price.pricing_type == PricingType.PER_UNIT:
            return price.amount * quantity

        elif price.pricing_type == PricingType.TIERED:
            return self._calculate_tiered(price, quantity)

        elif price.pricing_type == PricingType.VOLUME:
            return self._calculate_volume(price, quantity)

        return price.amount

    def _calculate_tiered(self, price: Price, quantity: int) -> Decimal:
        """Calculate tiered pricing."""
        total = Decimal("0")
        remaining = quantity
        previous_up_to = 0

        for tier in price.tiers:
            if remaining <= 0:
                break

            tier_quantity = min(
                remaining,
                (tier.up_to - previous_up_to) if tier.up_to else remaining
            )

            total += tier.flat_amount
            total += tier.unit_amount * tier_quantity

            remaining -= tier_quantity
            previous_up_to = tier.up_to or 0

        return total

    def _calculate_volume(self, price: Price, quantity: int) -> Decimal:
        """Calculate volume pricing."""
        for tier in price.tiers:
            if tier.up_to is None or quantity <= tier.up_to:
                return tier.flat_amount + (tier.unit_amount * quantity)
        return Decimal("0")


class InvoiceGenerator:
    """Generate invoices for subscriptions."""

    def __init__(self, store: BillingStore, calculator: PricingCalculator):
        self.store = store
        self.calculator = calculator

    def generate(self, subscription: Subscription) -> Invoice:
        """Generate invoice for subscription."""
        price = self.store.prices.get(subscription.price_id)
        if not price:
            raise ValueError(f"Price not found: {subscription.price_id}")

        product = self.store.products.get(price.product_id)

        # Calculate usage-based charges
        usage_quantity = sum(r.quantity for r in subscription.usage_records)
        total_quantity = subscription.quantity + usage_quantity

        amount = self.calculator.calculate(price, total_quantity)

        items = [
            InvoiceItem(
                description=product.name if product else "Subscription",
                quantity=total_quantity,
                unit_amount=price.amount,
                amount=amount,
                product_id=price.product_id
            )
        ]

        invoice = Invoice(
            id=str(uuid.uuid4()),
            customer_id=subscription.customer_id,
            subscription_id=subscription.id,
            items=items,
            subtotal=amount,
            total=amount,
            currency=price.currency,
            due_date=datetime.now() + timedelta(days=30)
        )

        self.store.save_invoice(invoice)
        return invoice


class BillingManager:
    """High-level billing management."""

    def __init__(self):
        self.store = BillingStore()
        self.calculator = PricingCalculator()
        self.generator = InvoiceGenerator(self.store, self.calculator)
        self._hooks: Dict[str, List[Callable]] = {
            "subscription.created": [],
            "subscription.canceled": [],
            "invoice.created": [],
            "invoice.paid": []
        }

    def add_hook(self, event: str, handler: Callable) -> None:
        """Add event hook."""
        if event in self._hooks:
            self._hooks[event].append(handler)

    def _trigger_hooks(self, event: str, data: Any) -> None:
        for handler in self._hooks.get(event, []):
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Hook error: {e}")

    def create_product(
        self,
        name: str,
        description: str = "",
        features: List[str] = None
    ) -> Product:
        """Create a product."""
        product = Product(
            id=str(uuid.uuid4()),
            name=name,
            description=description,
            features=features or []
        )
        self.store.save_product(product)
        return product

    def add_price(
        self,
        product_id: str,
        amount: float,
        currency: str = "usd",
        billing_cycle: BillingCycle = BillingCycle.MONTHLY,
        pricing_type: PricingType = PricingType.FLAT
    ) -> Optional[Price]:
        """Add price to product."""
        product = self.store.get_product(product_id)
        if not product:
            return None

        price = Price(
            id=str(uuid.uuid4()),
            product_id=product_id,
            amount=Decimal(str(amount)),
            currency=currency,
            billing_cycle=billing_cycle,
            pricing_type=pricing_type
        )

        product.prices.append(price)
        self.store.prices[price.id] = price
        return price

    def create_customer(
        self,
        email: str,
        name: Optional[str] = None
    ) -> Customer:
        """Create a customer."""
        customer = Customer(
            id=str(uuid.uuid4()),
            email=email,
            name=name
        )
        self.store.save_customer(customer)
        return customer

    def create_subscription(
        self,
        customer_id: str,
        price_id: str,
        quantity: int = 1,
        trial_days: int = 0
    ) -> Optional[Subscription]:
        """Create a subscription."""
        customer = self.store.get_customer(customer_id)
        price = self.store.prices.get(price_id)

        if not customer or not price:
            return None

        now = datetime.now()
        period_end = self._calculate_period_end(now, price.billing_cycle)

        subscription = Subscription(
            id=str(uuid.uuid4()),
            customer_id=customer_id,
            price_id=price_id,
            quantity=quantity,
            status=SubscriptionStatus.TRIALING if trial_days > 0 else SubscriptionStatus.ACTIVE,
            current_period_start=now,
            current_period_end=period_end,
            trial_end=now + timedelta(days=trial_days) if trial_days > 0 else None
        )

        self.store.save_subscription(subscription)
        self._trigger_hooks("subscription.created", subscription)

        return subscription

    def _calculate_period_end(self, start: datetime, cycle: BillingCycle) -> datetime:
        if cycle == BillingCycle.MONTHLY:
            return start + timedelta(days=30)
        elif cycle == BillingCycle.YEARLY:
            return start + timedelta(days=365)
        elif cycle == BillingCycle.WEEKLY:
            return start + timedelta(days=7)
        elif cycle == BillingCycle.DAILY:
            return start + timedelta(days=1)
        return start + timedelta(days=30)

    def cancel_subscription(self, subscription_id: str, immediate: bool = False) -> bool:
        """Cancel a subscription."""
        sub = self.store.get_subscription(subscription_id)
        if not sub:
            return False

        if immediate:
            sub.status = SubscriptionStatus.CANCELED
            sub.canceled_at = datetime.now()
        else:
            sub.cancel_at_period_end = True

        self.store.save_subscription(sub)
        self._trigger_hooks("subscription.canceled", sub)
        return True

    def record_usage(
        self,
        subscription_id: str,
        quantity: int,
        idempotency_key: Optional[str] = None
    ) -> Optional[UsageRecord]:
        """Record usage for metered billing."""
        sub = self.store.get_subscription(subscription_id)
        if not sub:
            return None

        # Check idempotency
        if idempotency_key:
            for record in sub.usage_records:
                if record.idempotency_key == idempotency_key:
                    return record

        record = UsageRecord(
            id=str(uuid.uuid4()),
            subscription_id=subscription_id,
            quantity=quantity,
            timestamp=datetime.now(),
            idempotency_key=idempotency_key
        )

        sub.usage_records.append(record)
        return record

    def generate_invoice(self, subscription_id: str) -> Optional[Invoice]:
        """Generate invoice for subscription."""
        sub = self.store.get_subscription(subscription_id)
        if not sub:
            return None

        invoice = self.generator.generate(sub)
        self._trigger_hooks("invoice.created", invoice)
        
        # Clear usage records after invoicing
        sub.usage_records = []
        
        return invoice

    def pay_invoice(self, invoice_id: str) -> bool:
        """Mark invoice as paid."""
        invoice = self.store.get_invoice(invoice_id)
        if not invoice or invoice.status != InvoiceStatus.OPEN:
            return False

        invoice.status = InvoiceStatus.PAID
        invoice.paid_at = datetime.now()
        self._trigger_hooks("invoice.paid", invoice)
        return True

    def get_customer_balance(self, customer_id: str) -> Dict[str, Any]:
        """Get customer billing balance."""
        invoices = self.store.get_customer_invoices(customer_id)
        subscriptions = self.store.get_customer_subscriptions(customer_id)

        total_paid = sum(i.total for i in invoices if i.status == InvoiceStatus.PAID)
        outstanding = sum(i.total for i in invoices if i.status == InvoiceStatus.OPEN)

        return {
            "customer_id": customer_id,
            "total_paid": str(total_paid),
            "outstanding": str(outstanding),
            "active_subscriptions": len([s for s in subscriptions if s.status == SubscriptionStatus.ACTIVE])
        }


# Example usage
def example_usage():
    """Example billing usage."""
    manager = BillingManager()

    # Create product
    product = manager.create_product(
        name="Pro Plan",
        description="Professional features",
        features=["Unlimited projects", "Priority support", "API access"]
    )

    # Add price
    price = manager.add_price(
        product.id,
        amount=29.99,
        billing_cycle=BillingCycle.MONTHLY
    )

    # Create customer
    customer = manager.create_customer(
        email="customer@example.com",
        name="John Doe"
    )

    # Create subscription
    subscription = manager.create_subscription(
        customer.id,
        price.id,
        trial_days=14
    )

    print(f"Subscription created: {subscription.id}")
    print(f"Status: {subscription.status.value}")
    print(f"Trial ends: {subscription.trial_end}")

    # Record usage
    manager.record_usage(subscription.id, quantity=100)
    manager.record_usage(subscription.id, quantity=50)

    # Generate invoice
    invoice = manager.generate_invoice(subscription.id)
    print(f"\nInvoice: {invoice.id}")
    print(f"Total: ${invoice.total}")

    # Get balance
    balance = manager.get_customer_balance(customer.id)
    print(f"\nCustomer balance: {balance}")
