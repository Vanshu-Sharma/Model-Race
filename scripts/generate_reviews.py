"""
Generate a 500-review sentiment dataset and save to data/reviews.csv.

Columns: review_id, text, label  (label: "positive" | "negative")
250 positive + 250 negative = 500 total, balanced.

Run once: python scripts/generate_reviews.py
"""

import csv
import pathlib
import random

POSITIVE_TEMPLATES = [
    "I absolutely loved {product}. {positive_detail}. Would highly recommend!",
    "{product} exceeded all my expectations. {positive_detail}.",
    "Five stars for {product}! {positive_detail}. Will buy again.",
    "What a great purchase! {product} is {positive_adj} and {positive_adj2}.",
    "{product} is exactly what I needed. {positive_detail}. Very satisfied.",
    "So happy with {product}. {positive_detail}. Perfect quality.",
    "{positive_detail}. This {product} is simply amazing.",
    "The {product} works like a charm. {positive_detail}.",
    "Best {product} I've ever bought. {positive_detail}. 10/10.",
    "Impressive quality on this {product}. {positive_detail}.",
    "{product} arrived on time and is {positive_adj}. {positive_detail}.",
    "Really pleased with {product}. {positive_detail}. Great value for money.",
    "Couldn't be happier with my {product}. {positive_detail}.",
    "Outstanding {product}! {positive_detail}. Exceeded expectations.",
    "{product} is top-notch. {positive_detail}. Highly recommended.",
    "Absolutely brilliant {product}. {positive_detail}. Very well made.",
    "Superb {product}! {positive_detail}. Fast shipping too.",
    "Great experience with {product}. {positive_detail}. Will order again.",
    "This {product} is a game-changer. {positive_detail}.",
    "{positive_detail}. I'm extremely impressed by {product}.",
]

NEGATIVE_TEMPLATES = [
    "Very disappointed with {product}. {negative_detail}. Not worth the money.",
    "{product} broke after just {time}. {negative_detail}. Avoid!",
    "Terrible quality {product}. {negative_detail}. Total waste of money.",
    "I regret buying this {product}. {negative_detail}.",
    "{product} did not work as advertised. {negative_detail}. 1 star.",
    "Extremely disappointed. {product} is {negative_adj} and {negative_adj2}.",
    "{negative_detail}. This {product} is awful and I want a refund.",
    "The {product} stopped working after {time}. {negative_detail}.",
    "Worst {product} I've ever bought. {negative_detail}. Zero stars.",
    "Poor quality {product}. {negative_detail}. Do not recommend.",
    "{product} arrived damaged. {negative_detail}. Very unhappy.",
    "Really unhappy with {product}. {negative_detail}. Waste of money.",
    "Cannot recommend {product}. {negative_detail}. Returned it immediately.",
    "Shocking quality {product}. {negative_detail}. Total disappointment.",
    "{product} is substandard. {negative_detail}. Expected much better.",
    "Absolutely terrible {product}. {negative_detail}. Very poorly made.",
    "Rubbish {product}! {negative_detail}. Customer service was no help either.",
    "Awful experience with {product}. {negative_detail}. Never buying again.",
    "This {product} is a rip-off. {negative_detail}.",
    "{negative_detail}. I am extremely frustrated with {product}.",
]

PRODUCTS = [
    "this laptop", "the headphones", "this coffee maker", "the phone case",
    "this blender", "the wireless mouse", "this jacket", "the sneakers",
    "this book", "the vacuum cleaner", "this TV", "the tablet",
    "this smartwatch", "the camera", "this kitchen knife set", "the backpack",
    "this USB hub", "the desk lamp", "this gaming chair", "the charger",
    "this skincare kit", "the yoga mat", "this water bottle", "the earbuds",
    "this mechanical keyboard", "the monitor", "this electric toothbrush",
    "the protein powder", "this board game", "the air fryer",
]

POSITIVE_DETAILS = [
    "The build quality is exceptional",
    "Setup was straightforward and intuitive",
    "Battery life is incredible – lasts all day",
    "It performs exactly as described",
    "The packaging was secure and eco-friendly",
    "Customer support was responsive and helpful",
    "It arrived two days early",
    "The design is elegant and modern",
    "Very comfortable to use for long periods",
    "Great value for the price point",
    "My whole family loves it",
    "Works perfectly out of the box",
    "Very durable – feels like it will last years",
    "The colour is exactly as shown in photos",
    "Super easy to clean and maintain",
    "Sound quality is crystal clear",
    "Fits perfectly and feels premium",
    "Instructions were clear and easy to follow",
    "The features are well thought out",
    "Significantly better than my previous one",
]

NEGATIVE_DETAILS = [
    "The build quality is cheap and flimsy",
    "It stopped working after just one week",
    "The description was completely misleading",
    "Customer support ignored all my emails",
    "It took three weeks to arrive",
    "Missing parts straight out of the box",
    "It made a horrible noise from the start",
    "The colour was nothing like the photos",
    "It scratched and dented within days",
    "Battery drains in under two hours",
    "Absolutely no instructions included",
    "Returned it but refund never arrived",
    "It overheats dangerously after minutes",
    "The stitching came apart immediately",
    "Far too small – completely wrong size",
    "It won't connect to any of my devices",
    "Smells of chemicals even after washing",
    "Buttons jam and are unresponsive",
    "Gave out electrical sparks – dangerous!",
    "Every setting is confusing and unusable",
]

POSITIVE_ADJ = [
    "sturdy", "elegant", "lightweight", "fast", "reliable", "gorgeous",
    "comfortable", "responsive", "well-made", "impressive",
]
NEGATIVE_ADJ = [
    "flimsy", "cheap", "slow", "unreliable", "ugly", "uncomfortable",
    "unresponsive", "poorly-made", "disappointing", "defective",
]
TIMES = ["one week", "two weeks", "a month", "two uses", "three days", "one day"]


def make_review(template: str, label: str, rng: random.Random) -> str:
    product = rng.choice(PRODUCTS)
    return template.format(
        product=product,
        positive_detail=rng.choice(POSITIVE_DETAILS),
        negative_detail=rng.choice(NEGATIVE_DETAILS),
        positive_adj=rng.choice(POSITIVE_ADJ),
        positive_adj2=rng.choice(POSITIVE_ADJ),
        negative_adj=rng.choice(NEGATIVE_ADJ),
        negative_adj2=rng.choice(NEGATIVE_ADJ),
        time=rng.choice(TIMES),
    )


def main() -> None:
    rng = random.Random(42)
    reviews: list[dict[str, str]] = []

    for i in range(250):
        template = rng.choice(POSITIVE_TEMPLATES)
        reviews.append(
            {
                "review_id": str(i + 1),
                "text": make_review(template, "positive", rng),
                "label": "positive",
            }
        )

    for i in range(250):
        template = rng.choice(NEGATIVE_TEMPLATES)
        reviews.append(
            {
                "review_id": str(250 + i + 1),
                "text": make_review(template, "negative", rng),
                "label": "negative",
            }
        )

    rng.shuffle(reviews)

    out = pathlib.Path(__file__).parent.parent / "data" / "reviews.csv"
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["review_id", "text", "label"])
        writer.writeheader()
        writer.writerows(reviews)

    print(f"Saved {len(reviews)} reviews to {out}")
    pos = sum(1 for r in reviews if r["label"] == "positive")
    neg = sum(1 for r in reviews if r["label"] == "negative")
    print(f"  Positive: {pos}, Negative: {neg}")


if __name__ == "__main__":
    main()
