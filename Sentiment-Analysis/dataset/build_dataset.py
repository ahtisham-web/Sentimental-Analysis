"""
Builds a labeled sentiment dataset (text, label) and saves it as CSV.
Label values: positive, neutral, negative
"""
import pandas as pd
import random
from pathlib import Path

random.seed(42)

positive_templates = [
    "I absolutely love this {thing}, it's amazing!",
    "This {thing} exceeded all my expectations.",
    "Best {thing} I have ever used, highly recommend it.",
    "I'm so happy with this {thing}, it works perfectly.",
    "What a fantastic {thing}, truly impressive quality.",
    "The {thing} made my day so much better.",
    "I can't stop smiling because of this {thing}.",
    "This {thing} is a game changer, thank you so much!",
    "Great job on the {thing}, everything works flawlessly.",
    "I'm thrilled with how well the {thing} performs.",
    "The customer service for this {thing} was outstanding.",
    "Such a wonderful experience with this {thing}.",
    "I appreciate how easy the {thing} is to use.",
    "This {thing} is exactly what I needed, perfect!",
    "Absolutely delighted with the {thing}, five stars.",
    "The {thing} works great and looks even better.",
    "I had a great time using this {thing} today.",
    "This is the best {thing} on the market right now.",
    "Really impressed by the quality of this {thing}.",
    "Everything about this {thing} feels premium and well made.",
]

neutral_templates = [
    "The {thing} arrived on Tuesday as scheduled.",
    "I used the {thing} for about an hour today.",
    "The {thing} comes in three different colors.",
    "This {thing} is similar to the previous version.",
    "The meeting about the {thing} is at 3 PM.",
    "The {thing} weighs approximately two kilograms.",
    "I have not decided yet whether I like the {thing}.",
    "The {thing} was delivered in a plain box.",
    "There is a new update available for the {thing}.",
    "The {thing} requires two batteries to operate.",
    "I read the instructions for the {thing} yesterday.",
    "The store has the {thing} in stock again.",
    "The {thing} is priced at fifty dollars.",
    "It took some time to set up the {thing}.",
    "The {thing} is compatible with most devices.",
    "I am still testing out the {thing} this week.",
    "The report on the {thing} will be released tomorrow.",
    "The {thing} has a standard one year warranty.",
    "They discussed the {thing} during the conference call.",
    "The {thing} is currently being reviewed by the team.",
]

negative_templates = [
    "I really hate how this {thing} turned out.",
    "This {thing} is a complete waste of money.",
    "I'm so disappointed with the {thing}, it barely works.",
    "The {thing} broke after just one day of use.",
    "Terrible experience with this {thing}, would not recommend.",
    "This {thing} is the worst I have ever purchased.",
    "I'm frustrated because the {thing} keeps failing.",
    "The {thing} was a huge letdown, poor quality overall.",
    "Nothing about this {thing} works the way it should.",
    "I regret buying this {thing}, total disappointment.",
    "The customer support for this {thing} was awful.",
    "This {thing} arrived damaged and unusable.",
    "I'm annoyed that the {thing} stopped working so quickly.",
    "Such a bad experience, the {thing} is unreliable.",
    "The {thing} is overpriced and underdelivers badly.",
    "I can't believe how poorly the {thing} performs.",
    "This {thing} made everything so much more difficult.",
    "Avoid this {thing}, it constantly malfunctions.",
    "The {thing} left me feeling angry and cheated.",
    "Absolutely awful quality, this {thing} fell apart fast.",
]

things = [
    "phone", "laptop", "app", "service", "product", "movie", "restaurant", "book",
    "software", "car", "hotel", "course", "game", "website", "device", "headphones",
    "camera", "watch", "delivery", "package", "flight", "meal", "software update",
    "tool", "gadget", "subscription", "experience", "team", "presentation", "vacation"
]

# Extra hand-written standalone examples for variety (not template-based)
extra_positive = [
    "You did an amazing job, I'm proud of you!",
    "This is wonderful news, congratulations to everyone involved.",
    "I feel great today, everything is going smoothly.",
    "Thanks a lot, this really made my week better.",
    "The weather is beautiful and I'm enjoying my walk.",
    "Our team won the championship, what an incredible moment!",
    "I love spending time with my family on weekends.",
    "The concert last night was absolutely unforgettable.",
    "She did a fantastic job presenting the project.",
    "I'm grateful for all the support I've received.",
]

extra_neutral = [
    "The train departs from platform four at noon.",
    "The document contains twelve pages in total.",
    "He walked to the store and bought some bread.",
    "The temperature today is around twenty degrees.",
    "The office will be closed on public holidays.",
    "She is currently reading a book about history.",
    "The file was saved to the shared folder.",
    "The bus route changes starting next month.",
    "The conference will be held in the main hall.",
    "The package weighs about three pounds.",
]

extra_negative = [
    "I am furious about how this situation was handled.",
    "This has been the worst week of my life.",
    "I can't believe they treated us so poorly.",
    "The traffic today was absolutely infuriating.",
    "I'm exhausted and nothing seems to be going right.",
    "This decision makes me really upset and disappointed.",
    "The service here was rude and completely unhelpful.",
    "I feel ignored and frustrated by the whole process.",
    "Everything about today has been stressful and unpleasant.",
    "This is unacceptable, I want a refund immediately.",
]

rows = []
for template in positive_templates:
    for thing in things:
        rows.append((template.format(thing=thing), "positive"))
for template in neutral_templates:
    for thing in things:
        rows.append((template.format(thing=thing), "neutral"))
for template in negative_templates:
    for thing in things:
        rows.append((template.format(thing=thing), "negative"))

for s in extra_positive:
    rows.append((s, "positive"))
for s in extra_neutral:
    rows.append((s, "neutral"))
for s in extra_negative:
    rows.append((s, "negative"))

df = pd.DataFrame(rows, columns=["text", "label"])
df = df.drop_duplicates(subset=["text"]).sample(frac=1, random_state=42).reset_index(drop=True)

OUTPUT_PATH = Path(__file__).resolve().parent / "sentiment.csv"
df.to_csv(OUTPUT_PATH, index=False)
print("Dataset shape:", df.shape)
print(df["label"].value_counts())
