import json

import requests

base_url = "https://api.poker44.net/api/v1/benchmark"

status = requests.get(base_url, timeout=30).json()["data"]
source_date = status["latestSourceDate"]

source_date = "2026-07-10"
print(source_date)



payload = requests.get(
    f"{base_url}/chunks",
    params={"sourceDate": source_date, "limit": 5},
    timeout=30,
).json()["data"]


def hero_view(hand):
    """Return the hero-seat-focused view of a single hand."""
    metadata = hand.get("metadata") or {}
    players = hand.get("players") or []
    streets = hand.get("streets") or []
    actions = hand.get("actions") or []
    outcome = hand.get("outcome") or {}

    hero_seat = metadata.get("hero_seat")

    hero_player = next((p for p in players if p.get("seat") == hero_seat), None)
    hero_actions = [a for a in actions if a.get("actor_seat") == hero_seat]

    return {
        "hero_seat": hero_seat,
        "metadata": metadata,
        "hero_player": hero_player,
        "players": players,
        "streets": streets,
        "hero_actions": hero_actions,
        "actions": actions,
        "outcome": outcome,
    }


# Show the hero-seat data for the first few batches/hands.
MAX_BATCHES = 2
MAX_HANDS = 3

for chunk in payload["chunks"]:
    batches = chunk["chunks"]          # list of batches (each a list of hands)
    labels = chunk["groundTruth"]      # one 0/1 label per batch

    print(f"\n=== chunk {chunk.get('chunkId')} (batches={len(batches)}) ===")

    for batch_idx, (batch, label) in enumerate(zip(batches, labels)):
        if batch_idx >= MAX_BATCHES:
            break
        tag = "bot" if label == 1 else "human"
        print(f"\n--- batch {batch_idx} label={label} ({tag}) hands={len(batch)} ---")

        for hand_idx, hand in enumerate(batch):
            if hand_idx >= MAX_HANDS:
                break
            view = hero_view(hand)
            print(f"\n[hand {hand_idx}] hero_seat={view['hero_seat']}")
            print(json.dumps(view, indent=2, default=str))
        c = input("Press Enter to continue...")