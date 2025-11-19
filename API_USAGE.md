# League of Legends API Integration Guide

## Quick Start

### Setup
1. Make sure your API key is in `keys.env.local`:
```
LEAGUE_API_KEY=RGAPI-your-key-here
```

2. Install required packages:
```bash
pip install requests python-dotenv pandas
```

## Usage Examples

### Method 1: Get Stats for a Specific Player

```python
from league_api import get_player_stats_from_match

match_id = "NA1_5404818015"
username = "coolkaw"

player_stats = get_player_stats_from_match(match_id, username)

if player_stats is not None:
    print(f"Champion: {player_stats['championName'].values[0]}")
    print(f"KDA: {player_stats['KDA'].values[0]:.2f}")
    
    # Save for model prediction
    player_stats.to_csv('my_stats.csv', index=False)
```

### Method 2: Get All Players from a Match

```python
from league_api import get_match_data, extract_player_stats

match_id = "NA1_5404818015"

# Fetch match
match_data = get_match_data(match_id)

# Extract all players
all_players = extract_player_stats(match_data)

# Filter for specific player (alternative method)
my_stats = all_players[all_players['summonerName'] == 'coolkaw']
```

### Method 3: Predict Rank from Stats

```python
from league_api import get_player_stats_from_match
import pickle

# Load model
model = pickle.load(open("models/vanilla_tree.sav", "rb"))
ranks = ["Unranked", "Iron", "Bronze", "Silver", "Gold", "Platinum", 
         "Emerald", "Diamond", "Master", "Grandmaster", "Challenger"]

# Get player stats
player_stats = get_player_stats_from_match("NA1_5404818015", "coolkaw")

# Prepare features (remove non-numeric columns)
features = player_stats.drop(columns=['summonerName', 'championName'])

# Predict
prediction = model.predict(features)[0]
print(f"Predicted Rank: {ranks[prediction]}")
```

## Available Functions

### `get_match_data(match_id)`
Fetches raw match data from Riot API.
- **Args:** `match_id` (str) - Match ID like "NA1_5404818015"
- **Returns:** Dictionary with full match data or None on error

### `extract_player_stats(match_data, username=None)`
Extracts formatted player statistics from match data.
- **Args:** 
  - `match_data` (dict) - Match data from `get_match_data()`
  - `username` (str, optional) - Filter for specific player
- **Returns:** pandas DataFrame with player stats

### `get_player_stats_from_match(match_id, username)`
Convenience function combining both steps above.
- **Args:**
  - `match_id` (str) - Match ID
  - `username` (str) - Player's summoner name
- **Returns:** pandas DataFrame with single player's stats

## Data Format

The extracted data includes **45 features** matching your model's training data:

### Basic Stats
- MinionsKilled, DmgDealt, DmgTaken, TurretDmgDealt, TotalGold, Win
- kills, deaths, assists, visionScore

### Items
- item1, item2, item3, item4, item5, item6

### Runes/Perks
- PrimaryKeyStone, PrimarySlot1-3, SecondarySlot1-2

### Summoner Spells
- SummonerSpell1, SummonerSpell2

### Objectives
- DragonKills, BaronKills

### Derived Features (Auto-calculated)
- KDA, GoldPerMin, CSPerMin, DmgPerMin, VisionPerMin
- DmgPerGold, DmgEfficiency, ItemCount, ObjectiveParticipation
- GamePhase (Early/Mid/Late)

### Position Info
- Lane (e.g., "TOP", "JUNGLE", "MIDDLE", "BOTTOM", "UTILITY")
- Role (e.g., "SOLO", "NONE", "DUO", "CARRY", "SUPPORT")

## Region Settings

Change the region in `league_api.py`:
```python
REGION_ROUTING = "americas"   # americas / europe / asia
```

## Error Handling

The functions handle common errors:
- Invalid API key → Returns None
- Player not found → Prints warning with available players
- API rate limits → Returns error message

## Notes

- API keys expire regularly - get new ones at [Riot Developer Portal](https://developer.riotgames.com/)
- Rate limits: 20 requests/second (development keys)
- Match IDs format: `REGION_MATCHNUMBER` (e.g., "NA1_5404818015")

