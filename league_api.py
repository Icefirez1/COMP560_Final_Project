import requests
import os
import pandas as pd
import pickle
from dotenv import load_dotenv

# Load the specific environment file
load_dotenv('.env.local')

API_KEY = os.getenv("LEAGUE_API_KEY")
print(f"✓ API Key loaded: {API_KEY[:10]}..." if API_KEY else "✗ API Key not found!")
REGION_ROUTING = "americas"   # americas / europe / asia

# Load the trained model and rank labels
MODEL = None
RANKS = ["Unranked", "Iron", "Bronze", "Silver", "Gold", "Platinum", 
         "Emerald", "Diamond", "Master", "Grandmaster", "Challenger"]

def load_model():
    """Load the trained model (lazy loading)"""
    global MODEL
    if MODEL is None:
        MODEL = pickle.load(open("models/vanilla_tree.sav", "rb"))
    return MODEL

# Encoding mappings for categorical variables (consistent with training data)
# These MUST match the exact order from pd.factorize() during training!
# Discovered by running check_encodings.py
LANE_ENCODING = {
    'BOTTOM': 0,
    'SUPPORT': 1,  # Old API lane value
    'NONE': 2,
    'JUNGLE': 3,
    'TOP': 4,
    'MIDDLE': 5,
    'UTILITY': 1   # Maps to SUPPORT for compatibility
}
ROLE_ENCODING = {
    'SUPPORT': 0,
    'ADC': 1,
    'NONE': 2,
    'JUNGLE': 3,
    'TOP': 4,
    'MIDDLE': 5,
    'SOLO': 4,     # Maps to TOP for compatibility
    'DUO': 1,      # Maps to ADC for compatibility  
    'CARRY': 1     # Maps to ADC for compatibility
}
GAME_PHASE_ENCODING = {
    'Mid': 0,
    'Late': 1,
    'Early': 2
}


def get_match_data(match_id):
    """Fetch match data from Riot API"""
    match_url = f"https://{REGION_ROUTING}.api.riotgames.com/lol/match/v5/matches/{match_id}"
    
    response = requests.get(
        match_url,
        headers={"X-Riot-Token": API_KEY}
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None


def extract_player_stats(match_data, username=None):
    """
    Extract player statistics from Riot API match data and format for the ML model.
    
    Args:
        match_data: Match data from Riot API
        username: (Optional) Summoner name to filter for. If None, returns all players.
    
    Returns:
        DataFrame with player statistics formatted for the ML model
    """
    
    participants = match_data['info']['participants']
    game_duration_seconds = match_data['info']['gameDuration']
    game_duration_min = game_duration_seconds / 60
    
    player_stats_list = []
    
    for participant in participants:
        # If username is specified, skip players that don't match
        if username and participant['riotIdGameName'].lower() != username.lower():
            continue
        # Basic stats
        stats = {
            'MinionsKilled': participant['totalMinionsKilled'],
            'DmgDealt': participant['totalDamageDealtToChampions'],
            'DmgTaken': participant['totalDamageTaken'],
            'TurretDmgDealt': participant['damageDealtToTurrets'],
            'TotalGold': participant['goldEarned'],
            'Win': 1 if participant['win'] else 0,
            
            # Items
            'item1': participant['item0'],
            'item2': participant['item1'],
            'item3': participant['item2'],
            'item4': participant['item3'],
            'item5': participant['item4'],
            'item6': participant['item5'],
            
            # KDA
            'kills': participant['kills'],
            'deaths': participant['deaths'],
            'assists': participant['assists'],
            
            # Runes (Perks)
            'PrimaryKeyStone': participant['perks']['styles'][0]['selections'][0]['perk'],
            'PrimarySlot1': participant['perks']['styles'][0]['selections'][1]['perk'],
            'PrimarySlot2': participant['perks']['styles'][0]['selections'][2]['perk'],
            'PrimarySlot3': participant['perks']['styles'][0]['selections'][3]['perk'],
            'SecondarySlot1': participant['perks']['styles'][1]['selections'][0]['perk'],
            'SecondarySlot2': participant['perks']['styles'][1]['selections'][1]['perk'],
            
            # Summoner Spells
            'SummonerSpell1': participant['summoner1Id'],
            'SummonerSpell2': participant['summoner2Id'],
            
            # Champion mastery and objectives
            'CurrentMasteryPoints': participant.get('championPoints', 0),  # Not always available
            'DragonKills': participant.get('dragonKills', 0),
            'BaronKills': participant.get('baronKills', 0),
            'visionScore': participant['visionScore'],
            
            # IDs
            'SummonerMatchId': 0,  # Placeholder
            'ChampionFk': participant['championId'],
            'SummonerFk': 0,  # Placeholder
            'GameDuration': game_duration_seconds,
            
            # Derived features
            'KDA': (participant['kills'] + participant['assists']) / max(participant['deaths'], 1),
            'GameDurationMin': game_duration_min,
            'GoldPerMin': participant['goldEarned'] / game_duration_min,
            'CSPerMin': participant['totalMinionsKilled'] / game_duration_min,
            'DmgPerMin': participant['totalDamageDealtToChampions'] / game_duration_min,
            'VisionPerMin': participant['visionScore'] / game_duration_min,
            'DmgPerGold': participant['totalDamageDealtToChampions'] / max(participant['goldEarned'], 1),
            'DmgEfficiency': participant['totalDamageDealtToChampions'] / max(participant['totalDamageTaken'], 1),
            'ItemCount': sum([1 for i in range(6) if participant[f'item{i}'] != 0]),
            'ObjectiveParticipation': participant.get('dragonKills', 0) + participant.get('baronKills', 0),
            
            # Champion and position
            'ChampionId': participant['championId'],
            'Lane': participant['lane'],
            'Role': participant['role'],
            'GamePhase': 'Early' if game_duration_min < 20 else ('Mid' if game_duration_min < 35 else 'Late'),
            
            # Additional info (not for model)
            'summonerName': participant['riotIdGameName'],
            'championName': participant['championName'],
        }
        
        player_stats_list.append(stats)
    
    df = pd.DataFrame(player_stats_list)
    
    # Encode categorical variables for model compatibility
    df['Lane'] = df['Lane'].map(LANE_ENCODING).fillna(5)  # Default to 5 (NONE) if unknown
    df['Role'] = df['Role'].map(ROLE_ENCODING).fillna(1)  # Default to 1 (NONE) if unknown
    df['GamePhase'] = df['GamePhase'].map(GAME_PHASE_ENCODING).fillna(0)  # Default to 0 (Early)
    
    # If username was specified but not found
    if username and df.empty:
        print(f"⚠ Warning: Player '{username}' not found in this match.")
        print("Available players:")
        for p in participants:
            print(f"  - {p['riotIdGameName']}")
    
    return df


def prepare_for_model(df):
    """
    Prepare player stats DataFrame for model prediction.
    Selects only the columns used in training and ensures correct order.
    
    Args:
        df: DataFrame from extract_player_stats()
    
    Returns:
        DataFrame ready for model.predict()
    """
    # Columns expected by the model (45 features) - matching model training order
    model_columns = [
        'MinionsKilled', 'DmgDealt', 'DmgTaken', 'TurretDmgDealt', 'TotalGold', 'Win',
        'item1', 'item2', 'item3', 'item4', 'item5', 'item6',
        'kills', 'deaths', 'assists',
        'PrimaryKeyStone', 'PrimarySlot1', 'PrimarySlot2', 'PrimarySlot3',
        'SecondarySlot1', 'SecondarySlot2',
        'SummonerSpell1', 'SummonerSpell2',
        'CurrentMasteryPoints', 'DragonKills', 'BaronKills', 'visionScore',
        'SummonerMatchId', 'ChampionFk', 'SummonerFk', 'GameDuration',
        'KDA', 'GameDurationMin', 'GoldPerMin', 'CSPerMin', 'DmgPerMin', 'VisionPerMin',
        'DmgPerGold', 'DmgEfficiency', 'ItemCount', 'ObjectiveParticipation',
        'ChampionId', 'Lane', 'Role', 'GamePhase'
    ]
    
    # Select only the model columns
    return df[model_columns].copy()


def get_player_stats_from_match(match_id, username):
    """
    Fetch a specific player's stats from a match.
    
    Args:
        match_id: The match ID (e.g., "NA1_5404818015")
        username: The player's summoner name
    
    Returns:
        DataFrame with the player's stats, or None if not found
    """
    match_data = get_match_data(match_id)
    
    if not match_data:
        return None
    
    player_df = extract_player_stats(match_data, username=username)
    
    if player_df.empty:
        return None
    
    return player_df


def predict_all_players(match_id):
    """
    Predict ranks for all 10 players in a match.
    
    Args:
        match_id: The match ID (e.g., "NA1_5404818015")
    
    Returns:
        DataFrame with player info and predictions
    """
    # Get match data
    match_data = get_match_data(match_id)
    
    if not match_data:
        return None
    
    # Extract all player stats
    all_players = extract_player_stats(match_data)
    
    if all_players.empty:
        return None
    
    # Prepare features for model
    model_features = prepare_for_model(all_players)
    
    # Load model and make predictions
    model = load_model()
    predictions = model.predict(model_features)
    predicted_ranks = [RANKS[pred] for pred in predictions]
    
    # Add predictions to dataframe
    all_players['PredictedRank'] = predicted_ranks
    all_players['PredictedRankId'] = predictions
    
    return all_players


def display_predictions(players_df):
    """
    Display predictions in a nicely formatted table.
    
    Args:
        players_df: DataFrame with player stats and predictions
    """
    if players_df is None or players_df.empty:
        return
    
    print("\n" + "=" * 100)
    print("RANK PREDICTIONS FOR ALL PLAYERS")
    print("=" * 100)
    
    # Separate by team
    blue_team = players_df[players_df['Win'] == 0].copy()
    red_team = players_df[players_df['Win'] == 1].copy()
    
    # Display Blue Team
    print("\n🔵 BLUE TEAM (Lost)")
    print("-" * 100)
    print(f"{'Player':<20} {'Champion':<12} {'Lane':<8} {'KDA':<12} {'CS':<6} {'Gold':<7} {'Predicted Rank':<15}")
    print("-" * 100)
    
    for _, player in blue_team.iterrows():
        kda_str = f"{player['kills']}/{player['deaths']}/{player['assists']}"
        cs = int(player['MinionsKilled'])
        gold = f"{player['TotalGold']/1000:.1f}k"
        lane_name = get_lane_name(player['Lane'])
        
        print(f"{player['summonerName']:<20} {player['championName']:<12} {lane_name:<8} "
              f"{kda_str:<12} {cs:<6} {gold:<7} {player['PredictedRank']:<15}")
    
    # Display Red Team
    print("\n🔴 RED TEAM (Won)")
    print("-" * 100)
    print(f"{'Player':<20} {'Champion':<12} {'Lane':<8} {'KDA':<12} {'CS':<6} {'Gold':<7} {'Predicted Rank':<15}")
    print("-" * 100)
    
    for _, player in red_team.iterrows():
        kda_str = f"{player['kills']}/{player['deaths']}/{player['assists']}"
        cs = int(player['MinionsKilled'])
        gold = f"{player['TotalGold']/1000:.1f}k"
        lane_name = get_lane_name(player['Lane'])
        
        print(f"{player['summonerName']:<20} {player['championName']:<12} {lane_name:<8} "
              f"{kda_str:<12} {cs:<6} {gold:<7} {player['PredictedRank']:<15}")
    
    # Summary statistics
    print("\n" + "=" * 100)
    print("RANK DISTRIBUTION")
    print("=" * 100)
    rank_counts = players_df['PredictedRank'].value_counts().sort_index()
    for rank, count in rank_counts.items():
        print(f"  {rank:<15}: {count} player(s)")
    
    avg_rank_id = players_df['PredictedRankId'].mean()
    avg_rank = RANKS[int(round(avg_rank_id))]
    print(f"\n  Average Rank: {avg_rank} (≈{avg_rank_id:.2f})")


def get_lane_name(lane_code):
    """Convert lane code back to readable name"""
    lane_map = {0: 'BOTTOM', 1: 'SUPPORT', 2: 'NONE', 3: 'JUNGLE', 4: 'TOP', 5: 'MIDDLE'}
    return lane_map.get(int(lane_code), 'UNKNOWN')


def save_predictions(players_df, filename='match_predictions.csv'):
    """
    Save predictions to CSV file.
    
    Args:
        players_df: DataFrame with player stats and predictions
        filename: Output filename (default: 'match_predictions.csv')
    """
    if players_df is None or players_df.empty:
        return
    
    # Select relevant columns
    output_df = players_df[[
        'summonerName', 'championName', 'kills', 'deaths', 'assists',
        'MinionsKilled', 'TotalGold', 'Win', 'KDA', 'PredictedRank', 'PredictedRankId'
    ]].copy()
    
    output_df.to_csv(filename, index=False)
    print(f"\n✓ Predictions saved to: {filename}")


# Example usage
if __name__ == "__main__":
    match_id = "NA1_5416214402"
    
    print("=" * 100)
    print("LEAGUE OF LEGENDS RANK PREDICTOR")
    print("=" * 100)
    
    # Predict ranks for all players
    print(f"\nFetching match data for: {match_id}")
    results = predict_all_players(match_id)
    
    if results is not None:
        # Display results
        display_predictions(results)
        
        # Save to file
        save_predictions(results, 'data/match_predictions.csv')
        
        print("\n" + "=" * 100)
        print("✓ Analysis complete!")
        print("=" * 100)
    else:
        print("❌ Failed to fetch or process match data")
