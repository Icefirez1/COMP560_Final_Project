import requests
import os
import pandas as pd
from dotenv import load_dotenv

# Load the specific environment file
load_dotenv('keys.env.local')

API_KEY = os.getenv("LEAGUE_API_KEY")
print(f"✓ API Key loaded: {API_KEY[:10]}..." if API_KEY else "✗ API Key not found!")
REGION_ROUTING = "americas"   # americas / europe / asia

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


# Example usage
if __name__ == "__main__":
    match_id = "NA1_5404818015"
    target_player = "coolkaw"  # Change this to the player you want to analyze
    
    print("=" * 80)
    print("OPTION 1: Get specific player stats")
    print("=" * 80)
    
    player_stats = get_player_stats_from_match(match_id, target_player)
    
    if player_stats is not None and not player_stats.empty:
        print(f"\n✓ Stats retrieved for player: {target_player}")
        print(f"  Champion: {player_stats['championName'].values[0]}")
        print(f"  KDA: {player_stats['kills'].values[0]}/{player_stats['deaths'].values[0]}/{player_stats['assists'].values[0]}")
        print(f"  Win: {'Yes' if player_stats['Win'].values[0] == 1 else 'No'}")
        print(f"  Gold/Min: {player_stats['GoldPerMin'].values[0]:.1f}")
        print(f"  CS/Min: {player_stats['CSPerMin'].values[0]:.1f}")
        print(f"  Lane: {player_stats['Lane'].values[0]} (encoded)")
        print(f"  Role: {player_stats['Role'].values[0]} (encoded)")
        
        # Save individual player stats for model
        output_file = f'data/{target_player}_stats.csv'
        model_df = prepare_for_model(player_stats)
        model_df.to_csv(output_file, index=False)
        print(f"\n✓ Player data saved to {output_file}")
        print(f"  Ready for rank prediction!")
    
    print("\n" + "=" * 80)
    print("OPTION 2: Get all players from match")
    print("=" * 80)
    
    match_data = get_match_data(match_id)
    
    if match_data:
        print(f"\n✓ Match data retrieved successfully!")
        print(f"  Game Duration: {match_data['info']['gameDuration'] // 60} minutes")
        print(f"  Game Mode: {match_data['info']['gameMode']}\n")
        
        # Extract all player stats
        all_players_df = extract_player_stats(match_data)
        
        print("All Player Statistics:")
        print("-" * 80)
        print(all_players_df[['summonerName', 'championName', 'kills', 'deaths', 'assists', 'Win', 'KDA']].to_string(index=False))
        
        # Save to CSV (model-ready format)
        output_file = 'data/fetched_match_data.csv'
        model_df = prepare_for_model(all_players_df)
        model_df.to_csv(output_file, index=False)
        print(f"\n✓ All match data saved to {output_file}")
        print(f"  {len(model_df)} players | {len(model_df.columns)} features (model-ready)")
