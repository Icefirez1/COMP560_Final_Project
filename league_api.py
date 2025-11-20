import requests
import os
import pandas as pd
import pickle
from dotenv import load_dotenv

# load the specific environment file
load_dotenv('.env.local')

API_KEY = os.getenv("LEAGUE_API_KEY")
REGION_ROUTING = "americas"   # switch between americas / europe / asia

# load the trained model and preset rank labels
MODEL = None
RANKS = ["Unranked", "Iron", "Bronze", "Silver", "Gold", "Platinum", 
         "Emerald", "Diamond", "Master", "Grandmaster", "Challenger"]

def load_model():
    """Load the trained model (lazy loading)"""
    global MODEL
    if MODEL is None:
        MODEL = pickle.load(open("models/vanilla_tree.sav", "rb"))
    return MODEL

# encoding mappings for categorical variables (consistent with training data)
LANE_ENCODING = {
    'BOTTOM': 0,
    'SUPPORT': 1,  
    'NONE': 2,
    'JUNGLE': 3,
    'TOP': 4,
    'MIDDLE': 5,
    'UTILITY': 1   # SUPPORT
}
ROLE_ENCODING = {
    'SUPPORT': 0,
    'ADC': 1,
    'NONE': 2,
    'JUNGLE': 3,
    'TOP': 4,
    'MIDDLE': 5,
    'SOLO': 4,     # TOP
    'DUO': 1,      # ADC 
    'CARRY': 1     # ADC
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
        if username and participant['riotIdGameName'].lower() != username.lower():
            continue
        stats = {
            'MinionsKilled': participant['totalMinionsKilled'],
            'DmgDealt': participant['totalDamageDealtToChampions'],
            'DmgTaken': participant['totalDamageTaken'],
            'TurretDmgDealt': participant['damageDealtToTurrets'],
            'TotalGold': participant['goldEarned'],
            'Win': 1 if participant['win'] else 0,
            'item1': participant['item0'],
            'item2': participant['item1'],
            'item3': participant['item2'],
            'item4': participant['item3'],
            'item5': participant['item4'],
            'item6': participant['item5'],
            'kills': participant['kills'],
            'deaths': participant['deaths'],
            'assists': participant['assists'],
            'PrimaryKeyStone': participant['perks']['styles'][0]['selections'][0]['perk'],
            'PrimarySlot1': participant['perks']['styles'][0]['selections'][1]['perk'],
            'PrimarySlot2': participant['perks']['styles'][0]['selections'][2]['perk'],
            'PrimarySlot3': participant['perks']['styles'][0]['selections'][3]['perk'],
            'SecondarySlot1': participant['perks']['styles'][1]['selections'][0]['perk'],
            'SecondarySlot2': participant['perks']['styles'][1]['selections'][1]['perk'],
            'SummonerSpell1': participant['summoner1Id'],
            'SummonerSpell2': participant['summoner2Id'],
            'CurrentMasteryPoints': participant.get('championPoints', 0),
            'DragonKills': participant.get('dragonKills', 0),
            'BaronKills': participant.get('baronKills', 0),
            'visionScore': participant['visionScore'],
            'SummonerMatchId': 0,
            'ChampionFk': participant['championId'],
            'SummonerFk': 0,
            'GameDuration': game_duration_seconds,
            
            # features engineered from the data analysis ipynb, need to do manually here
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
            
            'ChampionId': participant['championId'],
            'Lane': participant['lane'],
            'Role': participant['role'],
            'GamePhase': 'Early' if game_duration_min < 20 else ('Mid' if game_duration_min < 35 else 'Late'),
            'summonerName': participant['riotIdGameName'],
            'championName': participant['championName'],
        }
        
        player_stats_list.append(stats)
    
    df = pd.DataFrame(player_stats_list)
    
    #fill null vals
    df['Lane'] = df['Lane'].map(LANE_ENCODING).fillna(5)
    df['Role'] = df['Role'].map(ROLE_ENCODING).fillna(1)
    df['GamePhase'] = df['GamePhase'].map(GAME_PHASE_ENCODING).fillna(0)
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
    
    return df[model_columns].copy()


def add_rank_predictions(players_df):
    """
    Attach model rank predictions to a player stats DataFrame.
    
    Args:
        players_df: Output of extract_player_stats()
    
    Returns:
        DataFrame with PredictedRank and PredictedRankId columns
    """
    if players_df is None or players_df.empty:
        return None
    
    model_features = prepare_for_model(players_df)
    model = load_model()
    predictions = model.predict(model_features)
    
    with_predictions = players_df.copy()
    with_predictions['PredictedRankId'] = predictions
    with_predictions['PredictedRank'] = [RANKS[int(pred)] for pred in predictions]
    return with_predictions


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
    match_data = get_match_data(match_id)
    
    if not match_data:
        return None
    
    all_players = extract_player_stats(match_data)
    
    if all_players.empty:
        return None
    
    return add_rank_predictions(all_players)


def prepare_prediction_summary(players_df):
    """
    Prepare team tables and summary metrics for UI (e.g., Streamlit).
    
    Args:
        players_df: DataFrame with predictions (from predict_all_players)
    
    Returns:
        Dictionary with team tables, rank distribution, and average rank info
    """
    if players_df is None or players_df.empty:
        return None
    
    if 'PredictedRank' not in players_df.columns:
        raise ValueError("Players DataFrame must include predictions. Call add_rank_predictions first.")
    
    summary_df = players_df.copy()
    summary_df['LaneName'] = summary_df['Lane'].apply(get_lane_name)
    summary_df['KDAString'] = summary_df.apply(
        lambda row: f"{row['kills']}/{row['deaths']}/{row['assists']}", axis=1
    )
    summary_df['CS'] = summary_df['MinionsKilled'].astype(int)
    summary_df['GoldDisplay'] = summary_df['TotalGold'].apply(lambda gold: f"{gold / 1000:.1f}k")
    
    display_cols = [
        'summonerName', 'championName', 'LaneName', 'KDAString',
        'CS', 'GoldDisplay', 'PredictedRank'
    ]
    rename_map = {
        'summonerName': 'Player',
        'championName': 'Champion',
        'LaneName': 'Lane',
        'KDAString': 'KDA',
        'CS': 'CS',
        'GoldDisplay': 'Gold',
        'PredictedRank': 'Predicted Rank'
    }
    
    blue_team = summary_df[summary_df['Win'] == 0][display_cols].rename(columns=rename_map).reset_index(drop=True)
    red_team = summary_df[summary_df['Win'] == 1][display_cols].rename(columns=rename_map).reset_index(drop=True)
    
    rank_counts_series = summary_df['PredictedRank'].value_counts()
    rank_counts = {
        rank: int(rank_counts_series.get(rank, 0))
        for rank in RANKS
        if rank_counts_series.get(rank, 0)
    }
    
    avg_rank_id = float(summary_df['PredictedRankId'].mean())
    avg_rank_id_for_label = max(0, min(len(RANKS) - 1, round(avg_rank_id)))
    avg_rank = RANKS[int(avg_rank_id_for_label)]
    
    return {
        'blue_team': blue_team,
        'red_team': red_team,
        'rank_counts': rank_counts,
        'average_rank_id': avg_rank_id,
        'average_rank': avg_rank,
        'players': summary_df
    }


def get_match_prediction_summary(match_id):
    """
    Convenience helper to fetch a match, compute predictions, and prepare UI data.
    
    Args:
        match_id: The match ID (e.g., "NA1_5404818015")
    
    Returns:
        Dictionary from prepare_prediction_summary() or None on failure.
    """
    players_df = predict_all_players(match_id)
    if players_df is None:
        return None
    return prepare_prediction_summary(players_df)


def display_predictions(players_df):
    """
    Display predictions in a nicely formatted table.
    
    Args:
        players_df: DataFrame with player stats and predictions
    """
    if players_df is None or players_df.empty:
        return
    
    summary = prepare_prediction_summary(players_df)
    if summary is None:
        return
    
    print("Player Rank Predictions:")
    
    def _print_team(label, team_df):
        print(f"\n{label}")
        print(f"{'Player':<20} {'Champion':<12} {'Lane':<8} {'KDA':<12} {'CS':<6} {'Gold':<7} {'Predicted Rank':<15}")
        for _, row in team_df.iterrows():
            print(f"{row['Player']:<20} {row['Champion']:<12} {row['Lane']:<8} "
                  f"{row['KDA']:<12} {row['CS']:<6} {row['Gold']:<7} {row['Predicted Rank']:<15}")
    
    _print_team("BLUE TEAM (set to losing team)", summary['blue_team'])
    _print_team("RED TEAM (set to winning team)", summary['red_team'])
    
    print("\nRank distribution:")
    for rank, count in summary['rank_counts'].items():
        print(f"  {rank}: {count} player(s)")
    
    print(f"\n  Average Rank: {summary['average_rank']} (≈{summary['average_rank_id']:.2f})")


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
    
    output_df = players_df[[
        'summonerName', 'championName', 'kills', 'deaths', 'assists',
        'MinionsKilled', 'TotalGold', 'Win', 'KDA', 'PredictedRank', 'PredictedRankId'
    ]].copy()
    
    output_df.to_csv(filename, index=False)
    print(f"\n Predictions saved at: {filename}")


# Example (for predictions on a give match)
if __name__ == "__main__":
    match_id = "NA1_5416214402"
    
    print("predictions")
    print(f"\nFetching match data for: {match_id}")
    results = predict_all_players(match_id)
    
    if results is not None:
        display_predictions(results)
        save_predictions(results, 'data/match_predictions.csv')
    else:
        print("Failed to fetch or process data")
