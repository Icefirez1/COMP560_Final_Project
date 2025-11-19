"""
Predict ranks for all players in a League of Legends match
"""

from league_api import get_match_data, extract_player_stats, prepare_for_model
import pickle
import pandas as pd

# Load the trained model
print("Loading model...")
model = pickle.load(open("models/vanilla_tree.sav", "rb"))
ranks = ["Unranked", "Iron", "Bronze", "Silver", "Gold", "Platinum", 
         "Emerald", "Diamond", "Master", "Grandmaster", "Challenger"]

def predict_all_players(match_id):
    """
    Predict ranks for all 10 players in a match.
    
    Args:
        match_id: The match ID (e.g., "NA1_5404818015")
    
    Returns:
        DataFrame with player info and predictions
    """
    print(f"\nFetching match data for: {match_id}")
    print("=" * 80)
    
    # Get match data
    match_data = get_match_data(match_id)
    
    if not match_data:
        print("❌ Failed to fetch match data")
        return None
    
    # Extract all player stats
    all_players = extract_player_stats(match_data)
    
    if all_players.empty:
        print("❌ No player data found")
        return None
    
    # Match info
    game_duration = match_data['info']['gameDuration']
    game_mode = match_data['info']['gameMode']
    
    print(f"✓ Match found: {game_mode}")
    print(f"✓ Duration: {game_duration // 60} minutes {game_duration % 60} seconds")
    print(f"✓ Players: {len(all_players)}\n")
    
    # Prepare features for model
    model_features = prepare_for_model(all_players)
    
    # Make predictions for all players
    predictions = model.predict(model_features)
    predicted_ranks = [ranks[pred] for pred in predictions]
    
    # Add predictions to dataframe
    all_players['PredictedRank'] = predicted_ranks
    all_players['PredictedRankId'] = predictions
    
    return all_players


def display_predictions(players_df):
    """
    Display predictions in a nicely formatted table.
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
        
        print(f"{player['summonerName']:<20} {player['championName']:<12} {get_lane_name(player['Lane']):<8} "
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
        
        print(f"{player['summonerName']:<20} {player['championName']:<12} {get_lane_name(player['Lane']):<8} "
              f"{kda_str:<12} {cs:<6} {gold:<7} {player['PredictedRank']:<15}")
    
    # Summary statistics
    print("\n" + "=" * 100)
    print("RANK DISTRIBUTION")
    print("=" * 100)
    rank_counts = players_df['PredictedRank'].value_counts().sort_index()
    for rank, count in rank_counts.items():
        print(f"  {rank:<15}: {count} player(s)")
    
    avg_rank_id = players_df['PredictedRankId'].mean()
    avg_rank = ranks[int(round(avg_rank_id))]
    print(f"\n  Average Rank: {avg_rank} (≈{avg_rank_id:.2f})")


def get_lane_name(lane_code):
    """Convert lane code back to readable name"""
    lane_map = {0: 'BOTTOM', 1: 'SUPPORT', 2: 'NONE', 3: 'JUNGLE', 4: 'TOP', 5: 'MIDDLE'}
    return lane_map.get(int(lane_code), 'UNKNOWN')


def save_predictions(players_df, filename='match_predictions.csv'):
    """
    Save predictions to CSV file.
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


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Match ID to analyze
    match_id = "NA1_5416850084"
    
    # Predict ranks for all players
    results = predict_all_players(match_id)
    
    if results is not None:
        # Display results
        display_predictions(results)
        
        # Save to file
        save_predictions(results, 'data/match_predictions.csv')
        
        print("\n" + "=" * 100)
        print("✓ Analysis complete!")
        print("=" * 100)

