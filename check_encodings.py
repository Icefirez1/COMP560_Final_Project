"""
Script to discover the actual encodings used during model training
"""
import pandas as pd

# Load the training data
df = pd.read_csv('data/league_match_stats_cleaned.csv')

# Run the EXACT same factorize as training
lane_codes, lane_unique = pd.factorize(df['Lane'])
role_codes, role_unique = pd.factorize(df['Role'])
phase_codes, phase_unique = pd.factorize(df['GamePhase'])

print("=" * 60)
print("ACTUAL ENCODINGS USED IN MODEL TRAINING")
print("=" * 60)

print("\nLANE MAPPING:")
for i, lane in enumerate(lane_unique):
    print(f"  '{lane}' = {i}")

print("\nROLE MAPPING:")
for i, role in enumerate(role_unique):
    print(f"  '{role}' = {i}")

print("\nGAME PHASE MAPPING:")
for i, phase in enumerate(phase_unique):
    print(f"  '{phase}' = {i}")

print("\n" + "=" * 60)
print("Compare with league_api.py encodings:")
print("=" * 60)

from league_api import LANE_ENCODING, ROLE_ENCODING, GAME_PHASE_ENCODING

print("\nLANE_ENCODING in league_api.py:")
for lane, code in LANE_ENCODING.items():
    print(f"  '{lane}' = {code}")

print("\nROLE_ENCODING in league_api.py:")
for role, code in ROLE_ENCODING.items():
    print(f"  '{role}' = {code}")

print("\nGAME_PHASE_ENCODING in league_api.py:")
for phase, code in GAME_PHASE_ENCODING.items():
    print(f"  '{phase}' = {code}")

# Check for mismatches
print("\n" + "=" * 60)
print("VERIFICATION:")
print("=" * 60)

mismatches = []
for i, lane in enumerate(lane_unique):
    if lane in LANE_ENCODING:
        if LANE_ENCODING[lane] != i:
            mismatches.append(f"LANE: '{lane}' should be {i}, but is {LANE_ENCODING[lane]}")

for i, role in enumerate(role_unique):
    if role in ROLE_ENCODING:
        if ROLE_ENCODING[role] != i:
            mismatches.append(f"ROLE: '{role}' should be {i}, but is {ROLE_ENCODING[role]}")

for i, phase in enumerate(phase_unique):
    if phase in GAME_PHASE_ENCODING:
        if GAME_PHASE_ENCODING[phase] != i:
            mismatches.append(f"PHASE: '{phase}' should be {i}, but is {GAME_PHASE_ENCODING[phase]}")

if mismatches:
    print("\n⚠️  MISMATCHES FOUND:")
    for mismatch in mismatches:
        print(f"  ❌ {mismatch}")
else:
    print("\n✅ All encodings match perfectly!")

