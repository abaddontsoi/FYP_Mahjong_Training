import json
import os 
import datetime
LOG_ROOT = 'raw_logs'
PONG_ROOT = 'pong_logs'
KONG_ROOT = 'kong_logs'
CHOW_ROOT = 'chow_logs'
PASS_ROOT = 'pass_logs'
DISCARD_ROOT = 'discard_logs'

# Feature encoding scheme:
# - Wind: feature[:1]
# - Round Position: feature[1:2]
# - Hand: feature[2:36] (34 tiles)
# - Self Pongs: feature[36:70] (34 tiles)
# - Self Kongs: feature[70:104] (34 tiles)
# - Self Chows: feature[104:125] (21 tiles)
# - Op Pongs: feature[125:159] (34 tiles)
# - Op Kongs: feature[159:193] (34 tiles)
# - Op Chows: feature[193:214] (21 tiles)
# - Next Pongs: feature[214:248] (34 tiles)
# - Next Kongs: feature[248:282] (34 tiles)
# - Next Chows: feature[282:303] (21 tiles)
# - Prev Pongs: feature[303:337] (34 tiles)
# - Prev Kongs: feature[337:371] (34 tiles)
# - Prev Chows: feature[371:392] (21 tiles)
# - Action tile: feature[392:426] (34 tiles)
# - Discard Pool: feature[426:460] (34 tiles)

def encoding(record: dict,) -> list:
    features = [0] * 460
    
    # Format checking
    if 'current_wind' not in record:
        raise ValueError("Record malformed: missing 'current_wind'")
    if 'round_position' not in record:
        raise ValueError("Record malformed: missing 'round_position'")
    if 'hand' not in record:
        raise ValueError("Record malformed: missing 'hand'")
    if 'called_tuples' not in record:
        raise ValueError("Record malformed: missing 'called_tuples'")
    if 'action' not in record:
        raise ValueError("Record malformed: missing 'action'")
    if 'action_tile' not in record:
        raise ValueError("Record malformed: missing 'action_tile'")
    if 'discard_pool' not in record:
        raise ValueError("Record malformed: missing 'discard_pool'")
    if 'opposite_player_called_tuples' not in record:
        raise ValueError("Record malformed: missing 'opposite_player_called_tuples''")
    if 'next_player_called_tuples' not in record:
        raise ValueError("Record malformed: missing 'next_player_called_tuples'")
    if 'previous_player_called_tuples' not in record:
        raise ValueError("Record malformed: missing 'previous_player_called_tuples'")

    # Wind encoding
    features[0] = record['current_wind'] / 3  # Normalize to [0,1]
    # Round Position encoding
    features[1] = record['round_position'] / 3  # Normalize to [0,1]
    # Hand encoding
    for tile in record['hand']:
        # Hand encoding
        tile_id = tile - 1
        # Apply normalization
        features[2 + tile_id] += 1 / 4
    
    # Self call tuple encoding (assume all tuples are valid)
    for t in record['called_tuples']:
        # Check tuple type
        if len(t) == 4:
            idx = t[0] - 1
            features[70 + idx] += 1
        if len(t) == 3:
            # Pong case
            if t[0] == t[1] == t[2]:
                idx = t[0] - 1
                features[36 + idx] += 1
            else:
                # Chow case
                t.sort()
                idx = t[0] - 1
                # Applly normalization
                features[104 + idx] += 1 / 4

    # Opposite side player call tuple encoding (assume all tuples are valid)
    for t in record['opposite_player_called_tuples']:
        # Check tuple type
        if len(t) == 4:
            idx = t[0] - 1
            features[159 + idx] += 1
        if len(t) == 3:
            # Pong case
            if t[0] == t[1] == t[2]:
                idx = t[0] - 1
                features[125 + idx] += 1
            else:
                # Chow case
                t.sort()
                idx = t[0] - 1
                # Apply normalization
                features[193 + idx] += 1 / 4

    # Next player call tuple encoding (assume all tuples are valid)
    for t in record['next_player_called_tuples']:
        # Check tuple type
        if len(t) == 4:
            idx = t[0] - 1
            features[248 + idx] += 1
        if len(t) == 3:
            # Pong case
            if t[0] == t[1] == t[2]:
                idx = t[0] - 1
                features[214 + idx] += 1
            else:
                # Chow case
                t.sort()
                idx = t[0] - 1
                # Apply normalization
                features[282 + idx] += 1 / 4

    # Previous side player call tuple encoding (assume all tuples are valid)
    for t in record['previous_player_called_tuples']:
        # Check tuple type
        if len(t) == 4:
            idx = t[0] - 1
            features[337 + idx] += 1
        if len(t) == 3:
            # Pong case
            if t[0] == t[1] == t[2]:
                idx = t[0] - 1
                features[303 + idx] += 1
            else:
                # Chow case
                t.sort()
                idx = t[0] - 1
                # Apply normalization
                features[371 + idx] += 1 / 4

    # Action tile encoding
    if record['action_tile']:
        action_tile_id = record['action_tile'] - 1
        features[392 + action_tile_id] = 1

    # Discard pool encoding
    for tile in record['discard_pool']:
        t_id = tile - 1
        # Apply normalization
        features[426 + t_id] += 1 / 4

    return features

def check_possible_calls(hand: list[int], called_tuples: list[list[int]], call_tile: int | None) -> list:
    hand.sort()
    actions = set([])
    # 1-9m: 1-9
    # 1-9p: 10-18
    # 1-9s: 19-27,
    # 1-7z: 28-34,


    # 'kong' and 'pong'
    count = 0
    for t in hand:
        if t == call_tile:
            count += 1
    if count >= 2:
        actions.add('pong')
    if count == 3:
        actions.add('kong')

    # 'additional kong'
    if call_tile is None:
        for t in called_tuples:
            t.sort()
            if len(t) == 3 and t[0] == t[1] == t[2] and t[0] in hand:
                actions.add('kong')

        # Count for 4 consecutive same tiles
        set_hand = set(hand)
        for t in set_hand:
            if hand.count(t) == 4:
                actions.add('kong')

    # 'chow'
    if call_tile is not None and call_tile < 28:
        m_range = range(1, 10)
        s_range = range(10, 19)
        p_range = range(19, 28)

        selected_range = None
        if call_tile in m_range:
            selected_range = m_range
        elif call_tile in s_range:
            selected_range = s_range
        elif call_tile in p_range:
            selected_range = p_range

        if selected_range:
            for idx in range(call_tile - 2, call_tile + 1):
                if idx in selected_range and idx in hand and idx + 1 in hand and idx + 2 in hand:
                    actions.add('chow')

    return list(actions)

def main():
    os.makedirs(PONG_ROOT, exist_ok=True)
    os.makedirs(KONG_ROOT, exist_ok=True)
    os.makedirs(CHOW_ROOT, exist_ok=True)
    os.makedirs(PASS_ROOT, exist_ok=True)
    os.makedirs(DISCARD_ROOT, exist_ok=True)

    log_files = [f for f in os.listdir(LOG_ROOT) if f.startswith('game_log_') and f.endswith('.json')]
    log_files.sort()  # Sort files by name

    pong_record = []
    kong_record = []
    chow_record = []
    pass_record = {
        'kong': [],
        'chow': [],
        'pong': []
    }
    discard_record = []

    for log_file in log_files:
        print(f"Processing log file: {log_file}")
        with open(os.path.join(LOG_ROOT, log_file), 'r') as f:
            log = json.load(f)
            # print(type(log)) # <class 'list'>
            for record in log:
                # if record['player_id'] != 'You':
                #     continue
                
                if record['action'] == 'pong':
                    pong_record.append(record)
                elif record['action'] == 'kong':
                    kong_record.append(record)
                elif record['action'] == 'chow':
                    chow_record.append(record)
                elif record['action'] == 'pass':
                    # Check on possible action
                    if 'pong' in check_possible_calls(record['hand'], record['called_tuples'], record['action_tile']):
                        pass_record['pong'].append(record)
                    if 'kong' in check_possible_calls(record['hand'], record['called_tuples'], record['action_tile']):
                        pass_record['kong'].append(record)
                    if 'chow' in check_possible_calls(record['hand'], record['called_tuples'], record['action_tile']):
                        pass_record['chow'].append(record)

                elif record['action'] == 'discard':
                    discard_record.append(record)

    print(f"Total pong records: {len(pong_record)}")
    print(f"Total pass pong records: {len(pass_record['pong'])}")
    print(f"Total kong records: {len(kong_record)}")
    print(f"Total pass kong records: {len(pass_record['kong'])}")
    print(f"Total chow records: {len(chow_record)}")
    print(f"Total pass chow records: {len(pass_record['chow'])}")
    print(f"Total discard records: {len(discard_record)}")

    # Process and save pong records
    print("Processing pong records...")
    encoded_pong_records = [encoding(r) for r in pong_record]
    # Dump encoded records to json file
    with open(os.path.join(PONG_ROOT, f'pong.json'), 'w') as f:
        json.dump(encoded_pong_records, f, indent=4)

    # Process and save kong records
    print("Processing kong records...")
    encoded_kong_records = [encoding(r) for r in kong_record]
    with open(os.path.join(KONG_ROOT, f'kong.json'), 'w') as f:
        json.dump(encoded_kong_records, f, indent=4)
    
    # Process and save chow records
    print("Processing chow records...")
    encoded_chow_records = [encoding(r) for r in chow_record]
    with open(os.path.join(CHOW_ROOT, f'chow.json'), 'w') as f:
        json.dump(encoded_chow_records, f, indent=4)
    
    # Process and save pass records
    print("Processing pass records...")
    encoded_pass_kong_records = [encoding(r) for r in pass_record['kong']]
    encoded_pass_pong_records = [encoding(r) for r in pass_record['pong']]
    encoded_pass_chow_records = [encoding(r) for r in pass_record['chow']]
    with open(os.path.join(KONG_ROOT, f'pass_kong.json'), 'w') as f:
        json.dump(encoded_pass_kong_records, f, indent=4)
    with open(os.path.join(PONG_ROOT, f'pass_pong.json'), 'w') as f:
        json.dump(encoded_pass_pong_records, f, indent=4)
    with open(os.path.join(CHOW_ROOT, f'pass_chow.json'), 'w') as f:
        json.dump(encoded_pass_chow_records, f, indent=4)

    # Process and save discard records
    print("Processing discard records...")
    encoded_discard_records = [encoding(r) for r in discard_record]
    with open(os.path.join(DISCARD_ROOT, f'discard.json'), 'w') as f:
        json.dump(encoded_discard_records, f, indent=4)

if __name__ == "__main__":
    main()