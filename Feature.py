# Step 3: Feature Engineering

def extract_features(data):
    # Initialize features list
    features = []

    # Loop through all tokens
    for index, row in data.iterrows():
        token = row['Token']

        # Extract features
        feature = {
            'word': str(token),
            'length': len(str(token)),
            'all_capital': 1 if str(token).isupper() else 0,
            'next_word': str(data.loc[int(index) + 1, 'Token']) if index != data.shape[0] - 1 else '',
            'next_word2': str(data.loc[int(index) + 2, 'Token']) if index != data.shape[0] - 2 and index != data.shape[
                0] - 1 else '',
            'prev_word': str(data.loc[int(index) - 1, 'Token']) if index != 0 else '',
            'prev_word2': str(data.loc[int(index) - 2, 'Token']) if index != 1 and index != 0 else '',
            'prefix': str(token)[0:2],
            'suffix': str(token)[-2:]
        }

        # Append features to list
        features.append(feature)
    return features
