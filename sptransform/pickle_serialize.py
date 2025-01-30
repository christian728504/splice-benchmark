def save(data, name):
    """Save object to a pickle file in current directory."""
    with open(f"{name}.pkl", 'wb') as f:
        pickle.dump(data, f)

def load(name):
    """Load object from a pickle file in current directory."""
    with open(f"{name}.pkl", 'rb') as f:
        return pickle.load(f)