import pandas as pd


def test_data():
    data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva'],
    'Age': [24, 27, 22, 32, 29],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Score': [85.5, 90.3, 88.0, 72.5, 91.2]
    }

    df = pd.DataFrame(data)
    return df