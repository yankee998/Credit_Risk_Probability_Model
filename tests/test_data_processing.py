def test_sample_data_load():
    import pandas as pd
    data = pd.read_csv('C:\\Users\\Skyline\\Credit Risk Probability Model\\data\\raw\\data.csv')
    assert not data.empty, "Data file is empty or not found"