import data_iris


def test_iris_load():
    df = data_iris.iris_df()
    assert df.shape == (148, 5)


def test_one_hot():
    df = data_iris.iris_df()
    one_hot_df = data_iris.one_hot_classes(df)
    assert one_hot_df.shape == (148, 7)
    assert "Species" not in one_hot_df.columns


def test_features_preds():
    df = data_iris.iris_df()
    one_hot_df = data_iris.one_hot_classes(df)
    feats_df = data_iris.data_features(one_hot_df)
    assert feats_df.shape == (148, 4)

    pred_df = data_iris.data_pred(one_hot_df)
    assert pred_df.shape == (148, 3)
