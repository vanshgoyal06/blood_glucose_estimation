from sklearn.model_selection import train_test_split

def random_row_split(df, feature_columns, target_column="Blood Glucose Level(BGL)", seed=42):
    """
    Legacy 70/15/15 row-level split used for the 7.8812 validation result.
    Age is used only for stratification, not grouping.
    """
    X = df[feature_columns]
    y = df[target_column]
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed, stratify=df.loc[X.index, "Age"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed,
        stratify=X_temp.index.map(df["Diabetic/NonDiabetic (D/N)"])
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def participant_split(df, feature_columns, train_participants, val_participants, test_participants,
                      participant_column="Age", target_column="Blood Glucose Level(BGL)"):
    train_idx=df.index[df[participant_column].isin(train_participants)]
    val_idx=df.index[df[participant_column].isin(val_participants)]
    test_idx=df.index[df[participant_column].isin(test_participants)]
    return (
        df.loc[train_idx, feature_columns], df.loc[val_idx, feature_columns], df.loc[test_idx, feature_columns],
        df.loc[train_idx, target_column], df.loc[val_idx, target_column], df.loc[test_idx, target_column]
    )
