from pathlib import Path

import pandas as pd


DATA_FILE_PATH = Path("sms_spam_collection/SMSSpamCollection.tsv")


def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0]
    ham_subset = df[df["Label"] == "ham"].sample(
        num_spam, random_state=123
    )
    return pd.concat(
        [ham_subset, df[df["Label"] == "spam"]],
        ignore_index=True,
    )


def random_split(df, train_frac, validation_frac):
    if not 0 <= train_frac <= 1 or not 0 <= validation_frac <= 1:
        raise ValueError("train_frac and validation_frac must be between 0 and 1")
    if train_frac + validation_frac > 1:
        raise ValueError("train_frac + validation_frac cannot exceed 1")

    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df


def main():
    if not DATA_FILE_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE_PATH}. Run download_sms_spam.py first."
        )

    df = pd.read_csv(
        DATA_FILE_PATH,
        sep="\t",
        header=None,
        names=["Label", "Text"],
    )
    balanced_df = create_balanced_dataset(df)
    print("Balanced label counts:")
    print(balanced_df["Label"].value_counts().to_string())

    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
    train_df, validation_df, test_df = random_split(
        balanced_df, train_frac=0.7, validation_frac=0.1
    )

    train_df.to_csv("train.csv", index=False)
    validation_df.to_csv("validation.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    for name, split_df in (
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ):
        print(f"\n{name}: {len(split_df)} rows")
        print(split_df["Label"].value_counts().sort_index().to_string())
    print("\nSaved train.csv, validation.csv, and test.csv")


if __name__ == "__main__":
    main()
