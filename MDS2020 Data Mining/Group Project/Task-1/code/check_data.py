import pandas as pd


def check():
    data_path = "..\\data\\"
    output_path = '..\\output\\'

    train_data = pd.read_csv(f"{data_path}bank_marketing_train.csv")
    test_data = pd.read_csv(f"{data_path}bank_marketing_test.csv")

    with open(f"{output_path}data_check.txt", "w") as f:
        for sample_data in [train_data, test_data]:
            # write DataFrame.info output directly to the file
            sample_data.info(buf=f)
            # write summaries and counts to the file
            print(sample_data["age"].describe(), file=f, end="\n\n")
            print(sample_data["job"].value_counts(), file=f, end="\n\n")
            print(sample_data["marital"].value_counts(), file=f, end="\n\n")
            print(sample_data["education"].value_counts(), file=f, end="\n\n")
            print(sample_data["default"].value_counts(), file=f, end="\n\n")
            print(sample_data["housing"].value_counts(), file=f, end="\n\n")
            print(sample_data["loan"].value_counts(), file=f, end="\n\n")
            print(sample_data["contact"].value_counts(), file=f, end="\n\n")
            print(sample_data["month"].value_counts(), file=f, end="\n\n")
            print(sample_data["day_of_week"].value_counts(), file=f, end="\n\n")
            print(sample_data["campaign"].describe(), file=f, end="\n\n")
            print(sample_data["pdays"].describe(), file=f, end="\n\n")
            print(sample_data["previous"].describe(), file=f, end="\n\n")
            print(sample_data["poutcome"].value_counts(), file=f, end="\n\n")
            print(sample_data["emp.var.rate"].describe(), file=f, end="\n\n")
            print(sample_data["cons.price.idx"].describe(), file=f, end="\n\n")
            print(sample_data["cons.conf.idx"].describe(), file=f, end="\n\n")
            print(sample_data["euribor3m"].describe(), file=f, end="\n\n")
            print(sample_data["nr.employed"].describe(), file=f, end="\n\n")
            print(sample_data["feature_1"].describe(), file=f, end="\n\n")
            print(sample_data["feature_2"].describe(), file=f, end="\n\n")
            print(sample_data["feature_3"].describe(), file=f, end="\n\n")
            print(sample_data["feature_4"].describe(), file=f, end="\n\n")
            print(sample_data["feature_5"].describe(), file=f, end="\n\n")
