"""Entry point: run the Adult income pipeline step by step from the command line."""

import argparse
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model


def main():
    parser = argparse.ArgumentParser(description="Adult income ML pipeline")
    parser.add_argument("--prepare", action="store_true", help="Only prepare the data")
    parser.add_argument("--train", action="store_true", help="Train and save the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the saved model")
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = prepare_data("adult.data", "adult.test")

    if args.prepare:
        print(f"Data ready: train {X_train.shape}, test {X_test.shape}")
    elif args.train:
        model = train_model(X_train, y_train)
        save_model(model)
    elif args.evaluate:
        model = load_model()
        evaluate_model(model, X_test, y_test)
    else:
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        save_model(model)


if __name__ == "__main__":
    main()
