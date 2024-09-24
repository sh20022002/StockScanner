class strategy_model(strategy):
    def train_lightgbm(self, df: pl.DataFrame, feature_cols: list):
        """
        Trains a LightGBM model on the provided data.

        Args:
            df (pl.DataFrame): The stock data with features and labels.
            feature_cols (list): List of feature column names.
        """
        # Extract features and target
        X = df.select(feature_cols).to_numpy()
        y = df['Signal'].to_numpy()

        # Split the data into training and testing sets (e.g., 80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # shuffle=False to maintain temporal order
        )

        # Initialize the LightGBM Classifier
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=10,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        # Train the model
        self.model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = self.model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)

        print(f"LightGBM Model Accuracy: {accuracy * 100:.2f}%")
        print("Confusion Matrix:")
        print(conf_matrix)
        print("Classification Report:")
        print(class_report)

    def apply_model_signals(self, df: pl.DataFrame, feature_cols: list) -> pl.DataFrame:
        """
        Applies the trained LightGBM model to generate signals.

        Args:
            df (pl.DataFrame): The stock data with features.
            feature_cols (list): List of feature column names.

        Returns:
            pl.DataFrame: DataFrame with model-generated signals.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Call train_lightgbm() first.")

        X = df.select(feature_cols).to_numpy()
        predictions = self.model.predict(X)
        signals_df = pl.DataFrame({
            'Model_Signal': predictions
        }, schema=[
            ('Model_Signal', pl.Int32)
        ]).with_columns([
            pl.col('Model_Signal').cast(pl.Int32)
        ])

        return signals_df
