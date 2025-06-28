import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class ChurnPredictionModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.feature_names = []
        self.model_path = 'churn_model.pkl'
        self.scaler_path = 'churn_scaler.pkl'
        self.encoders_path = 'churn_encoders.pkl'
        
    def load_data(self, file_path='churn_prediction_dataset/ecommerce_customer_data.csv'):
        """Load and prepare the churn prediction dataset"""
        print(f"🔍 DEBUG: load_data() called with file_path: {file_path}")
        try:
            print(f"🔍 DEBUG: About to read CSV file: {file_path}")
            df = pd.read_csv(file_path)
            print(f"🔍 DEBUG: CSV file read successfully")
            print(f"✅ Dataset loaded successfully: {df.shape[0]} customers, {df.shape[1]} features")
            print(f"🔍 DEBUG: DataFrame columns: {list(df.columns)}")
            print(f"🔍 DEBUG: First few rows:")
            print(df.head(3))
            return df
        except Exception as e:
            print(f"🔍 DEBUG: Exception in load_data: {str(e)}")
            print(f"🔍 DEBUG: Exception type: {type(e)}")
            import traceback
            print(f"🔍 DEBUG: Full traceback: {traceback.format_exc()}")
            print(f"❌ Error loading dataset: {str(e)}")
            return None
    
    def preprocess_data(self, df):
        """Preprocess the data for churn prediction"""
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Drop unnecessary columns
        columns_to_drop = ['customer_id', 'name', 'email']
        data = data.drop(columns=[col for col in columns_to_drop if col in data.columns])
        
        # Handle categorical variables
        categorical_columns = ['gender']
        for col in categorical_columns:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                self.label_encoders[col] = le
        
        # Create additional features
        data['purchase_frequency'] = data['total_purchases'] / (data['days_since_last_purchase'] + 1)
        data['total_spent'] = data['total_purchases'] * data['avg_purchase_value']
        data['avg_days_between_purchases'] = data['days_since_last_purchase'] / (data['total_purchases'] + 1)
        
        # Handle outliers in numeric columns
        numeric_columns = ['age', 'annual_income', 'total_purchases', 'avg_purchase_value', 
                          'days_since_last_purchase', 'customer_satisfaction']
        
        for col in numeric_columns:
            if col in data.columns:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                data[col] = np.where((data[col] < lower_bound) | (data[col] > upper_bound),
                                   data[col].median(), data[col])
        
        # Separate features and target
        if 'churn' in data.columns:
            X = data.drop('churn', axis=1)
            y = data['churn']
        else:
            print("❌ No 'churn' column found in dataset")
            return None, None
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"✅ Data preprocessing completed. Features: {len(self.feature_names)}")
        print(f"✅ Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
    
    def select_features(self, X, y, k=10):
        """Select the best features for the model"""
        self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = self.feature_selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        print(f"✅ Selected {len(selected_features)} best features: {selected_features}")
        
        return X_selected, selected_features
    
    def train_model(self, X, y, model_type='random_forest'):
        """Train the churn prediction model"""
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Select best model
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        elif model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=6)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        print(f"✅ Model training completed!")
        print(f"📊 Accuracy: {accuracy:.4f}")
        print(f"📊 AUC Score: {auc_score:.4f}")
        print(f"📊 Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        print(f"📊 Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return accuracy, auc_score
    
    def save_model(self):
        """Save the trained model and preprocessing objects"""
        try:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            joblib.dump(self.label_encoders, self.encoders_path)
            print(f"✅ Model saved successfully!")
        except Exception as e:
            print(f"❌ Error saving model: {str(e)}")
    
    def load_model(self):
        """Load the trained model and preprocessing objects"""
        try:
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            self.label_encoders = joblib.load(self.encoders_path)
            print(f"✅ Model loaded successfully!")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
    def predict_churn(self, customer_data):
        """Predict churn probability for a single customer"""
        if self.model is None:
            print("❌ Model not trained or loaded")
            return None
        
        try:
            # Preprocess the customer data
            processed_data = self.preprocess_single_customer(customer_data)
            
            # Scale the features
            scaled_data = self.scaler.transform([processed_data])
            
            # Make prediction
            churn_probability = self.model.predict_proba(scaled_data)[0][1]
            churn_prediction = self.model.predict(scaled_data)[0]
            
            return {
                'churn_probability': churn_probability,
                'churn_prediction': churn_prediction,
                'risk_level': self.get_risk_level(churn_probability)
            }
        except Exception as e:
            print(f"❌ Error making prediction: {str(e)}")
            return None
    
    def preprocess_single_customer(self, customer_data):
        """Preprocess a single customer's data"""
        # Create feature vector
        features = []
        
        # Basic features
        features.extend([
            customer_data.get('age', 0),
            customer_data.get('annual_income', 0),
            customer_data.get('total_purchases', 0),
            customer_data.get('avg_purchase_value', 0),
            customer_data.get('days_since_last_purchase', 0),
            customer_data.get('customer_satisfaction', 0)
        ])
        
        # Handle gender encoding
        if 'gender' in customer_data:
            gender = customer_data['gender']
            if 'gender' in self.label_encoders:
                features.append(self.label_encoders['gender'].transform([gender])[0])
            else:
                features.append(1 if gender.lower() == 'male' else 0)
        else:
            features.append(0)
        
        # Additional features
        total_purchases = customer_data.get('total_purchases', 0)
        days_since = customer_data.get('days_since_last_purchase', 0)
        avg_purchase = customer_data.get('avg_purchase_value', 0)
        
        features.extend([
            total_purchases / (days_since + 1),  # purchase_frequency
            total_purchases * avg_purchase,      # total_spent
            avg_purchase / (days_since + 1)      # avg_days_between_purchases
        ])
        
        return features
    
    def get_risk_level(self, probability):
        """Get risk level based on churn probability"""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.6:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def get_feature_importance(self):
        """Get feature importance if available"""
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance))
            return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
        return None

def train_churn_model():
    """Main function to train the churn prediction model"""
    print("🚀 DEBUG: Starting Churn Prediction Model Training...")
    
    # Initialize model
    print("🔍 DEBUG: Initializing ChurnPredictionModel")
    churn_model = ChurnPredictionModel()
    
    # Load data
    print("🔍 DEBUG: About to load data")
    df = churn_model.load_data()
    print(f"🔍 DEBUG: load_data() returned: {df is not None}")
    if df is None:
        print("🔍 DEBUG: Failed to load data")
        return None
    
    # Preprocess data
    print("🔍 DEBUG: About to preprocess data")
    X, y = churn_model.preprocess_data(df)
    print(f"🔍 DEBUG: preprocess_data() returned X shape: {X.shape if X is not None else 'None'}")
    print(f"🔍 DEBUG: preprocess_data() returned y shape: {y.shape if y is not None else 'None'}")
    if X is None or y is None:
        print("🔍 DEBUG: Failed to preprocess data")
        return None
    
    # Select features
    print("🔍 DEBUG: About to select features")
    X_selected, selected_features = churn_model.select_features(X, y, k=10)
    print(f"🔍 DEBUG: select_features() returned X_selected shape: {X_selected.shape}")
    print(f"🔍 DEBUG: selected_features: {selected_features}")
    
    # Train model
    print("🔍 DEBUG: About to train model")
    accuracy, auc_score = churn_model.train_model(X_selected, y, model_type='random_forest')
    print(f"🔍 DEBUG: train_model() returned accuracy: {accuracy}, auc_score: {auc_score}")
    
    # Save model
    print("🔍 DEBUG: About to save model")
    churn_model.save_model()
    
    # Get feature importance
    print("🔍 DEBUG: About to get feature importance")
    feature_importance = churn_model.get_feature_importance()
    if feature_importance:
        print(f"🔍 DEBUG: Top 5 Most Important Features:")
        for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
            print(f"   {i+1}. {feature}: {importance:.4f}")
    
    print("🎉 DEBUG: Churn prediction model training completed!")
    return churn_model

if __name__ == "__main__":
    train_churn_model() 