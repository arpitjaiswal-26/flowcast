import os
import sys
from src.clean import load_and_clean_data
from src.ml_models import train_classical_models
from src.dl_model import train_dl_model

def main():
    print("======================================================")
    print("   FLOWCAST AI PLATFORM: END-TO-END EXECUTION RUN   ")
    print("======================================================")
    
    # Step 1: Clean & Merge Data Streams
    df = load_and_clean_data()
    
    # Step 2: Train Classical Machine Learning & NumPy LR
    ml_metrics = train_classical_models()
    
    # Step 3: Train PyTorch LSTM Neural Network
    dl_metrics = train_dl_model()
    
    print("\n======================================================")
    print("      PIPELINE EXECUTION COMPLETED SUCCESSFULLY       ")
    print("======================================================")
    print("Run `streamlit run dashboard/app.py` to launch the interactive dashboard.")

if __name__ == "__main__":
    main()