
import sys
import os

# Add local directory to path just in case
sys.path.append(os.getcwd())

try:
    print("Attempting to import inference...")
    from inference import predict_from_user
    print("Successfully imported inference.")
    
    print("Testing prediction on dummy text...")
    sample_text = "We are looking for a data scientist. No experience needed. High salary."
    result = predict_from_user(sample_text)
    print(f"Prediction result: {result['status']} | Reason: {result['reason']} | Words: {result['words']}")
    
    sample_text_2 = "We are looking for a data scientist with 5 years experience. Company name is legit."
    result_2 = predict_from_user(sample_text_2)
    print(f"Prediction result 2: {result_2['status']} | Reason: {result_2['reason']} | Words: {result_2['words']}")
    
    print("Verification successful.")

except Exception as e:
    print(f"Verification failed: {e}")
    import traceback
    traceback.print_exc()
