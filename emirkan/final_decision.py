# final_decision.py
import json
from alert_engine import alert_classification_engine
from consistency_engine import consistency_engine
from report_generator import report_generator

def run_pipeline(rabia_in, onur_in):
    p1 = alert_classification_engine(rabia_in, onur_in)
    p2 = consistency_engine(p1)
    final_output = report_generator(p2)
    return final_output

if __name__ == "__main__":

    import test_cases
    
    # Hangi senaryoyu test etmek istiyorsan onu buraya yaz:
    result = run_pipeline(test_cases.rabia_test_1, test_cases.onur_test_1)
    
    print(json.dumps(result, indent=4, ensure_ascii=False))