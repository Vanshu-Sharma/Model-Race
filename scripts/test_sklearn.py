import sys
sys.path.insert(0, r'C:\Users\vansh\OneDrive\Desktop\Model_Race')
from race import load_data, run_sklearn
dataset = load_data()
result = run_sklearn(dataset)
print(f'Accuracy:    {result.accuracy:.4f}')
print(f'F1:          {result.f1:.4f}')
print(f'Precision:   {result.precision:.4f}')
print(f'Recall:      {result.recall:.4f}')
print(f'Cost USD:    {result.cost_usd:.6f}')
print(f'Time/review: {result.avg_time_ms:.4f}ms')
print(f'Test size:   {len(dataset.X_test)}')
print(f'Train size:  {len(dataset.X_train)}')
print('SKLEARN OK')
