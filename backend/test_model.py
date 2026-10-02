from xray_model import load_model, DISEASES

print("Testing model setup...")
print(f"Total diseases model can detect: {len(DISEASES)}")
print("Diseases list:")
for i, d in enumerate(DISEASES):
    print(f"  {i+1}. {d}")

print("\nLoading model...")
model, device = load_model()
print(f"✅ Model ready on: {device}")