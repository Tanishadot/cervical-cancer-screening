from enhanced_transforms import get_domain_robust_train_transforms
import numpy as np

# Test transform
transform = get_domain_robust_train_transforms()
test_img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
result = transform(image=test_img)
print(f'Transform result type: {type(result)}')
print(f'Image type: {type(result["image"]) if "image" in result else "No image key"}')
print(f'Image shape: {result["image"].shape if "image" in result else "No image key"}')
