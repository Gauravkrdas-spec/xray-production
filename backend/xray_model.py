import torch
import torch.nn.functional as F
import torchxrayvision as xrv
import numpy as np
import skimage.io
import skimage.transform

# 14 diseases CheXNet detects
DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration',
    'Mass', 'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation',
    'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
]

# Anatomical fallback positions for each disease
DISEASE_REGIONS = {
    'Atelectasis':        {'x': 0.50, 'y': 0.72, 'r': 0.07},
    'Cardiomegaly':       {'x': 0.50, 'y': 0.52, 'r': 0.10},
    'Effusion':           {'x': 0.50, 'y': 0.78, 'r': 0.08},
    'Infiltration':       {'x': 0.48, 'y': 0.50, 'r': 0.08},
    'Mass':               {'x': 0.35, 'y': 0.40, 'r': 0.06},
    'Nodule':             {'x': 0.38, 'y': 0.35, 'r': 0.05},
    'Pneumonia':          {'x': 0.50, 'y': 0.55, 'r': 0.08},
    'Pneumothorax':       {'x': 0.22, 'y': 0.28, 'r': 0.07},
    'Consolidation':      {'x': 0.45, 'y': 0.60, 'r': 0.08},
    'Edema':              {'x': 0.50, 'y': 0.65, 'r': 0.09},
    'Emphysema':          {'x': 0.50, 'y': 0.32, 'r': 0.09},
    'Fibrosis':           {'x': 0.50, 'y': 0.60, 'r': 0.07},
    'Pleural_Thickening': {'x': 0.15, 'y': 0.50, 'r': 0.06},
    'Hernia':             {'x': 0.50, 'y': 0.82, 'r': 0.06},
}


# ── Load model ────────────────────────────────────────
def load_model():
    print("Loading pretrained chest X-ray model...")
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Model loaded on: {device}")
    return model, device


# ── Preprocess image ──────────────────────────────────
def preprocess_image(image_path):
    img = skimage.io.imread(image_path)

    # Handle RGBA images
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = img[:, :, :3]

    # Normalize to [-1024, 1024] range for xrv
    img = xrv.datasets.normalize(img, 255)

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        img = img.mean(axis=2)

    # Resize to 224x224
    img = skimage.transform.resize(
        img, (224, 224), anti_aliasing=True
    )

    # Add batch and channel dimensions
    img = img[None, None, :, :]
    return torch.from_numpy(img).float()


# ── Get GradCAM location ──────────────────────────────
def get_gradcam_location(model, image_path, disease_idx, device):
    try:
        activations = []
        gradients   = []

        # Hook to capture forward activations
        def forward_hook(module, input, output):
            activations.append(output)

        # Hook to capture backward gradients
        def backward_hook(module, grad_in, grad_out):
            gradients.append(grad_out[0])

        # Attach hooks to last dense block
        target_layer = model.features.denseblock4
        h1 = target_layer.register_forward_hook(forward_hook)
        h2 = target_layer.register_full_backward_hook(backward_hook)

        # Fresh tensor with gradients enabled
        tensor = preprocess_image(image_path).to(device)
        tensor.requires_grad_(True)

        # Forward pass
        model.zero_grad()
        output = model(tensor)

        # Backward for specific disease index
        score = output[0, disease_idx]
        score.backward()

        # Remove hooks
        h1.remove()
        h2.remove()

        if not activations or not gradients:
            return None

        act  = activations[0].detach()
        grad = gradients[0].detach()

        # Compute GradCAM weights
        weights = grad.mean(dim=[2, 3], keepdim=True)
        cam     = F.relu((weights * act).sum(dim=1).squeeze())

        # Normalize CAM
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        cam_np = cam.cpu().numpy()

        # Find peak location
        flat_idx = cam_np.argmax()
        py, px   = divmod(int(flat_idx), cam_np.shape[1])
        x_norm   = px / cam_np.shape[1]
        y_norm   = py / cam_np.shape[0]

        # Estimate radius from active region size
        threshold   = float(cam_np.max()) * 0.5
        active_pct  = float((cam_np > threshold).sum()) / cam_np.size
        radius      = max(0.04, min(0.12, active_pct * 2.5))
        


        return {
            'x':      float(x_norm),
            'y':      float(y_norm),
            'radius': float(radius)
        }

    except Exception as e:
        print(f"GradCAM error (disease {disease_idx}): {e}")
        return None


# ── Main analysis function ────────────────────────────
def _prior(name):
    region = DISEASE_REGIONS.get(name, {'x': 0.5, 'y': 0.5, 'r': 0.07})
    return {'x': region['x'], 'y': region['y'], 'radius': region['r'], 'loc_source': 'prior'}


def analyze_xray(image_path, model, device, use_gradcam=True, return_probs=False):
    """Classify one radiograph and localise its top-5 findings.

    use_gradcam=False skips the backward passes and uses the anatomical prior
    positions for every marker (the "without GradCAM" ablation arm).
    return_probs=True also returns {disease: probability in [0, 1]} for all 14
    diseases, untruncated, for ROC/AUC evaluation.
    """
    try:
        # Step 1: Get predictions without gradients (fast)
        tensor = preprocess_image(image_path).to(device)
        with torch.no_grad():
            predictions = model(tensor)

        probs = predictions.squeeze().cpu().numpy()

        # Step 2: Build findings list.
        # The model's outputs follow model.pathologies (torchxrayvision order),
        # NOT the NIH order of DISEASES, so each disease is looked up by name.
        all_probs = {}
        findings = []
        for i, disease in enumerate(model.pathologies):
            if disease not in DISEASES or i >= len(probs):
                continue

            prob         = float(probs[i])
            prob         = max(0.0, min(1.0, prob))
            all_probs[disease] = prob
            prob_percent = round(prob * 100, 2)

            if prob_percent > 10:
                findings.append({
                    'name':        disease,
                    'probability': prob_percent,
                    'severity': (
                        'high'   if prob_percent > 50 else
                        'medium' if prob_percent > 30 else
                        'low'
                    ),
                    'disease_idx': i
                })

        missing = [d for d in DISEASES if d not in all_probs]
        if missing:
            raise ValueError(f"Model does not output: {missing}")

        # Sort by probability
        findings.sort(key=lambda x: x['probability'], reverse=True)

        # Step 3: Locations for the top-5 findings (GradCAM, else anatomical prior)
        for finding in findings[:5]:
            idx      = finding.pop('disease_idx')
            location = get_gradcam_location(model, image_path, idx, device) if use_gradcam else None
            if location:
                finding.update(x=location['x'], y=location['y'],
                               radius=location['radius'], loc_source='gradcam')
            else:
                finding.update(_prior(finding['name']))

        # Remaining findings always use the anatomical prior position
        for finding in findings[5:]:
            finding.pop('disease_idx', None)
            finding.update(_prior(finding['name']))

        print(f"Found {len(findings)} findings")
        return (findings, all_probs) if return_probs else findings

    except Exception as e:
        print(f"Analysis error: {e}")
        import traceback
        traceback.print_exc()
