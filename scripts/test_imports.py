import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.resolve()))

try:
    print("Testing imports...")
    import model.efficientnet
    import model.gradcam
    import preprocessing.image_quality
    import preprocessing.crop
    import preprocessing.enhance
    import preprocessing.pipeline
    import preprocessing.transforms
    import training.config
    import training.dataset
    import training.losses
    import training.metrics
    import training.train
    import training.evaluate
    import inference.predictor
    import inference.ordinal
    import inference.recommendations
    import inference.report
    import backend.main
    import backend.database
    import backend.models
    import backend.schemas
    print("SUCCESS: All module imports loaded successfully without errors!")
except Exception as e:
    print(f"FAILED: Import error: {e}")
    sys.exit(1)
