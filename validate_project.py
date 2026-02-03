import sys
from pathlib import Path

def check_project_structure():
    required_files = [
        "src/models/efficientnet.py",
        "src/models/resnet.py",
        "src/models/vit.py",
        "src/data/dataset.py",
        "src/data/transforms.py",
        "src/data/datamodule.py",
        "src/evaluation/metrics.py",
        "src/explainability/gradcam.py",
        "src/utils/seed.py",
        "src/utils/config_loader.py",
        "configs/model.yaml",
        "configs/training.yaml",
        "configs/dataset.yaml",
        "requirements.txt",
        "README.md",
    ]
    
    project_root = Path(__file__).parent
    missing = []
    
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing.append(file_path)
    
    if missing:
        print("❌ Missing files:")
        for f in missing:
            print(f"  - {f}")
        return False
    
    print("✅ All required files present")
    return True


def check_imports():
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        
        from src.models.efficientnet import EfficientNetClassifier, build_efficientnet
        from src.models.resnet import ResNetClassifier, build_resnet
        from src.models.vit import ViTClassifier, build_vit
        from src.evaluation.metrics import quadratic_weighted_kappa, compute_metrics
        from src.utils.seed import set_seed
        from src.utils.config_loader import ConfigManager
        
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False


def check_config_files():
    import yaml
    
    project_root = Path(__file__).parent
    configs = ["model.yaml", "training.yaml", "dataset.yaml"]
    
    for config_file in configs:
        config_path = project_root / "configs" / config_file
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f)
            if data is None:
                print(f"⚠️  {config_file} is empty")
            else:
                print(f"✅ {config_file} valid")
        except Exception as e:
            print(f"❌ Error reading {config_file}: {e}")
            return False
    
    return True


if __name__ == "__main__":
    print("🔍 Validating Diabetic Retinopathy Detection Project\n")
    
    checks = [
        ("Project Structure", check_project_structure),
        ("Configuration Files", check_config_files),
        ("Python Imports", check_imports),
    ]
    
    all_passed = True
    for name, check_func in checks:
        print(f"\n{name}:")
        if not check_func():
            all_passed = False
    
    print("\n" + "="*50)
    if all_passed:
        print("✅ All checks passed! Project is ready.")
    else:
        print("❌ Some checks failed. Please review above.")
    
    sys.exit(0 if all_passed else 1)
