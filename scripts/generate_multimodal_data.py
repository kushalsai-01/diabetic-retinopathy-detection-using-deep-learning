"""
Generate Sample Multimodal Dataset

Creates synthetic patient clinical data for demonstration.
In production, replace with real patient data.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import argparse


def generate_patient_data(
    num_samples: int,
    dr_severity: int,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate realistic patient clinical data based on DR severity."""
    if seed is not None:
        np.random.seed(seed)
    
    # Age: Higher for severe DR
    age_mean = 45 + dr_severity * 5
    age = np.clip(np.random.normal(age_mean, 10, num_samples), 18, 90)
    
    # Gender: Random
    gender = np.random.binomial(1, 0.5, num_samples)
    
    # Diabetes duration: Longer for severe DR
    duration_mean = 5 + dr_severity * 2
    diabetes_duration = np.clip(np.random.normal(duration_mean, 3, num_samples), 0, 40)
    
    # HbA1c: Higher for severe DR (normal: <5.7, prediabetes: 5.7-6.4, diabetes: >6.5)
    hba1c_mean = 6.5 + dr_severity * 0.5
    hba1c = np.clip(np.random.normal(hba1c_mean, 0.8, num_samples), 5.0, 14.0)
    
    # Blood pressure: Higher for severe DR
    bp_sys_mean = 125 + dr_severity * 5
    bp_sys = np.clip(np.random.normal(bp_sys_mean, 15, num_samples), 90, 200)
    
    bp_dia_mean = 80 + dr_severity * 3
    bp_dia = np.clip(np.random.normal(bp_dia_mean, 10, num_samples), 60, 120)
    
    # BMI: Slightly higher for severe DR
    bmi_mean = 26 + dr_severity * 1.5
    bmi = np.clip(np.random.normal(bmi_mean, 4, num_samples), 18, 45)
    
    # Smoking: More common in severe DR
    smoking_prob = 0.2 + dr_severity * 0.05
    smoking = np.random.choice([0, 1, 2], size=num_samples, p=[1-smoking_prob, smoking_prob*0.5, smoking_prob*0.5])
    
    # Insulin treatment: More common in severe DR
    insulin_prob = 0.3 + dr_severity * 0.15
    insulin = np.random.binomial(1, insulin_prob, num_samples)
    
    return {
        'age': age.astype(np.float32),
        'gender': gender.astype(np.int32),
        'diabetes_duration': diabetes_duration.astype(np.float32),
        'hba1c': hba1c.astype(np.float32),
        'bp_sys': bp_sys.astype(np.float32),
        'bp_dia': bp_dia.astype(np.float32),
        'bmi': bmi.astype(np.float32),
        'smoking': smoking.astype(np.int32),
        'insulin': insulin.astype(np.int32),
    }


def add_clinical_data_to_csv(
    input_csv: str,
    output_csv: str,
    image_col: str = "image_path",
    label_col: str = "diagnosis",
    seed: int = 42,
) -> None:
    """Add clinical data columns to existing DR dataset CSV."""
    
    print(f"📖 Reading input CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    
    print(f"Found {len(df)} samples with columns: {df.columns.tolist()}")
    
    # Check if clinical data already exists
    clinical_cols = ['age', 'gender', 'diabetes_duration', 'hba1c', 'bp_sys', 'bp_dia', 'bmi', 'smoking', 'insulin']
    if all(col in df.columns for col in clinical_cols):
        print("⚠️  Clinical data already exists. Skipping...")
        return
    
    # Group by DR severity and generate data
    print("\n🔬 Generating clinical data based on DR severity...")
    all_data = []
    
    for severity in sorted(df[label_col].unique()):
        severity_df = df[df[label_col] == severity]
        num_samples = len(severity_df)
        
        print(f"  - DR Stage {severity}: {num_samples} samples")
        
        # Generate clinical data
        clinical_data = generate_patient_data(num_samples, severity, seed + severity)
        
        # Create temporary DataFrame
        temp_df = severity_df.copy()
        for col, values in clinical_data.items():
            temp_df[col] = values
        
        all_data.append(temp_df)
    
    # Combine all data
    result_df = pd.concat(all_data, ignore_index=True)
    
    # Reorder columns: image_path, diagnosis, then clinical features
    column_order = [image_col, label_col] + clinical_cols
    result_df = result_df[column_order]
    
    # Save
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)
    
    print(f"\n✅ Saved multimodal dataset to: {output_csv}")
    print(f"   Total samples: {len(result_df)}")
    print(f"   Columns: {result_df.columns.tolist()}")
    
    # Show sample
    print("\n📊 Sample data:")
    print(result_df.head(3).to_string())
    
    # Show statistics
    print("\n📈 Clinical data statistics:")
    print(result_df[clinical_cols].describe())


def create_sample_dataset(output_dir: str, num_samples_per_class: int = 100):
    """Create a complete sample dataset for testing."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🏗️  Creating sample multimodal dataset in: {output_dir}")
    
    # Create sample data
    data = []
    for severity in range(5):
        for i in range(num_samples_per_class):
            img_name = f"img_{severity}_{i:04d}.jpg"
            data.append({
                'image_path': img_name,
                'diagnosis': severity,
            })
    
    df = pd.DataFrame(data)
    
    # Shuffle
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Split: 80% train, 10% val, 10% test
    n = len(df)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)
    
    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]
    
    # Save CSVs
    train_csv = output_path / "train.csv"
    val_csv = output_path / "val.csv"
    test_csv = output_path / "test.csv"
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    print(f"\n✅ Created sample CSVs:")
    print(f"   - Train: {train_csv} ({len(train_df)} samples)")
    print(f"   - Val: {val_csv} ({len(val_df)} samples)")
    print(f"   - Test: {test_csv} ({len(test_df)} samples)")
    
    # Add clinical data to each
    print("\n🔬 Adding clinical data...")
    for csv_file in [train_csv, val_csv, test_csv]:
        add_clinical_data_to_csv(
            input_csv=str(csv_file),
            output_csv=str(csv_file),
            seed=42,
        )
    
    print("\n🎉 Sample multimodal dataset created successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate multimodal DR dataset with clinical data")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["add", "create"],
        default="add",
        help="'add' to add clinical data to existing CSV, 'create' to generate sample dataset"
    )
    parser.add_argument(
        "--input",
        type=str,
        help="Input CSV file (for 'add' mode)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output CSV file or directory"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Samples per class (for 'create' mode)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "add":
        if not args.input or not args.output:
            print("❌ Error: --input and --output required for 'add' mode")
            exit(1)
        
        add_clinical_data_to_csv(
            input_csv=args.input,
            output_csv=args.output,
        )
    
    elif args.mode == "create":
        output_dir = args.output or "data/sample_multimodal"
        create_sample_dataset(
            output_dir=output_dir,
            num_samples_per_class=args.samples,
        )
