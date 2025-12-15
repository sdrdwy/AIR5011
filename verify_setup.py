#!/usr/bin/env python
"""
Verification script to ensure the project setup is complete and correct.
"""

import os
import sys
from pathlib import Path

def check_project_structure():
    """Check if all required files and directories exist."""
    project_root = Path("/workspace/something-something-prediction")
    
    required_files = [
        "README.md",
        "requirements.txt", 
        "train.py",
        "evaluate.py",
        "dataset.py",
        "utils.py",
        "run_training.sh",
        "run_evaluation.sh",
        "PROJECT_STRUCTURE.md",
        "data/prepare_dataset.py"
    ]
    
    print("Checking project structure...")
    all_good = True
    
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            print(f"❌ Missing file: {full_path}")
            all_good = False
        else:
            print(f"✅ Found file: {full_path}")
    
    return all_good

def check_file_contents():
    """Check if key files contain expected content."""
    print("\nChecking file contents...")
    
    # Check train.py
    train_path = Path("/workspace/something-something-prediction/train.py")
    if train_path.exists():
        content = train_path.read_text()
        if "Something-Something V2" in content and "frame prediction" in content:
            print("✅ train.py contains expected content")
        else:
            print("❌ train.py missing expected content")
            return False
    else:
        print("❌ train.py does not exist")
        return False
    
    # Check evaluate.py
    eval_path = Path("/workspace/something-something-prediction/evaluate.py")
    if eval_path.exists():
        content = eval_path.read_text()
        if "SSIM" in content and "PSNR" in content and "Something-Something" in content:
            print("✅ evaluate.py contains expected content")
        else:
            print("❌ evaluate.py missing expected content")
            return False
    else:
        print("❌ evaluate.py does not exist")
        return False
    
    # Check dataset.py
    dataset_path = Path("/workspace/something-something-prediction/dataset.py")
    if dataset_path.exists():
        content = dataset_path.read_text()
        if "Something-Something V2" in content and "move_object" in content and "drop_object" in content and "cover_object" in content:
            print("✅ dataset.py contains expected content")
        else:
            print("❌ dataset.py missing expected content")
            return False
    else:
        print("❌ dataset.py does not exist")
        return False
    
    return True

def check_executables():
    """Check if shell scripts are executable."""
    print("\nChecking executable permissions...")
    
    training_script = Path("/workspace/something-something-prediction/run_training.sh")
    eval_script = Path("/workspace/something-something-prediction/run_evaluation.sh")
    
    training_exec = os.access(training_script, os.X_OK)
    eval_exec = os.access(eval_script, os.X_OK)
    
    if training_exec:
        print("✅ run_training.sh is executable")
    else:
        print("❌ run_training.sh is not executable")
    
    if eval_exec:
        print("✅ run_evaluation.sh is executable")
    else:
        print("❌ run_evaluation.sh is not executable")
    
    return training_exec and eval_exec

def main():
    """Main verification function."""
    print("Verifying Something-Something V2 Frame Prediction Project Setup")
    print("=" * 65)
    
    structure_ok = check_project_structure()
    content_ok = check_file_contents()
    exec_ok = check_executables()
    
    print("\n" + "=" * 65)
    if structure_ok and content_ok and exec_ok:
        print("🎉 All checks passed! Project setup is complete and correct.")
        print("\nTo run the project:")
        print("1. cd /workspace/something-something-prediction")
        print("2. pip install -r requirements.txt")
        print("3. bash run_training.sh --dataset /path/to/dataset")
        print("4. bash run_evaluation.sh --model ./output_model --dataset /path/to/dataset")
        return True
    else:
        print("❌ Some checks failed. Please review the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)