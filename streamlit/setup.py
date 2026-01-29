#!/usr/bin/env python3
"""
Setup script for US Accident Severity Prediction System
Run this script to check dependencies and set up the application
"""

import sys
import os
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60 + "\n")

def check_python_version():
    """Check if Python version is 3.8 or higher"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ ERROR: Python 3.8 or higher is required")
        return False
    
    print("✅ Python version is compatible")
    return True

def install_dependencies():
    """Install required packages from requirements.txt"""
    print_header("Installing Dependencies")
    
    requirements_file = Path("requirements.txt")
    
    if not requirements_file.exists():
        print("❌ ERROR: requirements.txt not found")
        return False
    
    print("Installing packages from requirements.txt...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ All dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Failed to install dependencies: {e}")
        return False

def check_model_files():
    """Check if required model files exist"""
    print_header("Checking Model Files")
    
    files_to_check = [
        ("model.pkl", "Trained XGBoost model"),
        ("preprocessing/new/preprocessor.pkl", "Fitted preprocessor")
    ]
    
    all_exist = True
    for filepath, description in files_to_check:
        if Path(filepath).exists():
            print(f"✅ Found: {description} ({filepath})")
        else:
            print(f"❌ Missing: {description} ({filepath})")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  WARNING: Some model files are missing.")
        print("   The application will not work without these files.")
        print("   Please ensure you have trained the model and saved the files.")
    
    return all_exist

def setup_config():
    """Create config.py from template if it doesn't exist"""
    print_header("Setting Up Configuration")
    
    config_file = Path("config.py")
    template_file = Path("config_template.py")
    
    if config_file.exists():
        print("✅ config.py already exists")
        return True
    
    if not template_file.exists():
        print("❌ ERROR: config_template.py not found")
        return False
    
    try:
        with open(template_file, 'r') as src:
            content = src.read()
        
        with open(config_file, 'w') as dst:
            dst.write(content)
        
        print("✅ Created config.py from template")
        print("\n⚠️  IMPORTANT: Edit config.py and add your NOAA API token")
        print("   Get your free token from: https://www.ncdc.noaa.gov/cdo-web/token")
        return True
    except Exception as e:
        print(f"❌ ERROR: Failed to create config.py: {e}")
        return False

def create_directories():
    """Create necessary directories if they don't exist"""
    print_header("Creating Directories")
    
    directories = [
        "preprocessing/new",
        "data/processed",
        "data/new/training_ready"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Directory ready: {directory}")
    
    return True

def run_tests():
    """Run basic import tests"""
    print_header("Running Import Tests")
    
    packages_to_test = [
        "streamlit",
        "pandas",
        "numpy",
        "joblib",
        "sklearn",
        "xgboost",
        "requests",
        "folium",
        "plotly"
    ]
    
    all_success = True
    for package in packages_to_test:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError as e:
            print(f"❌ {package}: {e}")
            all_success = False
    
    return all_success

def print_next_steps():
    """Print instructions for next steps"""
    print_header("Next Steps")
    
    print("""
1. ✅ Edit config.py and add your NOAA API token
   Get free token: https://www.ncdc.noaa.gov/cdo-web/token

2. ✅ Ensure model files are in place:
   - model.pkl (root directory)
   - preprocessor.pkl (preprocessing/new/ directory)

3. ✅ Run the application:
   streamlit run app.py

4. ✅ Open your browser to the URL shown by Streamlit
   (Usually http://localhost:8501)

For more information, read README.md
    """)

def main():
    """Main setup function"""
    print_header("US Accident Severity Prediction System - Setup")
    
    # Run all checks
    checks = [
        ("Python Version", check_python_version),
        ("Dependencies", install_dependencies),
        ("Directories", create_directories),
        ("Configuration", setup_config),
        ("Model Files", check_model_files),
        ("Import Tests", run_tests)
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"❌ ERROR during {name}: {e}")
            results[name] = False
    
    # Summary
    print_header("Setup Summary")
    
    all_passed = True
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n🎉 Setup completed successfully!")
        print_next_steps()
        return 0
    else:
        print("\n⚠️  Setup completed with warnings or errors")
        print("   Please address the issues above before running the application")
        return 1

if __name__ == "__main__":
    sys.exit(main())
