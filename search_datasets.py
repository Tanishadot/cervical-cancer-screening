"""
Script to search for available cervical cancer datasets on Kaggle.
"""

import subprocess
import sys
import os

def find_kaggle_path():
    """Find the working Kaggle executable path."""
    kaggle_path = r"C:\Users\User\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\kaggle.exe"
    if os.path.exists(kaggle_path):
        return kaggle_path
    return "kaggle"  # fallback

def search_cervical_datasets():
    """Search for cervical cancer datasets on Kaggle."""
    kaggle_cmd = find_kaggle_path()
    
    print("Searching for cervical cancer datasets on Kaggle...")
    print("="*60)
    
    # Search terms
    search_terms = [
        "cervical cancer",
        "cervical",
        "pap smear",
        "sipakmed",
        "herlev",
        "cervix"
    ]
    
    for term in search_terms:
        print(f"\nSearching for: '{term}'")
        print("-" * 40)
        
        try:
            result = subprocess.run(
                [kaggle_cmd, "datasets", "list", "-s", term],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) > 1:  # Header + results
                    for line in lines[1:6]:  # Show first 5 results
                        if line.strip():
                            print(f"  {line}")
                else:
                    print("  No results found")
            else:
                print(f"  Error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            print("  Search timed out")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "="*60)
    print("Try downloading with:")
    print("python setup_dataset.py --dataset <dataset_name>")
    print("\nOr manually with:")
    print(f"{kaggle_cmd} datasets download -d <dataset_name>")

def test_specific_datasets():
    """Test specific dataset names."""
    kaggle_cmd = find_kaggle_path()
    
    print("\nTesting specific dataset names...")
    print("="*60)
    
    test_datasets = [
        "andrewmvd/cervical-cancer-dataset",
        "birdy654/cervical-cancer-dataset", 
        "aryashah2k/cervical-cancer-classification-dataset",
        "tanyashah/cervical-cancer",
        "rsufa/cervical-cancer",
        "mssnaufeel/cervical-cancer-dataset",
        "rabieelt/cervical-cancer-classification"
    ]
    
    for dataset in test_datasets:
        print(f"\nTesting: {dataset}")
        try:
            result = subprocess.run(
                [kaggle_cmd, "datasets", "metadata", "-d", dataset],
                capture_output=True,
                text=True,
                timeout=15
            )
            
            if result.returncode == 0:
                print(f"  ✅ Available")
                # Extract some info
                lines = result.stdout.split('\n')
                for line in lines:
                    if 'title:' in line.lower() or 'size:' in line.lower():
                        print(f"     {line.strip()}")
            else:
                print(f"  ❌ Not available")
                
        except subprocess.TimeoutExpired:
            print("  ❌ Timeout")
        except Exception as e:
            print(f"  ❌ Error: {e}")

def main():
    """Main function."""
    print("KAGGLE CERVICAL CANCER DATASET SEARCH")
    print("="*60)
    
    # First test Kaggle connection
    kaggle_cmd = find_kaggle_path()
    try:
        result = subprocess.run([kaggle_cmd, "datasets", "list"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ Kaggle connection working")
        else:
            print("❌ Kaggle connection failed")
            print(f"Error: {result.stderr}")
            return
    except Exception as e:
        print(f"❌ Kaggle test failed: {e}")
        return
    
    # Search for datasets
    search_cervical_datasets()
    
    # Test specific datasets
    test_specific_datasets()
    
    print("\n" + "="*60)
    print("SEARCH COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
