#!/usr/bin/env python3
"""
Data Manager for MIE Classification System
Handles cloud storage integration and data loading
"""

import os
import gdown
import pandas as pd
from pathlib import Path
import zipfile

class DataManager:
    def __init__(self, data_dir="data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Google Drive file IDs (you'll get these after uploading)
        self.drive_files = {
            "final_data_true.csv": "YOUR_GOOGLE_DRIVE_FILE_ID_HERE",
            "mie_articles_only.csv": "YOUR_GOOGLE_DRIVE_FILE_ID_HERE", 
            "non_mie_248.csv": "YOUR_GOOGLE_DRIVE_FILE_ID_HERE",
            "MIE_Coding_Instructions.txt": "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
        }
        
        # Large year files (optional - can be downloaded on demand)
        self.year_files = {
            "2015.csv": "YOUR_GOOGLE_DRIVE_FILE_ID_HERE",
            "2016.csv": "YOUR_GOOGLE_DRIVE_FILE_ID_HERE",
            "2017.csv": "YOUR_GOOGLE_DRIVE_FILE_ID_HERE",
            "2018.csv": "YOUR_GOOGLE_DRIVE_FILE_ID_HERE",
            "2019.csv": "YOUR_GOOGLE_DRIVE_FILE_ID_HERE",
            "2020.csv": "YOUR_GOOGLE_DRIVE_FILE_ID_HERE",
            "2021.csv": "YOUR_GOOGLE_DRIVE_FILE_ID_HERE",
            "2022.csv": "YOUR_GOOGLE_DRIVE_FILE_ID_HERE",
            "2023.csv": "YOUR_GOOGLE_DRIVE_FILE_ID_HERE",
            "NYT_articles_2015 (1).csv": "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"
        }
    
    def download_from_drive(self, filename, file_id=None):
        """Download a file from Google Drive"""
        if file_id is None:
            file_id = self.drive_files.get(filename)
            
        if not file_id:
            print(f"❌ No file ID found for {filename}")
            return False
            
        filepath = self.data_dir / filename
        
        try:
            print(f"📥 Downloading {filename} from Google Drive...")
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                str(filepath),
                quiet=False
            )
            print(f"✅ Downloaded {filename}")
            return True
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            return False
    
    def load_training_data(self, force_download=False):
        """Load the main training dataset"""
        filepath = self.data_dir / "final_data_true.csv"
        
        # Check if file exists locally
        if not filepath.exists() or force_download:
            print("📊 Training data not found locally. Downloading from cloud...")
            success = self.download_from_drive("final_data_true.csv")
            if not success:
                raise FileNotFoundError("Could not download training data")
        
        # Load the data
        print("📖 Loading training data...")
        df = pd.read_csv(filepath)
        print(f"✅ Loaded {len(df)} articles")
        return df
    
    def download_all_data(self):
        """Download all essential data files"""
        print("🚀 Downloading all MIE classification data...")
        
        essential_files = [
            "final_data_true.csv",
            "mie_articles_only.csv", 
            "non_mie_248.csv",
            "MIE_Coding_Instructions.txt"
        ]
        
        for filename in essential_files:
            self.download_from_drive(filename)
    
    def download_year_data(self, year):
        """Download specific year data on demand"""
        filename = f"{year}.csv"
        return self.download_from_drive(filename, self.year_files.get(filename))
    
    def cleanup_local_data(self):
        """Remove local data to save space"""
        print("🧹 Cleaning up local data files...")
        
        for file in self.data_dir.glob("*.csv"):
            if file.name in ["final_data_true.csv", "mie_articles_only.csv", "non_mie_248.csv"]:
                continue  # Keep essential files
            file.unlink()
            print(f"🗑️  Removed {file.name}")
    
    def get_data_info(self):
        """Get information about available data"""
        print("📊 Data Information:")
        print(f"Local data directory: {self.data_dir}")
        
        local_files = list(self.data_dir.glob("*.csv"))
        print(f"Local CSV files: {len(local_files)}")
        
        for file in local_files:
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"  - {file.name}: {size_mb:.1f} MB")

def main():
    """Setup and test data manager"""
    dm = DataManager()
    
    print("🔧 MIE Data Manager Setup")
    print("=" * 40)
    
    # Show current data info
    dm.get_data_info()
    
    print("\n📋 Next Steps:")
    print("1. Upload your CSV files to Google Drive")
    print("2. Get the file IDs from the shareable links")
    print("3. Update the file IDs in data_manager.py")
    print("4. Run: python data_manager.py --download-all")
    
    # Test loading training data
    try:
        df = dm.load_training_data()
        print(f"\n✅ Successfully loaded {len(df)} training articles")
    except FileNotFoundError:
        print("\n⚠️  Training data not found. Please upload to Google Drive first.")

if __name__ == "__main__":
    main() 