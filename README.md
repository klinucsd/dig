# Setup Guide

Follow these steps to set up the project:

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/klinucsd/dig
```

### 2. Download and Setup Vector Databases

#### Data Commons Vector DB
Download and unzip the Data Commons vector database:
```bash
# Download from: https://hubbub.sdsc.edu/.test/data_commons.zip
# Extract the downloaded file to your project directory
```

#### WENOKN Vector DB
Download and extract the WENOKN vector database:
```bash
# Download from: https://hubbub.sdsc.edu/.test/wenokn.zip
# Extract the downloaded file to your project directory
```

### 3. Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
Edit the `.env` file to configure your environment settings.

### 5. Test Installation
Run the test to verify everything is working correctly:
```bash
python smart_query/test/data_system_test.py
```

## Notes
- Make sure you have Python and pip installed on your system
- Ensure you have sufficient storage space for the vector databases
- Check that all file paths are correctly configured in your `.env` file
