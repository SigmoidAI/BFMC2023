# This is the repository for the Bosch Future Mobility Challenge 2022

## In order to prepare the project to be runned

### 1. Download the video files from the follwing [Link](https://drive.google.com/drive/folders/1RQziPcPgrpaDIH4r55o07M_YF2KlxJWh?usp=sharing) and place them in the folder "files"

### 2. Project uses Python 3.8, preferably create a virtual environement for your workspace:

#### For Windows
Before running this command, make sure you have python 3.8 installed and added to your PATH variable and it is your main python version.

```cmd
python -m venv venv
venv\Scripts\activate.bat 
```

#### For Ubuntu
```python
python3.8 -m venv venv
source venv/bin/activate
```

### 3. Install the requirements

```python
pip install -r requirements.txt
```

### 4. Download the [Pyzed sdk](https://www.stereolabs.com/developers/release/) 

### 5. Install the Pyzed sdk (idk if it will work)

```cmd
python scripts\get_python_api.py
```

### 6. Open Port 
```cmd
sudo chmod 777 /dev/ttyACM0
```

### 7. Run the project

```python
python scripts/main.py
```
