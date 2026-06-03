# Digital-Twins_exam-project

This project is a Streamlit application and preprocessing pipeline for Fiber Bragg Grating (FBG) bending experiment data.

The app loads interrogator wavelength recordings together with hand-held experiment metadata, then provides tools to inspect the signals, apply preprocessing, detect peaks, group peaks by pressure steps, build machine-learning features, and train crack-detection models.

## Features

* Load and inspect raw FBG interrogator data.
* Visualize Bragg wavelength signals over time.
* Compare wavelength-channel correlations.
* Convert wavelength shifts into strain signals.
* Apply centered rolling median filtering.
* Detect wavelength peaks using configurable distance and prominence settings.
* Match detected peaks with hand-held pressure and crack-label metadata.
* Generate preprocessing artifacts under `dataset/_computed`.
* Build feature tensors for model training.
* Train a CNN crack classifier.
* Train a physics-informed CNN head using beam/FBG configuration values.
* Run automated tests with `pytest`.

## Repository structure

```text
.
├── dataset/
│   ├── hand-held-data/
│   │   └── data.xlsx
│   ├── interrogator-data/
│   │   └── *-interrogator.txt
│   └── _computed/
│       ├── 04-median-filter/
│       ├── 05-find-peaks/
│       ├── 06-peaks-groups/
│       └── 08-model-checkpoints/
│
├── src/
│   └── digital_twin/
│       ├── config.py
│       ├── data.py
│       ├── features.py
│       ├── models.py
│       ├── paths.py
│       └── preprocessing.py
│
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_features_and_models.py
│   └── test_preprocessing.py
│
├── streamlit_app.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

### Main folders and files

`dataset/` contains the experiment data. The `interrogator-data` folder stores raw FBG interrogator text files, while `hand-held-data/data.xlsx` contains the corresponding manual experiment metadata. The `_computed` folder is used for generated outputs such as filtered signals, detected peaks, peak groups, physics configuration, and model checkpoints.

`src/digital_twin/` contains the reusable Python package used by the app.

* `paths.py` defines the main dataset, computed-output, and model paths.
* `data.py` loads interrogator files, parses experiment identifiers, estimates reference wavelengths, and loads crack-label metadata from the Excel workbook.
* `preprocessing.py` contains signal-processing utilities such as median filtering, strain conversion, active-channel detection, and peak detection.
* `features.py` builds machine-learning datasets from filtered signals and peak groups.
* `models.py` defines and trains the CNN crack classifier and the physics-informed model.
* `config.py` stores and loads physics constants used by the physics-informed model.

`streamlit_app.py` is the main dashboard entry point. It connects the data-loading, preprocessing, feature-building, and model-training modules into an interactive Streamlit interface.

`tests/` contains the automated test suite for the data, preprocessing, feature, and model utilities.

## Requirements

The project requires Python 3.10 or newer.

Python dependencies are listed in `requirements.txt` and include:

* Streamlit
* NumPy
* pandas
* SciPy
* Plotly
* openpyxl
* scikit-learn
* PyTorch
* pytest

## How to run

### 1. Clone the repository

```bash
git clone https://github.com/codes-de-master/Digital-Twins_exam-project.git
cd Digital-Twins_exam-project
```

### 2. Create and activate a virtual environment

On macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Optional, but useful for local development:

```bash
pip install -e .
```

### 4. Start the Streamlit app

```bash
streamlit run streamlit_app.py
```

The app will open in your browser. From there, you can select an interrogator file, inspect raw and filtered signals, detect peaks, generate computed artifacts, and train the crack-detection models.

## Typical workflow

1. Open the Streamlit app.
2. Select an interrogator file from the sidebar.
3. Inspect the raw wavelength signals.
4. Tune the median-filter and peak-detection settings.
5. Save filtered CSV files and detected peak files.
6. Generate peak groups for the dataset.
7. Build the training dataset from computed artifacts.
8. Train the CNN crack classifier.
9. Optionally configure physics constants and train the physics-informed model.

Generated files are written to `dataset/_computed/`.

## Running tests

To run the test suite:

```bash
pytest
```

The test configuration is defined in `pyproject.toml`, and the tests are located in the `tests/` directory.

## Notes

* The app expects the dataset folders to keep their current structure.
* Computed files and trained checkpoints are generated locally inside `dataset/_computed`.
* If the training dataset is empty, first generate the median-filtered files, peak files, and peak-group files from the Streamlit interface.
* Model checkpoints are saved under `dataset/_computed/08-model-checkpoints`.
