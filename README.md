# Island-Curve-Compression-CAE
1D-CAE compression framework for vector curves

## 1. Project Overview
Our framework introduces a multi-stage approach for vector data management:
- **Preprocessing:** Standardizing sequence length and spatial density.
- **Compression:** Using 1D-CAE to extract compact latent representations.
- **Evaluation:** Multi-metric fidelity assessment ($PD, RAE, RPE$).
- **Restoration:** Converting optimized representations back to GIS-compatible formats.

## 2. Directory Structure
- `data/`: Contains sample Shapefiles of island boundaries. Note: Each Shapefile layer includes multiple files (.shp, .dbf, .prj, .shx) which must be kept together.
- `scripts/`:
    - `Data_preprocessing/`: Scripts for coordinate resampling and sequence segmentation.
    - `Model_train_and_estimate/`: Core training scripts and comprehensive evaluation metrics.
    - `Data_restore/`: Utility to convert processed text outputs back to Shapefile format.
- `Network_Parameter.xlsx`: Detailed layer-wise configuration of the proposed CAE model.
- `CR_Comparison.xlsx`: Experimental logs documenting compression ratios (CR) across different models and thresholds.

## 3. Installation
```bash
pip install -r requirements.txt

## 4. Workflow
Preprocessing: Run Resampling.py to normalize the input curves.

Training: Execute train_gpu.py to train the autoencoder.

Evaluation: Use estimate.py and estimate2.py to verify geometric fidelity.

Restoration: Run txt2shp.py to export the reconstructed results for GIS visualization.

## 5. Experimental Results
The detailed comparison of compression performance and network hyperparameters can be found in the provided Excel files (CR_Comparison.xlsx and Network_Parameter.xlsx). These records validate the efficiency of 1D-CAE against traditional baselines.
