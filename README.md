README: Fluke Measurement and Analysis Tool

This project provides Python scripts designed to automate whale fluke measurements from drone imagery using Meta AI's Segment Anything Model (SAM). The scripts segment flukes, overlay measurement grids following the methodology of Bose & Lien (1989), and calculate key hydrodynamic and geometric metrics.


DIRECTORY STRUCTURE
-------------------

Fluke_Measurements/
├── UAV21_196e/
│   ├── corrected_frames/
│   ├── UAV21_196e_summary.csv
│   └── Bose_Lien_Fluke_Dimensions/
│       ├── cropped_fluke.jpg
│       ├── fluke_dimensions.csv
│       └── fluke_overlay.png
├── UAV22_201d/
│   ├── corrected_frames/
│   ├── UAV22_201d_summary.csv
│   └── Bose_Lien_Fluke_Dimensions/
│       └── ...
└── ...


KEY FEATURES
------------

1. **Interactive Fluke Cropping and Rotation:**
   - Interactive cropping of the fluke from drone images.
   - Manual rotation adjustment with an interactive slider.

2. **Automated SAM Segmentation:**
   - Uses Meta AI's Segment Anything Model (SAM) to extract precise fluke masks.

3. **Interactive Adjustment Sliders:**
   - Adjust symmetry axis, root chord (vertical baseline), horizontal pitching axis, datum, and cutoff line with visual feedback.

4. **Measurement Grid Overlay:**
   - Fluke is segmented into 11 sections following Bose & Lien (1989).
   - Section lines are drawn and mirrored accurately across the symmetry axis.

5. **Pitching Axis Computation:**
   - Calculates pitching axis positions based on chord lengths and b/chord ratios from Bose & Lien (1989).

6. **Detailed CSV Output:**
   - Outputs measurements in pixel units to ensure flexibility for downstream analysis.
   - Outputs include leading/trailing edges, chord lengths, distances from datum, and pitch axis positions.

7. **Visual Confirmation:**
   - Overlays and saves visual confirmations for manual verification.

8. **Automated Batch Processing:**
   - Scripts can iterate through multiple deployment folders, automating the process for large datasets.


DEPENDENCIES
------------

Python libraries required:

- OpenCV (`cv2`)
- NumPy (`numpy`)
- Pandas (`pandas`)
- Matplotlib (`matplotlib`)
- Segment Anything Model (SAM) by Meta AI
  - Installation and setup instructions: https://github.com/facebookresearch/segment-anything

Install libraries using pip:

```bash
pip install numpy opencv-python pandas matplotlib segment-anything torch torchvision
```


FILE DESCRIPTIONS
-----------------

- `fluke_11section.py`: Main module containing functions for fluke segmentation, measurement, and visualization.
- `run_fluke_extraction.py`: Wrapper script to iterate through UAV deployment folders and automate analysis.
- `__init__.py`: Marks directories as Python packages (typically empty).


HOW TO RUN
----------

1. Place drone deployment folders (containing `_summary.csv` and `corrected_frames` folders) inside your main data directory (`Fluke_Measurements`).

2. Modify paths within `run_fluke_extraction.py`:

```python
base_dir = "/your/path/to/Fluke_Measurements"
```

3. Run the wrapper script to process all folders automatically:

```bash
python run_fluke_extraction.py
```

Results are saved automatically to each deployment's `Bose_Lien_Fluke_Dimensions` folder.


KNOWN ISSUES & NOTES
--------------------

- The segmentation quality highly depends on accurate initial clicks during SAM initialization.
- Ensure proper installation and verification of SAM dependencies.
- Manual verification of overlay plots is recommended to ensure accuracy.


REFERENCES
----------

- Bose, N., & Lien, J. (1989). Propulsion of a fin whale (Balaenoptera physalus): why the fin whale is a fast swimmer. Proceedings of the Royal Society of London. Series B, Biological Sciences, 237(1287), 175–200.
- Segment Anything Model (SAM): https://github.com/facebookresearch/segment-anything