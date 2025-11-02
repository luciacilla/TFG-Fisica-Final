# Reference Implementations

This folder contains external reference code used for comparison and validation
of the results in this TFG.

## Files

- **chain_alt_bp_modified.py**
  Adapted reference implementation based on the original PythTB examples  
  `chain_alt.py` and `chain_alt_bp.py` by D. Vanderbilt (*Berry Phases in Electronic Structure Theory*).  
  This version preserves the logic of the original scripts but:
  - wraps the code into functions,
  - allows explicit control of the internal orbital positions (site- or midbond-centered),
  - prints both the explicit discrete Berry phase and the `wf_array` result  
    for direct comparison with the implementation in `src/model.py`.

  It is intended purely for cross-checking and educational purposes.

- **pythtb.py**
  Copy of the `PythTB` library, a minimal tight-binding Python toolkit
  developed by D. Vanderbilt’s group.
  Included here only for reproducibility; the main project does not depend on it.

## Notes

The files in this folder are **not imported** by the main code in `src/`.  
They can be executed independently to reproduce the reference results from Vanderbilt's methodology and to verify the consistency of the custom implementation developed for this project.
