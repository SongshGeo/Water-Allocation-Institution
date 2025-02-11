# Water-Allocation-Institution

![Python](https://badgen.net/badge/Python/<3.8/green)
![GitHub](https://badgen.net/badge/GitHub/SongshGeo/:color?icon=github)
![Star](https://badgen.net/github/stars/SongshGeo/Water-Allocation-Institution)

This repository is the code for the paper: [Institutional shifts and effects to the Yellow River Basins' water uses](https://doi.org/10.1016/j.jhydrol.2024.130638).

## Methods

>[!Warning]
> In the parameter settings, the commented out units (with `#`) are NOT included in the analysis.

- Firstly, I used PCA to reduce the dimensionality. You can check the [PCA.ipynb](notebooks/02_PCA.ipynb) for more details and [pca.yaml](config/vars/pca.yaml) for the parameters.
- Then, I used the [Synthetic Control Methods](https://github.com/SongshGeo/SyntheticControlMethods) package to get the weights of the control units. You can check the [03_SynthControl.ipynb](notebooks/03_SynthControl.ipynb) for the processes and results. The parameters are in the [synth.yaml](config/synth.yaml).
