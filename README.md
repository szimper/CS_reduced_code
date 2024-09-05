# On reduced inertial PDE models for Cucker-Smale flocking dynamics
This repository contains python code for the publication _On reduced inertial PDE models for Cucker-Smale flocking dynamics_. DOI: 
https://doi.org/10.48550/arXiv.2407.18717 by Sebastian Zimper, Federico Cornalba, Nataša Djurdjevac Conrad and Ana Djurdjevac. 

## Cucker-Smale model and reduced inertial PDE
The python code in this repository can be used to simulate the Cucker-Smale model as well as the associated reduced intertial PDE in one and two dimensions. It also includes the necessary code for processing the simulated data. In particular, code for:
*simulations of the one and two dimensional CS model, `CS_1D.py` and `CS_2D.py`, respectively.
*simulations of the reduced inertial PDE model and hydrodynamic model in one dimension `PDEs_IH_1D.py`.
*the reduced inertial PDE model in two dimensions `Inertial_PDE_2D.py`.
*processing the data generated `Data_processing.py`.

The code in this repository is for single realisations of the system, but it is straighforward to parralise it for different inital conditions/parameter settings.

## Citation
If you use these files in your academic projects, we politely ask you to aknowledge it in your manuscript by the following BibTex citation:

*@misc{zimper2024reducedinertialpdemodels,
      title={On reduced inertial PDE models for Cucker-Smale flocking dynamics}, 
      author={Sebastian Zimper and Federico Cornalba and Nataša Djurdjevac Conrad and Ana Djurdjevac},
      year={2024},
      eprint={2407.18717},
      archivePrefix={arXiv},
      primaryClass={math.AP},
      url={https://arxiv.org/abs/2407.18717}, 
}*