- pip install git+https://github.com/MakiResearchTeam/MultiClassLabelBalancing.git

## General steps

1. Perform dataset analysis using HCScanner:
    1. Generate `.npy` file containing information about labelsets (labelsets, alphas).
2. Perform balancing using available algorithms:
    1. Generate new `.npy` file containing information about labelsets (labelsets, alphas), but with new alphas.
3. Do construction of a new dataset:
    1. You can do copying of existing data using new alphas and then perform uniform sampling from the resulting dataset.
    2. Or you can build a generator that would sample pictures containing certain labelsets using a distribution in order to sample them.
