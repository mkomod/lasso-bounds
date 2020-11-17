# Modern Statistical Theory

Tool for studying locations of optimal lambda values in LASSO - built to verify and further explore results introduced
in _**Hebiri, M. and Lederer, J., 2012. How correlations influence lasso prediction. IEEE Transactions on Information Theory, 59(3), pp.1846-1854**_.

## Usage

```bash
$ pip install -r requirements.txt
$ python src/main.py num_experiments [parameter_to_vary [space_seperated_list_of_param_vals]]
```

```
parameters_to_vary:
    - ns: the number of observations
    - ps: the length of the parameter vector
    - ss: the sparsity of the parameter vector
    - sigs: the standard deviation of the noise term
    - rhos: the correlation use to generate the design matrix
    - etas: the noise used in constructing the extended design matrix
```

Output generated is saved to `./figs` and the experiment data is saved in `./results/`

### Example

```
$ python src/main.py 1000 sigs 1 0.1 1.5 2
```

