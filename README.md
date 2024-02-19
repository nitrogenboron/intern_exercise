# Readme

# Requirements
```
tensorflow
numpy
matplotlib
monotonic-nn
```
## Code Structure
```bash
-bbo_project--util.py

            --util_test.py
            
            --main_bbo_approximation.py
            
            --main_bbo_discrete_search.py 
```
1. The `util.py` file provides various utility functions for your project. Here is a summary of each function:

- `is_concave(f)`: Checks if the function `f` is concave based on its values.
- `is_convex(f)`: Checks if the function `f` is convex based on its values.
- `is_non_decreasing(f)`: Checks if the function `f` is non-decreasing.
- `is_non_increasing(f)`: Checks if the function `f` is non-increasing.
- `valid_u(u)`: Checks if each element in the array `u` satisfies a specific condition.
- `validate(func, a, b, c, d, u=[10,20,35,40,50,55,65])`: Validates the properties of a given function `func` within specified ranges and values of `u`.
- `example_f1(x, y, u)`: Defines an example function `f1` that depends on `x`, `y`, and `u`.
- `psinc(u)`: Computes a periodic sinc function.
- `example_f2(x, y, u)`: Defines an example function `f2` that depends on `x`, `y`, and `u`.
- `prepare_data(func, a, b, c, d, U, N=1000, file_name='data.csv')`: Prepares data for a given function `func` within specified ranges and values of `u`, and saves it in a CSV file.
- `prepare_example_dataset1()`: Prepares an example dataset using `example_f1`.
- `prepare_example_dataset2()`: Prepares an example dataset using `example_f2`.
- `split_odd_even(p, f)`: Splits the data into odd and even values based on the `u` attribute.
- `load_data(file_name)`: Loads data from a CSV file.
- `stop_con(f, f_l, f_r)`: Determines if the search for maxima should stop based on the function values.
- `back_tracking(x_t, g_t, f_t, eta, bbo_func)`: Performs backtracking to optimize the search for maxima.
- `search_maxima(bbo_func, x=10, eta=10, max_iter=30)`: Searches for maxima of a given function `bbo_func` using backtracking.


2. `util_test.py`: This file contains unit tests for the functions defined in util.py. Running the tests in util_test.py ensures the correctness and accuracy of the utility functions.
3. `main_bbo_approximation.py`: This demo file demonstrates the training of a monotone neural network to approximate a blackbox function. It showcases the usage of the functions defined in util.py to perform function approximation.
4. `main_bbo_discrete_search.py`: This demo file demonstrates the search for the maxima of a blackbox function using discrete search techniques. It utilizes the functions from util.py to optimize the search process and find the maximum value of the function.

## How to use
Download the `bbo_project` directory, then install the `requirements.txt`:

```
pip install -r requirements.txt
```
Demo Results for Neural network with Convexity and Monicity Guarantee:![net](https://github.com/nitrogenboron/intern_exercise/assets/152514931/cfa9ac97-7285-4572-949e-419c81469474)

```
run main_bbo_approximation.py 
```
![training_history_even](https://github.com/nitrogenboron/intern_exercise/assets/152514931/42fd8de2-d286-42e4-9ee8-f41c3f413a3e)
![training_history_odd](https://github.com/nitrogenboron/intern_exercise/assets/152514931/b6ab2274-781e-4561-89d4-6095767d041d)


Demo Results for Searching for discrete black box function's maxima:
```
run main_bbo_discrete_search.py
```
![search_history_u2](https://github.com/nitrogenboron/intern_exercise/assets/152514931/643c60c9-c24b-47fb-98a3-ce2b5b935650)
![search_history_u1](https://github.com/nitrogenboron/intern_exercise/assets/152514931/30fb10dc-3a84-4ac9-81bc-d03b1189a13e)

