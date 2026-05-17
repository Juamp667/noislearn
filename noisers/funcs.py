'''
    Module with tools to create noisy data.
'''

import numpy as np

def urlf(y, noise_level=0.1, random_state=42):
    """
    Uniform Randomized Label Flip noise.

    Randomly changes a given percentage of labels to another class, using a
    uniform random selection process.

    This function implements class-independent label noise. First, it randomly
    selects a number of instances according to the global noise level. Then, for
    each selected instance, it assigns a new label chosen uniformly from the set
    of available classes. If the new label is equal to the original one, it is
    shifted to a different class to ensure that the selected instance is actually
    modified.

    Parameters
    ----------
    y : array-like
        Original labels. It can be a list, tuple, NumPy array, or any array-like
        structure containing the class labels.

    noise_level : float, default=0.1
        Global proportion of labels to modify. It must usually be a value between
        0 and 1. For example, noise_level=0.1 means that approximately 10% of the
        labels will be selected for modification.

    random_state : int, default=42
        Seed used to initialize the random number generator. It allows the noise
        generation process to be reproducible.

    Returns
    -------
    y_out : numpy.ndarray
        Copy of the original labels with uniform randomized label flip noise
        applied.

    Notes
    -----
    This function applies a global noise level, so every instance has the same
    probability of being selected independently of its original class. Therefore,
    this corresponds to class-independent noise, not class-dependent noise.

    The number of modified labels is computed as:

        k = int(noise_level * n)

    where n is the total number of instances.

    Be aware that indices are selected with replacement because rng.integers is
    used. Therefore, the final number of different modified positions can be lower
    than k if the same index is selected more than once.

    Examples
    --------
    >>> y = np.array([0, 0, 1, 1, 2, 2])
    >>> y_noisy = urlf(y, noise_level=0.3, random_state=42)
    >>> y_noisy
    array([...])
    """
    # Save an array copy of the labels 
    y = np.asarray(y)
    y_out = y.copy()

    # Initilize random generator
    rng = np.random.default_rng(random_state)

    # Compute the set of (different) labels and the indices associated to each one
    classes, y_idx = np.unique(y_out, return_inverse=True)
    # Compute the number of instances 
    n = len(y_out)
    # Compute the desired number of noisy instances
    k = int(noise_level * n)
    if k <= 0:
        print("No noise has been added, since the noise_level is too low.")
        return y_out
    
    # Randomly select the (instance) indices to shift 
    idx_to_change = rng.integers(low=0, high=n, size=k)

    # Randomly select the labels to associate latter indices
    new_idx = rng.integers(low=0, high=len(classes), size=len(idx_to_change))

    # Reasure every idx_to_change is actually shifted
    same = new_idx == y_idx[idx_to_change]
    if np.any(same) and len(classes) > 1:
        new_idx[same] = (new_idx[same] + 1) % len(classes)

    # Shift and return idx_to_change
    y_out[idx_to_change] = classes[new_idx]
    return y_out


def nar(y, noise_levels=None, random_state=42, random_range=(0.0, 0.2)):
    """
    Not At Random label noise.

    Applies class-dependent label noise. Each class has its own probability of
    being shifted to a different randomly selected class.

    If noise_levels is provided, the function uses the probability assigned to
    each class. If noise_levels is None, each class receives a random noise
    probability sampled uniformly from random_range.

    Parameters
    ----------
    y : array-like
        Original labels. It can be a list, tuple, NumPy array, or any array-like
        structure containing the class labels.

    noise_levels : dict, optional
        Dictionary assigning one noise probability to each class.

        The keys must be exactly the classes present in y, and the values must be
        probabilities between 0 and 1.

        Example:
            {0: 0.05, 1: 0.20, 2: 0.10}

        If None, the probabilities are generated randomly using random_range.

    random_state : int, default=42
        Seed used to initialize the random number generator. It makes the noise
        generation process reproducible.

    random_range : tuple(float, float), default=(0.0, 0.2)
        Range used to randomly generate the class noise probabilities when
        noise_levels is None.

        It must satisfy:

            0 <= low <= high <= 1

    Returns
    -------
    y_out : numpy.ndarray
        Copy of the original labels with Not At Random noise applied.

    Notes
    -----
    This function does not apply the same noise probability to all instances.
    Instead, the probability of changing a label depends on its original class.
    Recall however that no bias is preseted when for a class to be shifted to one of all the available ones.

    For each class c, the function:
        1. Finds all instances whose label is c.
        2. Selects each of those instances with probability noise_levels[c].
        3. Replaces each selected label with a random label different from c.

    Therefore, this is class-dependent noise, also called Not At Random noise.

    Unlike the URLF function, this function does not force an exact global number
    of noisy instances. The final number of modified labels depends on the random
    sampling process inside each class.

    Examples
    --------
    >>> y = np.array([0, 0, 0, 1, 1, 2, 2, 2])
    >>> y_noisy = nar(
    ...     y,
    ...     noise_levels={0: 0.05, 1: 0.30, 2: 0.10},
    ...     random_state=42
    ... )

    >>> y_noisy = nar(
    ...     y,
    ...     noise_levels=None,
    ...     random_range=(0.05, 0.25),
    ...     random_state=42
    ... )
    """
    # Save an array copy of the labels 
    y = np.asarray(y)
    y_out = y.copy()

    # Initilize random generator
    rng = np.random.default_rng(random_state)

    # Compute the set of (different) labels and the indices associated to each one
    classes, y_idx = np.unique(y_out, return_inverse=True)

    if len(classes) <= 1:
        print("No noise has been added, since there are just one or no classes to be shifted.")
        return y_out

    # Check or randomly preset the noise level associated to each class
    if noise_levels is None:
        low, high = random_range

        if low < 0 or high > 1 or low > high:
            raise ValueError("random_range must satisfy 0 <= low <= high <= 1.")
        # Randomly asign each class a prob to be shifted (inside the range) 
        noise_levels = {c: rng.uniform(low, high) for c in classes}
        
    # If given, check that noise_levels has the correct format
    else:
        if len(noise_levels) != len(classes):
            raise ValueError(
                "noise_levels must contain exactly one probability per class."
            )

        if set(noise_levels.keys()) != set(classes):
            raise ValueError(
                "noise_levels keys must match the classes in y."
            )

        for c in noise_levels:
            if noise_levels[c] < 0 or noise_levels[c] > 1:
                raise ValueError(
                    f"Noise probability for class {c} must be between 0 and 1."
                )

    # Shift each class according to its own probability
    for c in classes:
        # Extract the indices for instances belonging to class `c`
        class_indices = np.where(y_out == c)[0]
        # Extract the prob. to randomly change  shift an instance (from class `c`)
        prob = noise_levels[c]
        # Randomly select the indices (from class `c`) to shift
        idx_to_change = class_indices[rng.random(len(class_indices)) < prob]

        if len(idx_to_change) == 0:
            continue

        # Extract the labels differente from `c`
        possible_classes = classes[classes != c]

        # Select the new labels to assign to each idx_to_change
        new_labels = rng.choice(
            possible_classes,
            size=len(idx_to_change),
            replace=True
        )

        # Shift and return idx_to_change
        y_out[idx_to_change] = new_labels

    return y_out


    