import numpy as np
from icecream import ic
import numpy as np
import random
from typing import Optional

from tqdm import tqdm


MATH_CONSTANTS = ['pi', 'e']
UNARY_OPS = ['sin', 'cos', 'exp', 'log']
BINARY_OPS = ['+', '-', '*', '/']

class Node:
    """ Represents a node in an expression tree for symbolic regression. """
    def __init__(self, type: str, value: str, parent: Optional['Node'] = None,
                 left: Optional['Node'] = None, right: Optional['Node'] = None):
        self.type = type  # Type of node: 'var', 'const', 'math_const', 'unary_op', 'binary_op'
        self.value = value  # The actual value (e.g., 'x0', '3.14', '+', 'sin')
        self.parent = parent
        self.left = left
        self.right = right
    
    def copy(self) -> 'Node':
        """ Creates a deep copy of the tree, preserving structure but not parent references. """
        new_node = Node(self.type, self.value)
        if self.left:
            new_node.left = self.left.copy()
            new_node.left.parent = new_node
        if self.right:
            new_node.right = self.right.copy()
            new_node.right.parent = new_node
        return new_node

    def __str__(self):
        """ Converts the tree into a human-readable mathematical expression. """
        if self.type == 'const' or self.type == 'math_const':
            return self.value
        elif self.type == 'var':
            return f"x[{self.value[1:]}]"
        elif self.type == 'unary_op':
            return f"np.{self.value}({self.left})"
        elif self.type == 'binary_op':
            return f"({self.left} {self.value} {self.right})"

def generate_random_tree(num_vars: int, mode: str = 'full', max_depth: int = 5, max_const: int = 10, depth: int = 0) -> Node:
    """
    Recursively builds a random expression tree.

    Parameters:
    - num_vars (int): Number of available input variables.
    - mode (str): 'full' forces operators at all depths, 'grow' allows early stopping.
    - max_depth (int): The maximum depth of the tree.
    - max_const (int): The maximum value for randomly generated constants.
    - depth (int): The current depth of the recursion.

    Returns:
    - Node: The root of the generated tree.
    """
    # Base case: If maximum depth is reached, return a  leaf - terminal node (constant or variable)
    if depth == max_depth:
        node_type = random.choices(['math_const', 'const', 'var'], weights=[0.1, 0.4, 0.5])[0]
        if node_type == 'const':
            value = str(random.randint(0, max_const))  # Random integer constant
        elif node_type == 'math_const':
            value = random.choice(MATH_CONSTANTS)  # Choose 'pi' or 'e'
        else:
            value = 'x' + str(random.randint(0, num_vars - 1))  # Choose a variable 'x0', 'x1', etc.
        return Node(node_type, value)

    # Mode selection: Determines how trees grow
    if mode == 'full':
        node_type = random.choice(['unary_op', 'binary_op'])  # Ensure tree fills completely
    elif mode == 'grow':
        node_type = random.choices(['const', 'math_const', 'var', 'unary_op', 'binary_op'],
                                   weights=[0.05, 0.1, 0.2, 0.325, 0.325])[0]  # Allows early stopping

    # Generate the appropriate node type
    if node_type == 'const':
        return Node('const', str(random.randint(0, max_const)))
    elif node_type == 'math_const':
        return Node('math_const', random.choice(MATH_CONSTANTS))
    elif node_type == 'var':
        return Node('var', 'x' + str(random.randint(0, num_vars)))
    elif node_type == 'unary_op':
        # Unary operators (e.g., sin, cos) only have a left child
        left_child = generate_random_tree(num_vars, mode, max_depth, max_const, depth + 1)
        return Node('unary_op', random.choice(UNARY_OPS), left=left_child)
    elif node_type == 'binary_op':
        # Binary operators (e.g., +, -, *, /) have both left and right children
        left_child = generate_random_tree(num_vars, mode, max_depth, max_const, depth + 1)
        right_child = generate_random_tree(num_vars, mode, max_depth, max_const, depth + 1)
        return Node('binary_op', random.choice(BINARY_OPS), left=left_child, right=right_child)

# Function to check if a tree is valid
def is_valid_tree(node: Node) -> bool:
    """
    Recursively checks if a tree contains invalid operations.
    Returns True if the tree is valid, False if it contains risky functions.
    """
    if node is None:
        return False  # Invalid if node is empty
    
    if node.type in ['const', 'math_const', 'var']:
        return True  # Constants and variables are always valid
    
    if node.type == 'unary_op':
        # We could reject functions that can cause issues (e.g., log(0), sqrt(-1))
        # However, this seems too drastic for now, so we allow it.
        return is_valid_tree(node.left)
    
    if node.type == 'binary_op':
        # Avoid division by zero by rejecting trees where right child is exactly 0
        if node.value == '/' and node.right and node.right.value == '0':
            return False
        return is_valid_tree(node.left) and is_valid_tree(node.right)
    
    return False  # Reject unknown node types

# Function to evaluate a tree safely
import warnings

def evaluate_tree(node: Node, x: np.ndarray) -> np.ndarray:
    """
    Evaluates a tree safely. If any computation results in NaN/Inf, returns np.inf.
    """
    try:
        if not is_valid_tree(node):
            return np.full(x.shape[1], np.inf)  # Penalize invalid trees immediately
        
        if node.type == 'const':
            return np.full(x.shape[1], float(node.value))  # Broadcast constant value across all samples
        elif node.type == 'math_const':
            return np.full(x.shape[1], np.pi if node.value == 'pi' else np.e)
        elif node.type == 'var':
            return x[int(node.value[1:]), :]  # Select the corresponding variable column
        elif node.type == 'unary_op':
            left = evaluate_tree(node.left, x)

            # 🔧 Special handling for log to avoid divide-by-zero
            if node.value == 'log':
                left = np.where(left <= 1e-6, np.nan, left)  # Convert bad logs to NaN

            # Suppress NumPy warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = getattr(np, node.value)(left)
            
        elif node.type == 'binary_op':
            left = evaluate_tree(node.left, x)
            right = evaluate_tree(node.right, x)

            # Prevent division by zero explicitly
            if node.value == '/':
                right = np.where(np.abs(right) < 1e-6, np.nan, right)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = getattr(np, node.value)(left, right)
        
        # If result contains NaN, replace with np.inf
        if np.any(np.isnan(result)) or np.any(np.isinf(result)):
            return np.full(x.shape[1], np.inf)

        return result
    except:
        return np.full(x.shape[1], np.inf)  # Penalize trees that crash

# Function to compute the depth of a tree
def get_tree_depth(node: Node) -> int:
    """
    Recursively computes the depth of a tree.
    The depth is defined as the longest path from the root to a leaf.
    """
    if node is None:
        return 0  # An empty tree has depth 0
    left_depth = get_tree_depth(node.left) if node.left else 0
    right_depth = get_tree_depth(node.right) if node.right else 0
    return 1 + max(left_depth, right_depth)  # Add 1 to count the current node
    
# Function to compute the fitness of a tree
def fitness(tree: Node, x: np.ndarray, y: np.ndarray, lambda_penalty=0.01) -> float:
    """
    Computes the fitness score of a tree based on Mean Squared Error (MSE) and a complexity penalty.
    Lower values indicate better trees.
    
    Parameters:
        tree (Node): The root node of the tree.
        x (np.ndarray): The input dataset of shape (num_features, num_samples).
        y (np.ndarray): The target output of shape (num_samples,).
        lambda_penalty (float): The weight of the complexity penalty.
    
    Returns:
        float: The computed fitness score (lower is better).
    """
    try:
        eval_y = evaluate_tree(tree, x)

        # If tree returns None, NaN, or Inf, assign worst score
        if eval_y is None or np.any(np.isnan(eval_y)) or np.any(np.isinf(eval_y)):
            return float('inf')

        mse = np.mean((eval_y - y) ** 2)
        tree_depth = get_tree_depth(tree)

        return mse + lambda_penalty * tree_depth  # Balance accuracy and simplicity
    except:
        return float('inf')


# Selection function using Tournament Selection
# Tournament Selection Function
def tournament_selection(population: list[Node], fitness_dict: dict, tournament_size: int = 2) -> Node:
    """
    Selects the best individual from a random subset of the population using Tournament Selection.
    
    Parameters:
        population (list[Node]): The population of trees.
        fitness_dict (dict): Dictionary containing precomputed fitness scores for each tree.
        tournament_size (int): The number of individuals randomly chosen for the tournament.
    
    Returns:
        Node: The selected best individual.
    """
    # Randomly pick 'tournament_size' individuals
    tournament = random.sample(population, tournament_size)
    
    # Select the best individual (lowest fitness score)
    winner = min(tournament, key=lambda ind: fitness_dict[ind])
    return winner

# Function to select a random subtree
def select_random_subtree(tree: Node):
    """
    Selects a random subtree from the given tree.
    
    Parameters:
        tree (Node): The root node of the tree.
    
    Returns:
        Node: A randomly selected subtree.
    """
    if tree is None or (tree.left is None and tree.right is None):
        return tree  # Return leaf nodes directly
    
    if random.random() < 0.5 and tree.left:
        return select_random_subtree(tree.left)
    elif tree.right:
        return select_random_subtree(tree.right)
    return tree

MUTATIONS = ['point', 'permutation', 'hoist', 'collapse']

# Function to collect all nodes in a tree
def collect_nodes(node: Node, take_root: bool = False) -> list:
    """
    Collects all nodes in a tree.
    
    Parameters:
        node (Node): The root node of the tree.
        take_root (bool): Whether to include the root node in the list.
    
    Returns:
        list: A list of nodes collected from the tree.
    """
    nodes = []
    if node is not None:
        if node.parent is not None or take_root:
            nodes.append(node)
        nodes.extend(collect_nodes(node.left, take_root))
        nodes.extend(collect_nodes(node.right, take_root))
    return nodes

# Function to perform crossover between two trees
def crossover_trees(parent1: Node, parent2: Node) -> tuple:
    """
    Performs subtree crossover between two parent trees by swapping random subtrees.

    Parameters:
        parent1 (Node): First parent tree.
        parent2 (Node): Second parent tree.

    Returns:
        tuple: Two new offspring trees after crossover.
    """
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Collect all nodes except root (we don't want to swap the entire tree)
    nodes1 = collect_nodes(child1, take_root=False)
    nodes2 = collect_nodes(child2, take_root=False)

    if not nodes1 or not nodes2:
        return child1, child2  # No crossover if there are no swappable nodes

    # Select random subtrees to swap
    node1 = random.choice(nodes1)
    node2 = random.choice(nodes2)

    # Get their parents
    parent1 = node1.parent
    parent2 = node2.parent

    if not parent1 or not parent2:
        return child1, child2  # Avoid swapping the root

    # Swap the subtrees
    if parent1.left == node1:
        parent1.left = node2
    else:
        parent1.right = node2

    if parent2.left == node2:
        parent2.left = node1
    else:
        parent2.right = node1

    # Fix parent pointers
    node1.parent, node2.parent = parent2, parent1

    return child1, child2

# Function to apply point mutation
def point_mutation(individual: Node, num_vars: int, max_const: int = 100) -> Node:
    """
    Mutates a random node in the individual, preserving node type.
    
    Parameters:
        individual (Node): The tree to mutate.
        num_vars (int): Number of input variables.
        max_const (int): Maximum integer value for constants.
    
    Returns:
        Node: The mutated tree.
    """
    nodes = collect_nodes(individual, take_root=True)
    mutation_node = np.random.choice(nodes)
    
    if mutation_node.type in ['math_const', 'const', 'var']:
        new_type = np.random.choice(['math_const', 'const', 'var'], p=[0.1, 0.4, 0.5])
        if new_type == 'const':
            mutation_node.type = 'const'
            mutation_node.value = str(random.randint(0, max_const))
        elif new_type == 'math_const':
            mutation_node.type = 'math_const'
            mutation_node.value = np.random.choice(['pi', 'e'])
        elif new_type == 'var':
            mutation_node.type = 'var'
            mutation_node.value = 'x' + str(np.random.randint(0, num_vars))
    elif mutation_node.type == 'unary_op':
        mutation_node.value = np.random.choice(['sin', 'cos', 'exp', 'log'])
    elif mutation_node.type == 'binary_op':
        mutation_node.value = np.random.choice(['+', '-', '*', '/'])
    return individual

# Function to apply permutation mutation
def permutation_mutation(individual: Node) -> Node:
    """
    Swaps left and right subtrees of a random binary operator node.
    
    Parameters:
        individual (Node): The tree to mutate.
    
    Returns:
        Node: The mutated tree.
    """
    binary_nodes = [node for node in collect_nodes(individual, take_root=True) if node.type == 'binary_op']
    if len(binary_nodes) != 0:
        mutation_node = np.random.choice(binary_nodes)
        mutation_node.left, mutation_node.right = mutation_node.right, mutation_node.left
    return individual

# Function to apply hoist mutation
def hoist_mutation(individual: Node) -> Node:
    """
    Selects a random subtree and returns it as a new individual.
    
    Parameters:
        individual (Node): The tree to mutate.
    
    Returns:
        Node: The mutated subtree as a new individual.
    """
    nodes = collect_nodes(individual, take_root=False)
    if len(nodes) == 0:
        return individual
    mutation_node = np.random.choice(nodes)
    mutation_node.parent = None
    return mutation_node

# Function to apply collapse mutation
def collapse_mutation(individual: Node) -> Node:
    """
    Selects a random non-leaf node and replaces it with its left-most leaf child.
    
    Parameters:
        individual (Node): The tree to mutate.
    
    Returns:
        Node: The mutated tree.
    """
    nonleaf_nodes = [node for node in collect_nodes(individual, take_root=True) if node.type in ['unary_op', 'binary_op']]
    if len(nonleaf_nodes) == 0:
        return individual
    mutation_node = np.random.choice(nonleaf_nodes)
    replacement_node = mutation_node
    while replacement_node.left is not None:
        replacement_node = replacement_node.left
    replacement_node.parent = mutation_node.parent
    if mutation_node.parent:
        if mutation_node.parent.left == mutation_node:
            mutation_node.parent.left = replacement_node
        else:
            mutation_node.parent.right = replacement_node
    return individual


def symreg(x: np.ndarray, y: np.ndarray, pop_size: int = 500, max_generations: int = 30, 
           max_depth: int = 10, max_const: int = 100, tournament_size: int = 2, 
           mutation_prob: float = 0.1, stagnation_window: int = 10) -> Node:
    """
    Runs a Genetic Programming algorithm to evolve symbolic expressions for regression.

    Parameters:
        x (np.ndarray): Input feature matrix.
        y (np.ndarray): Target output values.
        pop_size (int): Number of individuals in the population.
        max_generations (int): Maximum number of generations.
        max_depth (int): Maximum depth for the expression trees.
        max_const (int): Maximum value for numerical constants.
        tournament_size (int): Number of individuals in tournament selection.
        mutation_prob (float): Probability of mutation.
        stagnation_window (int): Number of generations with no improvement before stopping.

    Returns:
        Node: The best evolved expression tree.
    """
    # Initialize population
    population = []
    pbar = tqdm(total=pop_size, desc="Creating initial population")

    while len(population) < pop_size:
        tree = generate_random_tree(x.shape[0], 'grow' if np.random.random() < 0.5 else 'full', max_depth, max_const)
        evaluated = evaluate_tree(tree, x)

        if evaluated is not None and not np.any(np.isnan(evaluated)) and not np.any(np.isinf(evaluated)):
            population.append(tree)
            pbar.update(1)

    pbar.close()

    # Debug: Check initial diversity
    print("Initial Population Sample :")
    for i in range(min(5, len(population))):
        print(f"Tree {i+1}: {population[i]}")

    # Track the best individual
    best_fitness = float('inf')
    best_individual = None
    generations_without_improvement = 0

    for generation in tqdm(range(max_generations), desc="Evolving Generations"):
        offspring = []

        for _ in range(pop_size):
            if np.random.rand() < mutation_prob:
                parent = tournament_selection(population, {ind: fitness(ind, x, y) for ind in population}, tournament_size)
                mutation_type = random.choice(MUTATIONS)

                mutated_child = None
                if mutation_type == 'point':
                    mutated_child = point_mutation(parent.copy(), x.shape[0], max_const)
                elif mutation_type == 'permutation':
                    mutated_child = permutation_mutation(parent.copy())
                elif mutation_type == 'hoist':
                    mutated_child = hoist_mutation(parent.copy())
                elif mutation_type == 'collapse':
                    mutated_child = collapse_mutation(parent.copy())

                if mutated_child is not None:
                    evaluated = evaluate_tree(mutated_child, x)
                    if evaluated is not None and not np.any(np.isnan(evaluated)) and not np.any(np.isinf(evaluated)):
                        offspring.append(mutated_child)  # Only keep valid mutations

            else:
                fitness_dict = {ind: fitness(ind, x, y) for ind in population}
                parent1 = tournament_selection(population, fitness_dict, tournament_size)
                parent2 = tournament_selection(population, fitness_dict, tournament_size)

                while parent2 == parent1:
                    parent2 = tournament_selection(population, fitness_dict, tournament_size)

                child1, child2 = crossover_trees(parent1, parent2)

                for child in [child1, child2]:
                    evaluated = evaluate_tree(child, x)
                    if evaluated is not None and not np.any(np.isnan(evaluated)) and not np.any(np.isinf(evaluated)):
                        offspring.append(child)

        # Select survivors and update population
        population = sorted(population + offspring, key=lambda i: fitness(i, x, y))[:pop_size]

        # Debug: Check  diversity
        print(f"New Population n°{generation} Sample:")
        for i in range(min(5, len(population))):
            print(f"Tree {i+1}: {population[i]}")

        # Check best individual
        current_best = min(population, key=lambda i: fitness(i, x, y))
        current_fitness = fitness(current_best, x, y)

        # Debugging: Track evolution progress
        print(f"Generation {generation}: Best Fitness = {best_fitness}")
        print(f"Best Formula so far: {best_individual}")

        # Stagnation handling
        if current_fitness < best_fitness:
            best_fitness = current_fitness
            best_individual = current_best
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        if generations_without_improvement >= stagnation_window:
            print(f"⚠️ Stagnation detected at Generation {generation}, stopping early.")
            break

    return best_individual


# Run Genetic Programming on Training Data
# Load dataset
problem = np.load('data/problem_1.npz')  # Adjust path if necessary
x = problem['x']
y = problem['y']

# Ensure x and y have compatible shapes
print(f"Shape of x: {x.shape}")
print(f"Shape of y: {y.shape}")

# Define the maximum constant value based on dataset properties
max_const = int(np.rint(np.maximum(np.max(np.abs(x)), np.max(np.abs(y)))))

# Run symbolic regression
best_tree = symreg(
    x, y,
    pop_size=1000,  # Increase population size
    max_generations=100,  # Allow more evolution
    max_depth=10,  # Allow deeper trees
    max_const=max_const,
    tournament_size=5,  # Stronger selection pressure
    mutation_prob=0.3,  # Increase mutation rate
    stagnation_window=20  # Reduce stagnation threshold
)

# Print the best found formula
print("Best Found Formula:", best_tree)
