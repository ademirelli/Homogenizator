# 0. Import necessary libraries
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

# 1. Ask for G(p) and beta
def define_G():
    p = sp.symbols('p')
    expression = input("Enter continuous and coercive function G(p): ")
    G = sp.lambdify(p, expression, 'numpy')
    return G, sp.sympify(expression)
beta = float(input("Enter beta: "))

# 2. Find the minimum of G(p) and shift the minimum to origin if necessary
num_points = 1000
start_p = -100
end_p = 100
G, G_expr = define_G()
p = sp.symbols('p')
G_prime = sp.diff(G_expr, p)
critical_points = sp.solve(G_prime, p)
valid_critical_points = []
for point in critical_points:
    if point.is_real:
        valid_critical_points.append(float(point))
    else:
        real_part = sp.re(point)
        valid_critical_points.append(float(real_part))
valid_critical_points.sort()
critical_points = valid_critical_points

# Evaluate G(p) at each critical point to find the minimum value
min_val0 = float('inf')  # Initialize with positive infinity
min_p0 = None  # Initialize with None
for point in critical_points:
    G_val = G(point)
    if G_val < min_val0:
        min_val0 = G_val
        min_p0 = point

if min_val0 != 0 or min_p0 != 0:
    critical_points = sp.solve(G_prime, p)
    valid_critical_points = []
    for point in critical_points:
        if point.is_real:
            valid_critical_points.append(float(point))
        else:
            real_part = sp.re(point)
            valid_critical_points.append(float(real_part))
    valid_critical_points.sort()
    critical_points = valid_critical_points
    G_expr = G_expr.subs(p, p + min_p0) - min_val0
    G = sp.lambdify(p, G_expr, 'numpy')
    for k in range(len(valid_critical_points)):
        critical_points[k] = critical_points[k] - min_p0

# 4. Define G_i functions based on critical points
G_functions = {}
if len(valid_critical_points) == 1:
    G_functions[f'G_1'] = lambda p : G(p) if p <= valid_critical_points[0] else None
    G_functions[f'G_2'] = lambda p : G(p) if p > valid_critical_points[0] else None
else:
    for i in range(len(valid_critical_points)+1):
        if i == 0:
            G_functions[f'G_1'] = lambda p : G(p) if p <= valid_critical_points[0] else None
        elif i == len(valid_critical_points):
            G_functions[f'G_{i+1}'] = lambda p : G(p) if p > valid_critical_points[-1] else None
        else:
            G_functions[f'G_{i+1}'] = lambda p, a=valid_critical_points[i-1], b=valid_critical_points[i]: G(p) if a < p <= b else None

# 5. Determine the domain of each G_i function
domains = {}
if len(valid_critical_points) == 1:
    domains['G_1'] = (-100, valid_critical_points[0])
    domains['G_2'] = (valid_critical_points[0], 100)
else:
    for i in range(len(valid_critical_points)+1):
        if i == 0:
            domains['G_1'] = (-100, valid_critical_points[0])
        elif i == len(valid_critical_points):
            domains[f'G_{i+1}'] = (valid_critical_points[-1], 100)
        else:
            domains[f'G_{i+1}'] = (valid_critical_points[i-1], valid_critical_points[i])

# Generate linearly spaced arrays for each G_i function
arrays = {}
for key, domain in domains.items():
    lower_bound, upper_bound = domain
    p_values_real = np.linspace(float(lower_bound), float(upper_bound), num_points)
    G_values = [G(float(p_real)) for p_real in p_values_real]
    function_array = np.array([(float(p_real), G_val) for p_real, G_val in zip(p_values_real, G_values)])
    arrays[key] = function_array

# 7. Find arrays for inverse of each G_i function
inverse_arrays = {}
for key, array in arrays.items():
    inverse_arrays[f'{key}_inverse'] = np.array([(float(y), float(x)) for x, y in array])  # Convert to float explicitly
arrays.update(inverse_arrays)

# 8. Define the function V(x,omega)
def V3(x, omegas):
    floor_x = np.floor(x).astype(int)
    ceil_x = np.ceil(x).astype(int)
    if np.equal(floor_x, ceil_x).all():
        return omegas[floor_x]
    else:
        return omegas[floor_x] + (x - floor_x) * (omegas[ceil_x] - omegas[floor_x])
x_values = np.linspace(0, 10, 1000)
u = np.random.uniform(0, 1, size=1)
omega = np.random.uniform(0, 1, size=1000)
V3_values = V3(x_values + u, omega)
arrays['V3'] = np.array(list(zip(x_values, V3_values)))

# 9. Define the Ergodic approximation
def ergodic_approximation(G_inverse_array, lambda_val, beta, V_array, B, N):
    # Calculate the step size
    delta_x = 2 * B / (2 * N)
    # Initialize sum
    integral_sum = 0
    # Perform summation
    for k in range(-N, N+1):
        # Calculate x_k
        x_k = k * B / N
        # Find the nearest index in V_array
        index = np.abs(V_array[:, 0] - x_k).argmin()
        # Find the value of V at that index
        V_val = V_array[index, 1]
        # Evaluate lambda - beta * V_val
        lambda_minus_beta_V = lambda_val - beta * V_val
        # Find the nearest index in G_inverse_array
        indexx = np.abs(G_inverse_array[:, 0] - lambda_minus_beta_V).argmin()
        # Find the value of G_i^{-1} at that index
        G_inverse_val = G_inverse_array[indexx, 1]
        # Update the integral sum
        integral_sum += G_inverse_val * delta_x
    # Return the approximate value
    return integral_sum / (2 * B)

theta_1_beta = 0.0
theta_2_beta = 0.0

#10. Define Base Case 1
def BaseCase1(G, critical_points, beta):
    # 10.i. Define the lambda range
    lambda_range = np.arange(beta, beta + 99)
    theta_1_lambda = [ergodic_approximation(arrays['G_1_inverse'], lam, beta, arrays['V3'], 10, 1000) for lam in
                      lambda_range]
    theta_2_lambda = [ergodic_approximation(arrays['G_2_inverse'], lam, beta, arrays['V3'], 10, 1000) for lam in
                      lambda_range]

    global theta_1_beta, theta_2_beta

    # 10.ii. Find theta_1(beta) and theta_2(beta) values
    theta_1_beta = ergodic_approximation(arrays['G_1_inverse'], beta, beta, arrays['V3'], 10, 1000)
    theta_2_beta = ergodic_approximation(arrays['G_2_inverse'], beta, beta, arrays['V3'], 10, 1000)

    # 10.iii. Define the function bar_H
    def bar_H(theta, theta_1_beta, theta_2_beta, beta):
        if theta < theta_1_beta:
            return np.interp(theta, theta_1_lambda[::-1], lambda_range[::-1])
        elif theta >= theta_1_beta and theta <= theta_2_beta:
            return beta
        elif theta > theta_2_beta:
            return np.interp(theta, theta_2_lambda, lambda_range)

    plt.axhline(y=beta, color='g', linestyle='--')
    plt.scatter(theta_1_beta, beta, color='r')
    plt.scatter(theta_2_beta, beta, color='r')
    plt.text(theta_1_beta, beta, f'({theta_1_beta:.2f}, {beta:.2f})', fontsize=8, ha='right', va='bottom')
    plt.text(theta_2_beta, beta, f'({theta_2_beta:.2f}, {beta:.2f})', fontsize=8, ha='left', va='bottom')

    # 10.iv. Define theta_range
    theta_range = np.linspace(min(theta_1_lambda), max(theta_2_lambda), 1000)

    # 10.v. Call bar_H for each theta and collect the results
    result = [bar_H(theta, theta_1_beta, theta_2_beta, beta) for theta in theta_range]

    # 10.vi. Return the collected results
    return theta_range, result

# 11. Define Base Case 2
def BaseCase2(G, critical_points, beta):
    # 11.o. Sort the critical points
    critical_points = sorted(critical_points)

    # 11.i. Define theta_i(lambda) functions for each G_i function
    lambda_range = np.arange(beta, beta + 99)
    theta_lambda_functions = {}
    for i in range(1, len(critical_points) + 2):
        G_inverse_key = f'G_{i}_inverse'
        theta_i_lambda = [ergodic_approximation(arrays[G_inverse_key], lam, beta, arrays['V3'], 10, 1000) for lam in
                          lambda_range]
        theta_lambda_functions[f'theta_{i}_lambda'] = theta_i_lambda

    # 11.ii. Find the jump points
    def find_tilde_p(G, critical_point, end_p, tolerance=1e-5):
        def root_func(p):
            return G(p) - G(critical_point)

        left = critical_point
        right = end_p
        while right - left > tolerance:
            mid = (left + right) / 2
            if root_func(mid) == 0:
                return mid
            elif root_func(mid) * root_func(left) < 0:
                right = mid
            else:
                left = mid
        return (left + right) / 2

    # 11.iii. Find the branch number of the point jumped to
    def find_interval_index(critical_points, tilde_p):
        for i in range(len(critical_points) - 1):
            if critical_points[i] <= tilde_p <= critical_points[i + 1]:
                return i + 2
        return len(critical_points) + 1

    # 11.iv. Store jump points and branch numbers
    jump_points = []
    jump_branches = []
    jump_points.append(critical_points[0])
    # jump_points.append(G(critical_points[0])+beta)
    jump_branches.append(1)
    cutoff_points = []
    cutoff_points.append(
        ergodic_approximation(arrays['G_1_inverse'], G(critical_points[0]) + beta, beta, arrays['V3'], 10, 1000))
    a = critical_points[0]
    dummy = 1
    while dummy < len(critical_points):
        tilde_p_next = find_tilde_p(G, a, 0)
        jump_points.append(tilde_p_next)
        dummy = find_interval_index(critical_points, tilde_p_next)
        cutoff_points.append(
            ergodic_approximation(arrays[f'G_{dummy}_inverse'], G(tilde_p_next) + beta, beta, arrays['V3'], 10, 1000))
        jump_branches.append(dummy)
        a = tilde_p_next

    # 11.v. Define theta_{2N-1}(beta) and theta_{2N}(beta)
    lambda_range = np.arange(beta, beta + 99)
    theta_one_before_last_lambda = [ergodic_approximation(arrays[f'G_{len(valid_critical_points)}_inverse'], lam, beta, arrays['V3'], 10, 1000)
                                    for lam in lambda_range]
    theta_last_lambda = [ergodic_approximation(arrays[f'G_{len(valid_critical_points) + 1}_inverse'], lam, beta, arrays['V3'], 10, 1000) for lam
                         in lambda_range]

    # 11.vi. Find theta_{2N-1}(beta) and theta_{2N}(beta) values
    theta_first_lambda = [ergodic_approximation(arrays['G_1_inverse'], lam, beta, arrays['V3'], 10, 1000) for lam in
                          lambda_range]
    theta_one_before_last_beta = ergodic_approximation(arrays[f'G_{len(valid_critical_points)}_inverse'], beta, beta, arrays['V3'], 10, 1000)
    theta_last_beta = ergodic_approximation(arrays[f'G_{len(valid_critical_points) + 1}_inverse'], beta, beta, arrays['V3'], 10, 1000)
    jump_points.append(theta_one_before_last_beta)
    jump_branches.append(len(critical_points))
    jump_points.append(theta_last_beta)
    jump_branches.append(len(critical_points) + 1)
    cutoff_points.append(jump_points[-2])
    cutoff_points.append(jump_points[-1])
    print(jump_points)
    print(jump_branches)
    print(cutoff_points)

    # 11.vii. Define the bar_H function
    def bar_H(theta, critical_points, cutoff_points, jump_branches, beta, lambda_range):
        for i in range(len(cutoff_points) - 1):
            if theta < cutoff_points[0]:
                return np.interp(theta, theta_lambda_functions['theta_1_lambda'][::-1],
                                 lambda_range[::-1])  # theta_lambda_functions['theta_1_lambda'][::-1][0]
            elif theta >= cutoff_points[-1]:
                return np.interp(theta, theta_last_lambda, lambda_range)  # theta_last_lambda
            elif cutoff_points[i] <= theta <= cutoff_points[i + 1]:
                if i % 2 == 0:
                    return G(critical_points[i]) + beta
                else:
                    return np.interp(theta, theta_lambda_functions[f'theta_{jump_branches[i]}_lambda'][::-1],
                                     lambda_range[
                                     ::-1])  # np.interp(theta, theta_something(jump_branches[i // 2], critical_points[i // 2], beta, omega_0)[::-1], lambda_range[::-1])

    plt.scatter(cutoff_points,
                [bar_H(cutoff_point, critical_points, cutoff_points, jump_branches, beta, lambda_range) for cutoff_point
                 in cutoff_points], color='r')
    for i, cutoff_point in enumerate(cutoff_points):
        plt.text(cutoff_point, bar_H(cutoff_point, critical_points, cutoff_points, jump_branches, beta, lambda_range),
                 f'({cutoff_point:.2f}, {bar_H(cutoff_point, critical_points, cutoff_points, jump_branches, beta, lambda_range):.2f})',
                 fontsize=8, ha='right', va='bottom')

    # Horizontal lines at cutoff points with labels
    for i in range(len(cutoff_points) - 1):
        if i % 2 == 0:
            y_value = G(critical_points[i]) + beta
            plt.axhline(y=y_value, color='g', linestyle='--')

    # 11.ix. Define theta_range
    theta_range = np.linspace(min(theta_first_lambda), max(theta_last_lambda), 1000)

    # 11.x. Call bar_H for each theta and collect the results
    result = [bar_H(theta, critical_points, cutoff_points, jump_branches, beta, lambda_range) for theta in
                    theta_range]

    # 11.xi. Return the collected results
    return theta_range, result

# 12. Define Gluing at the Origin
def Gluing1(G):
    def barG_1(p):
        if p <= 0:
            return G(p)
        else:
            return 100 * p ** 2
    def barG_2(p):
        if p >= 0:
            return G(p)
        else:
            return 100 * p ** 2
    def reflected_barG_2(p):
        return barG_2(-p)
    critical_points_barG_1 = [p for p in critical_points if p <= 0]
    critical_points_barG_1.sort()
    print(critical_points_barG_1)
    critical_points_reflected_barG_2 = [-p for p in critical_points if p >= 0]
    critical_points_reflected_barG_2.sort()
    print(critical_points_reflected_barG_2)
    return barG_1, critical_points_barG_1, reflected_barG_2, critical_points_reflected_barG_2

# 13. Define the second gluing
def Gluing2(G, critical_points):
    max_val = max([G(p) for p in critical_points])
    max_critical_point = critical_points[np.argmax([G(p) for p in critical_points])]
    min_val = min([G(p) for p in critical_points[:-2]])
    min_critical_point = critical_points[np.argmin([G(p) for p in critical_points[:-2]])]
    def tildeG_1(p):
        if p <= max_critical_point:
            return 10*(p - max_critical_point)**2 + max_val
        else:
            return G(p)
    def tildeG_2(p):
        if p >= min_critical_point:
            return 10*(p - min_critical_point)**2 + min_val
        else:
            return G(p)
    def shifted_tildeG_2(p):
        return tildeG_2(p + min_critical_point) - min_val
    critical_points_tildeG_1 = [p for p in critical_points if p > max_critical_point]
    critical_points_tildeG_1.sort()
    print(critical_points_tildeG_1)
    critical_points_shifted_tildeG_2 = [p - min_critical_point for p in critical_points if p <= min_critical_point]
    critical_points_shifted_tildeG_2.sort()
    print(critical_points_shifted_tildeG_2)
    return tildeG_1, critical_points_tildeG_1, shifted_tildeG_2, critical_points_shifted_tildeG_2

def Gluing3(G, critical_points):
    max_val = max([G(p) for p in critical_points])
    max_critical_point = critical_points[np.argmax([G(p) for p in critical_points])]
    min_val = min([G(p) for p in critical_points[:-2]])
    min_critical_point = critical_points[np.argmin([G(p) for p in critical_points[:-2]])]
    def tildesG_1(p):
        if p <= max_critical_point:
            return 10*(p - max_critical_point)**2 + max_val
        else:
            return G(p)
    def tildesG_2(p):
        if p >= max_critical_point:
            return 10*(p - max_critical_point)**2 + max_val
        else:
            return G(p)
    min_val_before_max = min([tildesG_2(p) for p in critical_points if p < max_critical_point])
    min_point_before_max = critical_points[np.argmin([tildesG_2(p) for p in critical_points if p < max_critical_point])]
    def shifted_tildesG_2(p):
        return tildesG_2(p + min_point_before_max) - min_val_before_max
    def reflected_shifted_tildesG_2(p):
        return shifted_tildesG_2(-p)
    critical_points_tildesG_1 = [p for p in critical_points if p > max_critical_point]
    critical_points_tildesG_1.sort()
    print(critical_points_tildesG_1)
    critical_points_reflected_shifted_tildesG_2 = [- p + min_point_before_max for p in critical_points if p < max_critical_point]
    critical_points_reflected_shifted_tildesG_2.sort()
    print(critical_points_reflected_shifted_tildesG_2)
    return tildesG_1, critical_points_tildesG_1, reflected_shifted_tildesG_2, critical_points_reflected_shifted_tildesG_2

def Homogenizeti(G, critical_points, beta):
    max_val = max([G(p) for p in critical_points])
    max_val_index = np.argmax([G(p) for p in critical_points])
    min_val = min([G(p) for p in critical_points if p != 0] or [0])
    auxillary_critical_value = min([G(p) for p in critical_points[:max_val_index]] or [0])
    if len(critical_points) == 1:
        print("BaseCase1")
        return BaseCase1(G, critical_points, beta)
    elif all(cp <= 0 for cp in critical_points) and beta > max_val - min_val:
        print("BaseCase2")
        return BaseCase2(G, critical_points, beta)
    elif any(cp > 0 for cp in critical_points):
        print("GluingAtTheOrigin")
        a, b, c, d = Gluing1(G)
        A = Homogenizeti(a, b, beta)
        B = Homogenizeti(c, d, beta)
        return A[0], A[1], B[0], B[1]
    elif max_val - auxillary_critical_value < beta:
        print("Gluing2")
        a, b, c, d = Gluing2(G, critical_points)
        A = Homogenizeti(a, b, beta)
        B = Homogenizeti(c, d, beta)
        min_critical_point = min([G(p) for p in critical_points[:-2]])
        min_critical_value = G(min_critical_point)
        modified_theta_range = [x - min_critical_point for x in B[0]]
        modified_barH_range = [x + min_critical_value for x in B[1]]
        return A[0], A[1], modified_theta_range, modified_barH_range, min_critical_point, min_critical_value
    elif max_val - auxillary_critical_value > beta:
        print("Gluing3")
        a, b, c, d = Gluing3(G, critical_points)
        A = Homogenizeti(a, b, beta)
        B = Homogenizeti(c, d, beta)
        max_critical_point = critical_points[np.argmax([G(p) for p in critical_points])]
        min_val_before_max = min([G(p) for p in critical_points if p < max_critical_point])
        min_point_before_max = critical_points[np.argmin([G(p) for p in critical_points if p < max_critical_point])]
        modified_theta_range = [-x + min_point_before_max for x in B[0]]
        modified_barH_range = [x + min_val_before_max for x in B[1]]
        return A[0], A[1], B[0], modified_barH_range, min_point_before_max, min_val_before_max, min_point_before_max, max_critical_point

min_critical_point = min([G(p) for p in critical_points[:-2]]) if len(critical_points) > 1 else 0
min_critical_value = G(min_critical_point)

max_critical_point = critical_points[np.argmax([G(p) for p in critical_points])]
min_val_before_max = min([G(p) for p in critical_points if p < max_critical_point]) if len(critical_points) > 1 else 0
min_point_before_max = critical_points[np.argmin([G(p) for p in critical_points if p < max_critical_point])] if len(critical_points) > 1 else 0

print(min_p0)
print(min_val0)

if len(Homogenizeti(G, critical_points, beta)) == 2:
    theta_range = [theta + min_p0 for theta in Homogenizeti(G, critical_points, beta)[0]]
    bar_H_values = [bar_H - min_val0 for bar_H in Homogenizeti(G, critical_points, beta)[1]]
    plt.plot(theta_range, bar_H_values)
    plt.xlabel('Theta')
    plt.ylabel('bar_H')
    plt.title('bar_H function')
    plt.ylim(-1,41)
    plt.grid(True)
    plt.show()
elif len(Homogenizeti(G, critical_points, beta)) == 4:
    theta_range_1 = [theta + min_p0 for theta in Homogenizeti(G, critical_points, beta)[0]]
    theta_range_1 = np.array(theta_range_1)
    bar_H_values_1 = [bar_H - min_val0 for bar_H in Homogenizeti(G, critical_points, beta)[1]]
    bar_H_values_1 = np.array(bar_H_values_1)
    mask_theta_negative = theta_range_1 < 0
    plt.plot(theta_range_1[mask_theta_negative], bar_H_values_1[mask_theta_negative], color='blue')
    theta_range_2 = [-theta + min_p0 for theta in Homogenizeti(G, critical_points, beta)[2]]
    theta_range_2 = np.array(theta_range_2)
    bar_H_values_2 = [bar_H - min_val0 for bar_H in Homogenizeti(G, critical_points, beta)[3]]
    bar_H_values_2 = np.array(bar_H_values_2)
    mask_theta_positive = theta_range_2 > 0
    plt.plot(theta_range_2[mask_theta_positive], bar_H_values_2[mask_theta_positive], color='blue')
    plt.xlabel('Theta')
    plt.ylabel('bar_H')
    plt.title('bar_H function')
    plt.ylim(-1,41)
    plt.grid(True)
    plt.show()
elif len(Homogenizeti(G, critical_points, beta)) == 6:
    theta_range_1 = Homogenizeti(G, critical_points, beta)[0]
    bar_H_values_1 = Homogenizeti(G, critical_points, beta)[1]
    theta_range_2 = Homogenizeti(G, critical_points, beta)[2]
    bar_H_values_2 = Homogenizeti(G, critical_points, beta)[3]
    theta_range_1 = np.array(theta_range_1)
    theta_range_2 = np.array(theta_range_2)
    bar_H_values_1 = np.array(bar_H_values_1)
    bar_H_values_2 = np.array(bar_H_values_2)
    mask_theta_negative = theta_range_1 > -3.38
    mask_theta_positive = theta_range_2 < -3.38
    plt.plot(theta_range_1[mask_theta_negative], bar_H_values_1[mask_theta_negative], color='blue')
    plt.plot(theta_range_2[mask_theta_positive], bar_H_values_2[mask_theta_positive], color='blue')
    plt.ylim(-1,41)
    plt.xlabel('Theta')
    plt.ylabel('bar_H')
    plt.title('bar_H function')
    plt.grid(True)
    plt.show()
elif len(Homogenizeti(G, critical_points, beta)) == 8:
    theta_range_1 = Homogenizeti(G, critical_points, beta)[0]
    bar_H_values_1 = Homogenizeti(G, critical_points, beta)[1]
    theta_range_2 = Homogenizeti(G, critical_points, beta)[2]
    bar_H_values_2 = Homogenizeti(G, critical_points, beta)[3]

    theta_range_1 = np.array(theta_range_1)
    theta_range_2 = np.array(theta_range_2)
    bar_H_values_1 = np.array(bar_H_values_1)
    bar_H_values_2 = np.array(bar_H_values_2)

    mask_theta_negative = (theta_range_2 < -3.16)
    mask_theta_positive = (theta_range_1 > -3.16)

    plt.plot(theta_range_2[mask_theta_negative], bar_H_values_2[mask_theta_negative], color='blue')
    plt.plot(theta_range_1[mask_theta_positive], bar_H_values_1[mask_theta_positive], color='blue')
    plt.xlabel('Theta')
    plt.ylabel('bar_H')
    plt.title('bar_H function')
    plt.grid(True)
    plt.ylim(-1,41)
    plt.show()

else:
    print("Error. Update the input function G(p).")

# Example of Base Case 1: p**2
# Example of Base Case 2: 4.5*p**4 + 17*p**3 + 16.5*p**2
# Example of Gluing at the Origin: p**6 - 5*p**4 + 0.25 * p**3 + 7*p**2
# Example of Gluing 2: p**6 - 5*p**4 + 0.25 * p**3 + 7*p**2 - 2*p
# Example of Gluing 3: p**6 - 5*p**4 + 0.25 * p**3 + 7*p**2 - 2*p
# A new example: 1.2*p**6 - 6*p**4 + 0.25 * p**3 + 7*p**2 - 1.2*p
# Last example: 1.2*(x+1.595)**6 - 6*(x+1.595)**4 + 0.25*(x+1.595)**3 + 7*(x+1.595)**2 - 1.2*(x+1.595) + 2.166