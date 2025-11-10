import numpy as np

def read_lp_from_file(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    obj_line = lines[0]
    parts = obj_line.split(":")
    sense = parts[0].strip().lower()
    c = np.array(list(map(float, parts[1].split())))

    A, b, signs = [], [], []

    for line in lines[1:]:
        if not line.startswith("c:"):
            continue
        tokens = line.replace("c:", "").strip().split()
        rhs = float(tokens[-1])
        sign = tokens[-2]
        coeffs = list(map(float, tokens[:-2]))
        A.append(coeffs)
        b.append(rhs)
        signs.append(sign)

    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    return c, A, b, signs, sense


def canonical_form(c, A, b, signs):
    """
    Преобразует задачу в каноническую форму для симплекс-метода
    """
    m, n = A.shape
    
    slack_vars = 0
    artificial_vars = 0
    
    for s in signs:
        if s == "<=":
            slack_vars += 1
        elif s == ">=":
            slack_vars += 1
            artificial_vars += 1
        elif s == "=":
            artificial_vars += 1
    
    total_vars = n + slack_vars + artificial_vars
    
    A_ext = np.zeros((m, total_vars))
    c_ext = np.zeros(total_vars)
    
    A_ext[:, :n] = A
    c_ext[:n] = c
    
    slack_idx = n
    artificial_idx = n + slack_vars
    
    base = []
    artificial_list = []
    
    for i in range(m):
        if signs[i] == "<=":
            A_ext[i, slack_idx] = 1
            base.append(slack_idx)
            slack_idx += 1
            
        elif signs[i] == ">=":
            A_ext[i, slack_idx] = -1
            A_ext[i, artificial_idx] = 1
            base.append(artificial_idx)
            artificial_list.append(artificial_idx)
            slack_idx += 1
            artificial_idx += 1
            
        elif signs[i] == "=":
            A_ext[i, artificial_idx] = 1
            base.append(artificial_idx)
            artificial_list.append(artificial_idx)
            artificial_idx += 1
    
    return A_ext, b.copy(), c_ext, artificial_list, base


def simplex_phase1(A, b, artificial_vars, base):
    """
    Фаза 1 симплекс-метода: убираем искусственные переменные
    """
    m, n = A.shape
    
    c_phase1 = np.zeros(n)
    for art_idx in artificial_vars:
        c_phase1[art_idx] = 1
    
    iterations = 0
    max_iterations = 100
    
    while iterations < max_iterations:
        iterations += 1
        
        cb = c_phase1[base]
        
        z = np.zeros(n)
        for j in range(n):
            for i in range(m):
                z[j] += cb[i] * A[i, j]
        
        reduced_cost = c_phase1 - z
        
        if np.all(reduced_cost >= -1e-8):
            break
        
        entering = np.argmin(reduced_cost)
        
        # Вычисляем направления
        direction = A[:, entering]
        
        ratios = np.full(m, np.inf)
        for i in range(m):
            if direction[i] > 1e-8:
                ratios[i] = b[i] / direction[i]
        
        if np.all(ratios == np.inf):
            return None, None, None, False 
        
        leaving_idx = np.argmin(ratios)
        leaving_var = base[leaving_idx]
        
        base[leaving_idx] = entering
        
        pivot = A[leaving_idx, entering]
        A[leaving_idx, :] /= pivot
        b[leaving_idx] /= pivot
        
        for i in range(m):
            if i != leaving_idx:
                factor = A[i, entering]
                A[i, :] -= factor * A[leaving_idx, :]
                b[i] -= factor * b[leaving_idx]
    
    x = np.zeros(n)
    for i, var_idx in enumerate(base):
        x[var_idx] = b[i]
    
    artificial_in_base = False
    for art_idx in artificial_vars:
        if art_idx in base:
            idx_in_base = base.index(art_idx)
            if abs(b[idx_in_base]) > 1e-8:
                artificial_in_base = True
                break
    
    if artificial_in_base:
        return None, None, None, False
    
    return A, b, base, True


def simplex(c, A, b, base):
    """
    Основная фаза симплекс-метода
    """
    m, n = A.shape
    
    iterations = 0
    max_iterations = 100
    
    while iterations < max_iterations:
        iterations += 1
        
        cb = c[base]
        
        z = np.zeros(n)
        for j in range(n):
            for i in range(m):
                z[j] += cb[i] * A[i, j]
        
        reduced_cost = c - z
        
        if np.all(reduced_cost >= -1e-8):
            break
        
        entering = np.argmin(reduced_cost)
        
        direction = A[:, entering]
        
        ratios = np.full(m, np.inf)
        for i in range(m):
            if direction[i] > 1e-8:
                ratios[i] = b[i] / direction[i]
        
        if np.all(ratios == np.inf):
            return None, None, False
        
        leaving_idx = np.argmin(ratios)
        leaving_var = base[leaving_idx]
        
        base[leaving_idx] = entering
        
        pivot = A[leaving_idx, entering]
        A[leaving_idx, :] /= pivot
        b[leaving_idx] /= pivot
        
        for i in range(m):
            if i != leaving_idx:
                factor = A[i, entering]
                A[i, :] -= factor * A[leaving_idx, :]
                b[i] -= factor * b[leaving_idx]
    
    x = np.zeros(n)
    for i, var_idx in enumerate(base):
        x[var_idx] = b[i]
    
    objective_value = np.dot(c, x)
    
    return x, objective_value, True


def solve_lp(filename):
    c, A, b, signs, sense = read_lp_from_file(filename)

    print("=== Прочитанная задача ===")
    print("Целевая функция:", sense, c)
    print("Ограничения:")
    for i in range(len(signs)):
        print(f"  {A[i]} {signs[i]} {b[i]}")
    
    c_original = c.copy()
    
    if sense == "max":
        c = -c
    
    A_ext, b_ext, c_ext, artificial_vars, base = canonical_form(c, A, b, signs)
    
    print(f"\nРазмерность расширенной матрицы A: {A_ext.shape}")
    print(f"Искусственные переменные: {artificial_vars}")
    print(f"Начальный базис: {base}")
    print("Расширенная матрица A:")
    print(A_ext)
    print("Вектор b:", b_ext)
    print("Расширенный вектор c:", c_ext)
    
    if artificial_vars:
        print("\n=== Фаза 1 ===")
        A_phase1 = A_ext.copy()
        b_phase1 = b_ext.copy()
        A_phase1, b_phase1, base, success = simplex_phase1(A_phase1, b_phase1, artificial_vars, base)
        
        if not success:
            print("❌ Задача не имеет допустимого решения")
            return
        print(f"Базис после фазы 1: {base}")
        print("Матрица A после фазы 1:")
        print(A_phase1)
        print("Вектор b после фазы 1:", b_phase1)
    else:
        A_phase1 = A_ext
        b_phase1 = b_ext
    
    print("\n=== Фаза 2 ===")
    
    c_final = np.zeros(A_phase1.shape[1])
    c_final[:len(c)] = c
    
    for art_idx in artificial_vars:
        c_final[art_idx] = 0
    
    print("Целевая функция для фазы 2:", c_final)
    
    x, val, success = simplex(c_final, A_phase1, b_phase1, base)
    
    if not success:
        print("❌ Решение не найдено")
        return
    
    print("Полное решение (все переменные):", x)
    
    if sense == "max":
        val_original = np.dot(c_original, x[:len(c_original)])
        val = val_original
    else:
        val_original = val
    
    print("\n=== Оптимальное решение ===")
    print(f"Оптимальное значение целевой функции Z = {val_original:.4f}")
    for i in range(len(c_original)):
        print(f"x{i+1} = {x[i]:.4f}")
    
    print("\n=== Проверка ограничений ===")
    for i in range(A.shape[0]):
        lhs = np.dot(A[i], x[:len(c_original)])
        rhs = b[i]
        sign = signs[i]
        satisfied = False
        if sign == "<=" and lhs <= rhs + 1e-6:
            satisfied = True
        elif sign == ">=" and lhs >= rhs - 1e-6:
            satisfied = True
        elif sign == "=" and abs(lhs - rhs) < 1e-6:
            satisfied = True
        print(f"Ограничение {i+1}: {lhs:.4f} {sign} {rhs:.4f} -> {'✓' if satisfied else '✗'}")


if __name__ == "__main__":
    solve_lp("input.txt")