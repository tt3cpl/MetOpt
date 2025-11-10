import numpy as np, math, time
import matplotlib.pyplot as plt
from sympy import sympify, symbols, lambdify

def parse_function(expr_str):
    x = symbols('x')
    return lambdify(x, sympify(expr_str), modules=['numpy', 'math'])

def estimate_L(f, a, b, n=300):
    xs = np.linspace(a, b, n)
    ys = f(xs)
    return 1.2 * np.max(np.abs(np.diff(ys) / np.diff(xs)))

def piyavskii(f, a, b, eps, L=None, max_iter=5000):
    if L is None:
        L = estimate_L(f, a, b)
    xs, ys = [a, b], [f(a), f(b)]
    t0 = time.time()
    for it in range(max_iter):
        best_i, best_m = 0, float('inf')
        for i in range(len(xs) - 1):
            xl, xr = xs[i], xs[i + 1]
            yl, yr = ys[i], ys[i + 1]
            x_c = 0.5 * (xl + xr) + (yl - yr) / (2 * L)
            x_c = min(max(x_c, xl), xr)
            m_val = 0.5 * (yl + yr) - L * (xr - xl) / 2
            if m_val < best_m:
                best_m, best_i, x_best = m_val, i, x_c
        y_best = f(x_best)
        xs.insert(best_i + 1, x_best)
        ys.insert(best_i + 1, y_best)
        if xs[best_i + 2] - xs[best_i] < eps:
            break
    idx = np.argmin(ys)
    return {
        'x': xs[idx],
        'f': ys[idx],
        'L': L,
        'iter': it + 1,
        'time': time.time() - t0,
        'xs': xs,
        'ys': ys
    }

def plot_result(f, a, b, res):
    X = np.linspace(a, b, 1000)
    plt.plot(X, f(X), label='функция f(x)')
    plt.plot(res['xs'], res['ys'], 'o-', label='выборки точек')
    plt.axvline(res['x'], color='gray', ls='--', label='минимум')
    plt.legend()
    plt.title('Глобальный поиск по методу Пиявского')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.show()
    print(f"x* = {res['x']:.5f}, f* = {res['f']:.5f}, итераций = {res['iter']}, время = {res['time']:.3f} с")

f_str = "10 + x**2 - 10*cos(2*pi*x)" 
f = parse_function(f_str)
res = piyavskii(f, -5.12, 5.12, eps=0.01)
plot_result(f, -5.12, 5.12, res)

f_ackley = parse_function('-20*exp(-0.2*abs(x)) - exp(cos(2*pi*x)) + 20 + exp(1)')
res_ack = piyavskii(f_ackley, -5, 5, eps=0.01)
plot_result(f_ackley, -5, 5, res_ack)

f_sin = parse_function('sin(3*x) + 0.1*x**2')
res_sin = piyavskii(f_sin, 0, 10, eps=0.01)
plot_result(f_sin, 0, 10, res_sin)