import itertools

initial_portfolio = {
    "CB1": 100,
    "CB2": 800,
    "Dep": 400,
    "Cash": 600
}

min_assets = {
    "CB1": 30,
    "CB2": 150,
    "Dep": 100
}

commission = {
    "CB1": 0.04,
    "CB2": 0.07,
    "Dep": 0.05
}

steps = {
    "CB1": 25,
    "CB2": 200,
    "Dep": 100
}

scenarios = [
    {
        "probs": [0.6, 0.3, 0.1],
        "CB1": [1.2, 1.05, 0.8],
        "CB2": [1.1, 1.02, 0.95],
        "Dep": [1.07, 1.03, 1.0]
    },
    {
        "probs": [0.3, 0.2, 0.5],
        "CB1": [1.4, 1.05, 0.6],
        "CB2": [1.15, 1.0, 0.9],
        "Dep": [1.01, 1.0, 1.0]
    },
    {
        "probs": [0.4, 0.4, 0.2],
        "CB1": [1.15, 1.05, 0.7],
        "CB2": [1.12, 1.01, 0.94],
        "Dep": [1.05, 1.01, 1.0]
    }
]

def apply_scenario(portfolio, scenario):
    new_portfolio = {}
    for asset in ["CB1", "CB2", "Dep"]:
        new_portfolio[asset] = sum(
            p * portfolio[asset] * r
            for p, r in zip(scenario["probs"], scenario[asset])
        )
    new_portfolio["Cash"] = portfolio["Cash"]
    return new_portfolio

def adjust_portfolio(portfolio, adjustments):
    new_portfolio = portfolio.copy()
    cash = portfolio["Cash"]
    for asset, delta in adjustments.items():
        if delta > 0:
            cost = delta * (1 + commission[asset])
            if cost > cash:
                delta = cash / (1 + commission[asset])
                cost = delta * (1 + commission[asset])
            cash -= cost
            new_portfolio[asset] += delta
        elif delta < 0:
            delta = -delta
            if new_portfolio[asset] - delta < min_assets[asset]:
                delta = new_portfolio[asset] - min_assets[asset]
            cash += delta * (1 - commission[asset])
            new_portfolio[asset] -= delta
    new_portfolio["Cash"] = cash
    return new_portfolio

plan_steps = [
    {"CB1": 25, "CB2": -200, "Dep": 0},
    {"CB1": -100, "CB2": 0, "Dep": 0},
    {"CB1": 0, "CB2": 200, "Dep": -100}
]

portfolio = initial_portfolio.copy()
print("\n--- Начальный портфель ---")
for k, v in portfolio.items():
    print(f"{k}: {v:.2f}")
    

for t in range(3):
    portfolio = adjust_portfolio(portfolio, plan_steps[t])
    portfolio = apply_scenario(portfolio, scenarios[t])
    
    print(f"Этап {t+1}:")
    print("Портфель после применения плана и сценария:")
    for k, v in portfolio.items():
        print(f"  {k}: {v:.2f}")
    total = sum(portfolio.values())
    print(f"  Общая стоимость портфеля: {total:.2f}\n")

print("=== Итоговый портфель ===")
for k, v in portfolio.items():
    print(f"{k}: {v:.2f}")
total = sum(portfolio.values())
print(f"\nМаксимальный ожидаемый доход: {total:.2f}")
