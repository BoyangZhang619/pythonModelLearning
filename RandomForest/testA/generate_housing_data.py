import os
import numpy as np
import pandas as pd

def generate_housing_data(n=3000, random_state=42):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_DIR = os.path.join(BASE_DIR, "csv")
    rng = np.random.default_rng(random_state)

    # 基础特征
    area = rng.normal(90, 20, n).clip(40, 200)
    bedrooms = (area / 30 + rng.normal(0, 0.5, n)).round().clip(1, 6)
    bathrooms = (bedrooms / 1.5 + rng.normal(0, 0.3, n)).round().clip(1, 4)
    year_built = rng.integers(1960, 2022, n)
    renovated = rng.choice([0, 1], p=[0.7, 0.3], size=n)
    distance_to_metro = rng.gamma(2, 0.8, n).clip(0, 10)
    community_score = rng.normal(7, 1.2, n).clip(3, 10)

    # 房价
    price = (
        area * rng.normal(1.0, 0.05, n) +
        bedrooms * 10 +
        bathrooms * 8 +
        (2024 - year_built) * -0.4 +
        renovated * 30 +
        (10 - community_score) * -4 +
        distance_to_metro * -6 +
        rng.normal(0, 20, n)
    )
    price = price.round().clip(50, 1000)

    # days_on_market：减少噪声、强化结构
    days_on_market = (
        50
        + (price - price.mean()) * 0.10
        + distance_to_metro * 4
        - community_score * 5
        - renovated * 8
        + rng.normal(0, 15, n)   # 噪声减半
    ).clip(3, 180)

    # logistic 转成概率（更平滑、机器学习更容易）
    prob_sell_fast = 1 / (1 + np.exp((days_on_market - 40) / 10))
    sold_fast = (rng.random(n) < prob_sell_fast).astype(int)

    df = pd.DataFrame({
        "price": price,
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "year_built": year_built,
        "renovated": renovated,
        "distance_to_metro": distance_to_metro,
        "community_score": community_score,
        "days_on_market": days_on_market,
        "sold_fast": sold_fast,
        "prob": prob_sell_fast
    })
    df.to_csv(os.path.join(CSV_DIR,"./generated_housing_data.csv"))
    return df

if __name__ == "__main__":
    df = generate_housing_data()