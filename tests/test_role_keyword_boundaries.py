import pandas as pd

from framevitals.column_roles import infer_column_roles


def test_role_keywords_use_token_boundaries():
    frame = pd.DataFrame({
        "paid_amount": [10.0, 20.0, 30.0, 40.0],
        "average_score": [0.1, 0.2, 0.3, 0.4],
        "customer_id": [101, 102, 103, 104],
        "customer_age": [20, 21, 22, 23],
    })

    roles = infer_column_roles(frame)

    assert "id_like" not in roles["paid_amount"]["roles"]
    assert "price_like" in roles["paid_amount"]["roles"]
    assert "sensitive" not in roles["average_score"]["roles"]
    assert "id_like" in roles["customer_id"]["roles"]
    assert "sensitive" in roles["customer_age"]["roles"]
