# ----------------------------------------------------------------- initialisation
from randomwalk import GraphRecommender
model = GraphRecommender.load("model.pkl")

# ----------------------------------------------------------------- data loading
""" 
Your data looks like this (one same user, timestamp in ascending order)
note that these are the minimum columns required (Userid too so it can populate back the pkl file)
rating parent_asin                   user_id  timestamp  price  average_rating  rating_number           categories
0     188701         5      B09LR4L158  AHZZGB4QNYBKTBIBVT7SG46RNK3Q     1.654140e+12    0.0            4.6            644  Science Fiction & Fantasy
1     188702         5      B09TQ6DR39  AHZZGB4QNYBKTBIBVT7SG46RNK3Q     1.654710e+12    0.0            4.8           1316      Teen & Young Adult
2     188703         5      B09L9G8MCV  AHZZGB4QNYBKTBIBVT7SG46RNK3Q     1.655150e+12    0.0            4.5            829    Literature & Fiction
"""
df = pd.DataFrame({
    "rating": [
        188701,
        188702,
        188703
    ],
    "parent_asin": [
        "B09LR4L158",
        "B09TQ6DR39",
        "B09L9G8MCV"
    ],
    "user_id": [
        "AHZZGB4QNYBKTBIBVT7SG46RNK3Q",
        "AHZZGB4QNYBKTBIBVT7SG46RNK3Q",
        "AHZZGB4QNYBKTBIBVT7SG46RNK3Q"
    ],
    "timestamp": [
        1.654140e12,
        1.654710e12,
        1.655150e12
    ],
    "price": [
        0.0,
        0.0,
        0.0
    ],
    "average_rating": [
        4.6,
        4.8,
        4.5
    ],
    "rating_number": [
        644,
        1316,
        829
    ],
    "categories": [
        "Science Fiction & Fantasy",
        "Teen & Young Adult",
        "Literature & Fiction"
    ]
})

# ----------------------------------------------------------------- prediction
recommendations = model.recommend(new_user_df, policy='breadth20', store=False)
print(recommendations)
#e.g. ['B07G6RZYG8', 'B00NFW2OOA', 'B00RAKQF9I', 'B01990X3WS', 'B00HZ5K8AU']

#other policies include: 
"""
'one' – depth-1 from current node
'breadth20' – breadth-first search up to 20 unique neighbours
'breadth10_depth2' – BFS for 10 items, each explored to depth 2
'breadth5_depth4' – BFS for 5 items, each explored to depth 4
'depth20' – depth-first walk of length 20
"""

# ----------------------------------------------------------------- save results
"""
model.save("model.pkl")
"""


# ----------------------------------------------------------------- optional additional items to run
"""
model.evaluate_metrics(df)
model.visualise(num_nodes=30)
model.fit(df) #this resets the model and does the ml model again, the df formatting is the same
model.save("model.pkl")
"""