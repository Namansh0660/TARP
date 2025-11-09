# Complete README

This README contains **all Python code** for:
- Linear Regression
- Logistic Regression
- Decision Tree

And **all Neo4j (Cypher + GDS)** code for:
- BFS
- DFS
- Minimum Spanning Tree
- Triangle Count
- Clustering Coefficient
- Louvain Modularity
- Betweenness Centrality
- Closeness Centrality
- PageRank

Using **your own synthetic datasets**.

---

# 1. Python Machine Learning
Requires:
```
pip install scikit-learn numpy pandas
```

## 1.1 Linear Regression
Create file: `linear_regression_demo.py`
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

rng = np.random.RandomState(42)
X = rng.rand(200, 3)
y = 4.0*X[:,0] - 2.0*X[:,1] + 0.5*X[:,2] + rng.normal(0, 0.2, size=200)

model = LinearRegression().fit(X, y)
pred = model.predict(X)

print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("R2:", r2_score(y, pred))
print("RMSE:", mean_squared_error(y, pred, squared=False))
```

## 1.2 Logistic Regression
Create file: `logistic_regression_demo.py`
```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X, y = make_classification(n_samples=400, n_features=5,
                           n_informative=3, n_redundant=0,
                           random_state=7)

clf = LogisticRegression(max_iter=1000).fit(X, y)
pred = clf.predict(X)

print("Accuracy:", accuracy_score(y, pred))
print(classification_report(y, pred))
```

## 1.3 Decision Tree
Create file: `decision_tree_demo.py`
```python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)

tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
pred = tree.predict(X)

print("Accuracy:", accuracy_score(y, pred))
print(export_text(tree, feature_names=["sepal_len", "sepal_wid", "petal_len", "petal_wid"]))
```

Run:
```
python file.py
```

---

# 2. Neo4j + GDS Graph Algorithms
Requires Neo4j with **GDS 2.x plugin**.

Paste all commands in **Neo4j Browser**.

---

# 2.1 Create Synthetic Graph
```cypher
MATCH (n) DETACH DELETE n;

CREATE (a:Person {name:'A'}),
       (b:Person {name:'B'}),
       (c:Person {name:'C'}),
       (d:Person {name:'D'}),
       (e:Person {name:'E'}),
       (f:Person {name:'F'});

CREATE (a)-[:KNOWS {weight:1.0}]->(b), (b)-[:KNOWS {weight:1.5}]->(a),
       (a)-[:KNOWS {weight:0.8}]->(c), (c)-[:KNOWS {weight:0.8}]->(a),
       (b)-[:KNOWS {weight:2.0}]->(d), (d)-[:KNOWS {weight:2.0}]->(b),
       (c)-[:KNOWS {weight:1.2}]->(d), (d)-[:KNOWS {weight:1.2}]->(c),
       (c)-[:KNOWS {weight:1.1}]->(e), (e)-[:KNOWS {weight:1.1}]->(c),
       (d)-[:KNOWS {weight:0.9}]->(e), (e)-[:KNOWS {weight:0.9}]->(d),
       (e)-[:KNOWS {weight:0.5}]->(f), (f)-[:KNOWS {weight:0.5}]->(e);
```

---

# 2.2 Project In-Memory Graph
```cypher
CALL gds.graph.drop('demo', false) YIELD graphName RETURN graphName;

MATCH (source:Person)
OPTIONAL MATCH (source)-[r:KNOWS]->(target:Person)
RETURN gds.graph.project(
  'demo',
  source,
  target,
  { relationshipProperties: r { .weight } },
  { undirectedRelationshipTypes: ['KNOWS'] }
);
```

---

# 2.3 BFS
```cypher
MATCH (src:Person {name:'A'})
CALL gds.bfs.stream('demo', { sourceNode: src })
YIELD path
RETURN path;
```

# 2.4 DFS
```cypher
MATCH (src:Person {name:'A'})
CALL gds.dfs.stream('demo', { sourceNode: src })
YIELD path
RETURN path;
```

# 2.5 Minimum Spanning Tree (MST)
```cypher
MATCH (src:Person {name:'A'})
CALL gds.spanningTree.stream('demo', {
  sourceNode: src,
  relationshipWeightProperty: 'weight',
  objective: 'minimum'
})
YIELD nodeId, parentId, weight
RETURN gds.util.asNode(parentId).name AS parent,
       gds.util.asNode(nodeId).name   AS node,
       weight
ORDER BY parent, node;
```

---

# 2.6 Triangle Count
```cypher
CALL gds.triangleCount.stream('demo')
YIELD nodeId, triangleCount
RETURN gds.util.asNode(nodeId).name AS name, triangleCount
ORDER BY triangleCount DESC, name;
```

# 2.7 Local Clustering Coefficient
```cypher
CALL gds.localClusteringCoefficient.stream('demo')
YIELD nodeId, localClusteringCoefficient
RETURN gds.util.asNode(nodeId).name AS name, localClusteringCoefficient
ORDER BY localClusteringCoefficient DESC, name;
```

---

# 2.8 Louvain Modularity
### Node communities
```cypher
CALL gds.louvain.stream('demo', { relationshipWeightProperty: 'weight' })
YIELD nodeId, communityId
RETURN gds.util.asNode(nodeId).name AS name, communityId
ORDER BY communityId, name;
```

### Global modularity
```cypher
CALL gds.louvain.stats('demo', { relationshipWeightProperty: 'weight' })
YIELD communityCount, modularity
RETURN communityCount, modularity;
```

---

# 2.9 Betweenness Centrality
```cypher
CALL gds.betweenness.stream('demo')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC, name;
```

# 2.10 Closeness Centrality
```cypher
CALL gds.closeness.stream('demo')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC, name;
```

# 2.11 PageRank
```cypher
CALL gds.pageRank.stream('demo', { maxIterations: 20, dampingFactor: 0.85 })
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC, name;
```

---

# 2.12 Drop Graph
```cypher
CALL gds.graph.drop('demo');
```

---

# End of README

