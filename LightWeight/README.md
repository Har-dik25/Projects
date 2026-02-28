# 💡 LightWeight — Deterministic Text-to-SQL Engine

A **lightweight, offline Text-to-SQL converter** that transforms natural language queries into SQL — using pure Python with **no LLM or AI model** required.

---

## ✨ Features

- **Zero Dependencies on LLMs** — Runs entirely offline using regex and rule-based parsing
- **Natural Language Understanding:**
  - Synonym resolution (e.g., "staff" → "employees", "income" → "salary")
  - Operator parsing ("greater than", "less than", "between")
  - Aggregation detection (COUNT, AVG, MAX, MIN)
- **Condition Parsing:**
  - Numeric comparisons (`salary > 50000`)
  - Range queries (`age between 25 and 35`)
  - Text matching (`name like John`)
  - Department filtering
- **SQLite Integration** — Includes `sample.db` for testing

---

## ▶️ How to Run

```bash
python text_to_sql_app.py
```

### Example Queries
```
Enter: Show salary of employees older than 30
SQL:   SELECT salary FROM employees WHERE age > 30

Enter: Count staff in HR
SQL:   SELECT COUNT(*) FROM employees WHERE department = 'HR'

Enter: Average income of workers
SQL:   SELECT AVG(salary) FROM employees
```

No external dependencies required.

---

## 🏗️ Architecture

```
User Query → Normalize → Resolve Table → Resolve Columns → Parse Conditions → Build SQL
```

| Stage | Description |
|-------|-------------|
| Normalize | Lowercase, replace phrases ("greater than" → ">") |
| Resolve Table | Match table names or synonyms |
| Resolve Columns | Identify mentioned columns |
| Parse Conditions | Extract WHERE clause conditions |
| Build SQL | Assemble final SELECT statement |

---

## 🧠 Concepts Demonstrated
- Deterministic NLP (no model, no API)
- Text normalization and synonym resolution
- Regex-based parsing
- SQL query generation
