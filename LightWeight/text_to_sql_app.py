import re


# =========================================================
# SCHEMA CONFIGURATION (replace with real schema in product)
# =========================================================
SCHEMA = {
    "employees": ["id", "name", "age", "salary", "department"]
}


# =========================================================
# SYNONYMS & KEYWORDS
# =========================================================
TABLE_SYNONYMS = {
    "staff": "employees",
    "workers": "employees",
    "people": "employees"
}

COLUMN_SYNONYMS = {
    "income": "salary",
    "pay": "salary",
    "dept": "department"
}

AGGREGATIONS = {
    "count": "COUNT",
    "average": "AVG",
    "avg": "AVG",
    "max": "MAX",
    "min": "MIN"
}


# =========================================================
# TEXT NORMALIZATION
# =========================================================
def normalize(text: str) -> str:
    text = text.lower()

    replacements = {
        "less than": "<",
        "greater than": ">",
        "more than": ">",
        "older than": ">",
        "younger than": "<",
        "equal to": "=",
        "equals": "=",
        "is": "="
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text


# =========================================================
# ENTITY RESOLUTION
# =========================================================
def resolve_table(text: str) -> str:
    for word, table in TABLE_SYNONYMS.items():
        if word in text:
            return table

    for table in SCHEMA:
        if table in text:
            return table

    return list(SCHEMA.keys())[0]


def resolve_columns(text: str, table: str):
    resolved = []

    for word, col in COLUMN_SYNONYMS.items():
        if word in text:
            resolved.append(col)

    for col in SCHEMA[table]:
        if col in text:
            resolved.append(col)

    return list(set(resolved))


# =========================================================
# CONDITION PARSERS
# =========================================================
def parse_between(text: str):
    match = re.search(r"(age|salary)\s+between\s+(\d+)\s+and\s+(\d+)", text)
    if match:
        col, a, b = match.groups()
        return f"{col} BETWEEN {a} AND {b}"
    return None


def parse_numeric(text: str):
    match = re.search(r"(age|salary)\s*(>|<|=)\s*(\d+)", text)
    if match:
        col, op, val = match.groups()
        return f"{col} {op} {val}"
    return None


def parse_text_match(text: str):
    match = re.search(r"name\s+like\s+(\w+)", text)
    if match:
        return f"name LIKE '%{match.group(1)}%'"
    return None


def parse_department(text: str):
    match = re.search(r"(hr|it|finance)", text)
    if match:
        return f"department = '{match.group(1).upper()}'"
    return None


def parse_conditions(text: str):
    conditions = []

    for parser in [parse_between, parse_numeric, parse_department, parse_text_match]:
        c = parser(text)
        if c:
            conditions.append(c)

    if not conditions:
        return None

    return " AND ".join(conditions)


# =========================================================
# SQL GENERATOR CORE
# =========================================================
def build_sql(natural_text: str) -> str:
    text = normalize(natural_text)

    table = resolve_table(text)
    columns = resolve_columns(text, table)
    condition = parse_conditions(text)

    # -------- aggregation detection --------
    for word, func in AGGREGATIONS.items():
        if word in text:
            col = columns[0] if columns else "*"

            if func == "COUNT":
                sql = f"SELECT COUNT(*) FROM {table}"
            else:
                if col == "*":
                    col = "salary"
                sql = f"SELECT {func}({col}) FROM {table}"

            if condition:
                sql += f" WHERE {condition}"

            return sql

    # -------- normal select --------
    column_str = ", ".join(columns) if columns else "*"
    sql = f"SELECT {column_str} FROM {table}"

    if condition:
        sql += f" WHERE {condition}"

    return sql


# =========================================================
# SIMPLE CLI (prints only SQL, no execution)
# =========================================================
def main():
    print("\nDeterministic Text → SQL Engine (Offline)")
    print("Type 'exit' to quit.\n")

    while True:
        user_input = input("Enter natural language query: ")

        if user_input.lower() == "exit":
            break

        sql = build_sql(user_input)

        print("\nGenerated SQL:")
        print(sql)
        print("\n" + "-" * 50 + "\n")


# =========================================================
if __name__ == "__main__":
    main()
