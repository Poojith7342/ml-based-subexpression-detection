import re
from sklearn.tree import DecisionTreeClassifier

# --------------------------------------------------
# 1. TRAIN A SMALL ML MODEL
# Features:
# [is_repeated, occurrence_count, distance,
#  operand_modified, same_basic_block]
# --------------------------------------------------

training_data = [
    [0, 1, 0, 0, 1],
    [1, 2, 1, 0, 1],
    [1, 3, 2, 0, 1],
    [1, 2, 4, 0, 1],
    [1, 2, 1, 1, 1],
    [1, 3, 5, 1, 1],
    [0, 1, 0, 0, 0],
    [1, 2, 2, 0, 0],
    [1, 2, 1, 1, 0],
    [1, 4, 3, 0, 1],
]

labels = [
    0, 1, 1, 1, 0, 0, 0, 0, 0, 1
]

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(training_data, labels)


# --------------------------------------------------
# 2. FEATURE EXTRACTION
# --------------------------------------------------

def extract_expression(line):
    pattern = r"(\w+)\s*=\s*(\w+)\s*([+\-*/])\s*(\w+)"
    match = re.fullmatch(pattern, line.strip())

    if match:
        target, left, operator, right = match.groups()
        return target, left, operator, right

    return None


# --------------------------------------------------
# 3. COMMON SUBEXPRESSION DETECTION AND OPTIMIZATION
# --------------------------------------------------

def optimize_code(code):
    optimized = []

    # Valid expressions currently available for reuse
    cache = {}

    # History used for ML feature extraction
    history = {}

    eliminated = 0

    for line_number, original_line in enumerate(code):
        line = original_line.strip()

        if not line:
            continue

        expression = extract_expression(line)

        # Handle binary arithmetic expressions
        if expression:
            target, left, operator, right = expression
            key = (left, operator, right)

            # Also recognize reversed operands for + and *
            reverse_key = (right, operator, left)

            repeated_before = key in history or (
                operator in ("+", "*") and reverse_key in history
            )

            previous = history.get(key)

            if previous is None and operator in ("+", "*"):
                previous = history.get(reverse_key)

            occurrence_count = (
                previous["count"] + 1 if previous else 1
            )

            distance = (
                line_number - previous["line"] if previous else 0
            )

            # If an expression occurred before but is no longer
            # in the valid cache, it may have been invalidated.
            cached_key = key
            if cached_key not in cache and operator in ("+", "*"):
                if reverse_key in cache:
                    cached_key = reverse_key

            operand_modified = (
                repeated_before and cached_key not in cache
            )

            features = [[
                int(repeated_before),
                occurrence_count,
                distance,
                int(operand_modified),
                1
            ]]

            prediction = int(model.predict(features)[0])

            # ML proposes a candidate; compiler checks decide safety.
            if (
                prediction == 1
                and cached_key in cache
                and not operand_modified
            ):
                previous_temp = cache[cached_key]
                line = f"{target} = {previous_temp}"
                eliminated += 1
            else:
                # Store the expression if it is not already cached.
                if key not in cache:
                    cache[key] = target

            history[key] = {
                "count": occurrence_count,
                "line": line_number
            }

            # An assignment changes the target variable.
            # Invalidate cached expressions that use that variable.
            cache = {
                expr: temp
                for expr, temp in cache.items()
                if target not in (expr[0], expr[2])
            }

            # Keep the newly computed expression available,
            # unless its target was overwritten as part of the RHS.
            if line == original_line.strip() and key not in cache:
                cache[key] = target

        else:
            # Handle simple assignments such as a = 10.
            assignment = re.fullmatch(
                r"(\w+)\s*=\s*(.+)", line
            )

            if assignment:
                target, _ = assignment.groups()

                cache = {
                    expr: temp
                    for expr, temp in cache.items()
                    if target not in (expr[0], expr[2])
                }

        optimized.append(line)

    return optimized, eliminated


# --------------------------------------------------
# 4. INPUT INTERMEDIATE CODE (THREE-ADDRESS CODE)
# --------------------------------------------------

original_code = [
    "t1 = a * b",
    "t2 = t1 + c",
    "x = t2",
    "t3 = a * b",
    "t4 = t3 + d",
    "y = t4"
]


# --------------------------------------------------
# 5. DISPLAY RESULTS
# --------------------------------------------------

optimized_code, eliminated = optimize_code(original_code)

print("ORIGINAL INTERMEDIATE CODE")
print("--------------------------")
for line in original_code:
    print(line)

print("\nOPTIMIZED INTERMEDIATE CODE")
print("---------------------------")
for line in optimized_code:
    print(line)

print("\nRESULTS")
print("-------")
print("Original instructions:", len(original_code))
print("Optimized instructions:", len(optimized_code))
print("Redundant computations eliminated:", eliminated)