from mysklearn.myrandomforestclassifier import MyRandomForestClassifier



def test_fit_and_predict():
    X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    y = ["yes", "no", "yes", "no", "yes"]
    clf = MyRandomForestClassifier(N=5, M=3, F=1)
    clf.fit(X, y)
    predictions = clf.predict([[3, 4], [9, 10]])
    assert len(predictions) == 2
    assert predictions == [ "no", "yes"]


def test_random_forest_easy():
    X_train = [
        [1, "red"],
        [2, "red"],
        [3, "blue"],
        [4, "blue"]
    ]
    y_train = ["A", "A", "B", "B"]
    X_test = [[2, "red"], [3, "blue"]]
    y_test = ["A", "B"]

    clf = MyRandomForestClassifier(N=5, M=3, F=1)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    assert predictions == y_test, f"Expected {y_test}, but got {predictions}"

def test_random_forest_with_noise():
    X_train = [
        [1, "red"], [2, "red"], [3, "blue"], [4, "blue"],
        [5, "red"], [6, "blue"], [7, "red"], [8, "blue"]
    ]
    y_train = ["A", "A", "B", "B", "A", "B", "A", "B"]
    X_test = [[2, "red"], [7, "red"], [4, "blue"], [5, "blue"]]
    y_test = ["A", "A", "B", "B"]

    clf = MyRandomForestClassifier(N=10, M=5, F=1)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    assert predictions == y_test, f"Expected {y_test}, but got {predictions}"

def test_random_forest_with_redundant_features():
    X_train = [
        [1, "red", 99], [2, "red", 100], [3, "blue", 101], [4, "blue", 102],
        [5, "red", 103], [6, "blue", 104], [7, "red", 105], [8, "blue", 106]
    ]
    y_train = ["A", "A", "B", "B", "A", "B", "A", "B"]
    X_test = [[2, "red", 100], [7, "red", 105], [4, "blue", 102], [5, "blue", 103]]
    y_test = ["A", "A", "B", "B"]

    clf = MyRandomForestClassifier(N=20, M=7, F=2)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    assert predictions == y_test, f"Expected {y_test}, but got {predictions}"

def test_random_forest_imbalanced():
    X_train = [
        [1, "red"], [2, "red"], [3, "blue"], [4, "blue"],
        [5, "red"], [6, "red"], [7, "red"], [8, "red"]
    ]
    y_train = ["A", "A", "B", "B", "A", "A", "A", "A"]
    X_test = [[3, "blue"], [4, "blue"], [5, "red"], [6, "red"]]
    y_test = ["B", "B", "A", "A"]

    clf = MyRandomForestClassifier(N=50, M=15, F=1)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    assert predictions == y_test, f"Expected {y_test}, but got {predictions}"

def test_random_forest_multiclass():
    X_train = [
        [1, "red", 2.5], [2, "red", 3.6], [3, "blue", 1.5], [4, "blue", 3.2],
        [5, "green", 2.1], [6, "green", 1.8], [7, "red", 3.7], [8, "blue", 2.9]
    ]
    y_train = ["A", "A", "B", "B", "C", "C", "A", "B"]
    X_test = [[2, "red", 3.6], [3, "blue", 1.5], [5, "green", 2.1], [6, "green", 1.8]]
    y_test = ["A", "B", "C", "C"]

    clf = MyRandomForestClassifier(N=100, M=20, F=2)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    assert predictions == y_test, f"Expected {y_test}, but got {predictions}"
