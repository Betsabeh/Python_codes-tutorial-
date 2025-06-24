import numpy as np

# تعریف داده‌های آموزشی
A_train = [
    [0,1,0, 1,0,1, 1,1,1],
    [0,1,0, 1,1,1, 1,0,1],
    [0,1,0, 1,0,1, 1,1,0],
    [0,1,0, 1,1,0, 1,1,1],
    [0,1,0, 0,1,1, 1,1,1],
    [1,1,0, 1,0,1, 1,1,1],
]

B_train = [
    [1,1,0, 1,0,1, 1,1,0],
    [1,1,1, 1,0,1, 1,1,0],
    [1,1,0, 1,0,1, 1,1,1],
    [1,1,1, 1,1,0, 1,1,0],
    [1,1,0, 1,1,1, 1,1,0],
    [1,1,1, 1,0,1, 1,0,1],
]

# تبدیل به آرایه numpy و برچسب‌ها
X_train = np.array(A_train + B_train)
d_train = np.array([0]*len(A_train) + [1]*len(B_train))

# تعریف داده‌های تست
A_test = np.array([0,1,0, 1,0,1, 1,0,1])  # کلاس 0
B_test = np.array([1,1,0, 1,1,1, 1,1,0])  # کلاس 1

X_test = np.array([A_test, B_test])
d_test = np.array([0, 1])

# پارامترها
w = np.zeros(X_train.shape[1])
b = 0
eta = 0.1
epochs = 50

def step(x):
    return 1 if x >= 0 else 0

# آموزش پرسپترون
for epoch in range(epochs):
    errors = 0
    for xi, di in zip(X_train, d_train):
        z = np.dot(w, xi) + b
        y = step(z)
        e = di - y
        if e != 0:
            errors += 1
            w += eta * e * xi
            b += eta * e
    print(f"Epoch {epoch+1} - errors: {errors}")
    if errors == 0:
        print("Training converged.")
        break

# تست مدل
print("\nTest Results:")
for xi, label in zip(X_test, d_test):
    z = np.dot(w, xi) + b
    pred = step(z)
    print(f"Input: {xi} -> Predicted: {pred}, Actual: {label}, {'Correct' if pred == label else 'Wrong'}")
