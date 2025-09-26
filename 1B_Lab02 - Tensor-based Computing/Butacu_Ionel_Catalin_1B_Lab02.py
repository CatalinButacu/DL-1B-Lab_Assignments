import torch

def run_problem1():
    print("\nTask 1.1")
    tensor = torch.rand((2,2)) * 5
    print(tensor)

    print("\nTask 1.2")
    tensor = torch.eye(5)
    print(tensor)

    print("\nTask 1.3")
    tensor = torch.arange(0, 10.1, 0.5)
    print(tensor)

    print("\nTask 1.4")
    tensor = torch.arange(1, 17).reshape(4, 4)
    print(tensor)

    print(f"The first row is: \n{tensor[0, :]}")
    print(f"The first column is: \n{tensor[:, 0]}")
    print(f"The submatrix is: \n{tensor[1:3, 2:4]}")
    tensor.diagonal().zero_()
    print(f"The diagonal modified to zero: \n{tensor}")
    print(f"The reversed rows are: \n{torch.flip(tensor, dims=[0])}")


    print("\nTask 1.5")
    tensor = torch.arange(24).reshape(2, 3, 4)
    print(tensor)
    print(f"The reshaped tensor is: \n{tensor.reshape(4, 6)}")
    print(f"The flattened tensor is: \n{tensor.flatten()}")
    print(f"The transposed tensor is: \n{tensor.transpose(0, -1)}")


    print("\nTask 1.6")
    tensor = torch.arange(25).reshape(5, 5).float()
    print(tensor)
    print(f"The sum of elements is: {torch.sum(tensor)}")
    print(f"The mean of rows is: {torch.mean(tensor, 1, True)}")
    print(f"The max of columns is: {torch.max(tensor, 0)}")
    print(f"The min indices of rows are: {torch.argmin(tensor, 1)}")


    print("\nTask 1.7")
    tensor1 = torch.randn(4, 1)
    tensor2 = torch.randn(1, 5)
    print(f"The sum of the two tensors is: \n{tensor1 + tensor2}")


    print("\nTask 1.8")
    tensor = torch.normal(0, 1, (5, 5))
    print(tensor)


def run_problem2():
    print("\nTask 2")

    A = torch.rand(5, 5)
    print(f"Initial matrix A: \n{A}")

    B = A[[1, 3], :]
    print(f"Matrix B extracted froom matrix A: \n{B}")

    V = torch.tensor([[1], [-1], [2], [-2], [3]])
    print(f"Vector V: \n{V}")

    C = torch.mm(A, B.T)
    print(f"Matrix C: \n{C}")

    C = (C - C.mean()) / C.std()
    print(f"Normalized matrix C: \n{C}")

    C = C + V
    print(f"Matrix C after adding V: \n{C}")

    A = A @ A.T
    print(f"Symmetrical matrix A: \n{A}")

    A = A + 5 * torch.eye(5)
    print(f"Matrix A after adding 5 to its main diagonal: \n{A}")

    x = torch.linalg.solve(A, V.float())
    print(f"Solving the system of equations Ax = V: \n{x}")

def run_problem3(N=5):
    print("\nTask 3")

    A = torch.rand(N, N)
    print(f"Initial matrix A: \n{A}")

    print("\nTask 3.1")
    exp_A = torch.exp(A)
    row_sums = exp_A.sum(dim=1, keepdim=True)
    A_prime = exp_A / row_sums
    print(f"Transformed matrix A': \n{A_prime}")

    print("\nTask 3.2")
    W = torch.mm(A, A.T)
    print(f"Matrix W: \n{W}")

    D = torch.diag(W.sum(dim=1))
    print(f"Matrix D: \n{D}")

    L = D - W
    print(f"Matrix L: \n{L}")


    print("\nTask 3.3")
    I = torch.eye(N, dtype=torch.float64)
    b = torch.randn(N, dtype=torch.float64)
    lambda_param = 0.1
    L_plus_lambda_I = L + lambda_param * I
    x = torch.linalg.solve(L_plus_lambda_I, b)
    print(f"Solving the system of equations (L + lambda * I)x = b: \n{x}")


    print("\nTask 3.4")
    x = torch.randn(N, dtype=torch.float64)
    y = torch.tanh(L @ x.float())    
    print(f"Approximate solution of the system of equations Lx = 0: \n{y}")
    

def main():
    run_problem1()
    run_problem2()
    run_problem3()    

if __name__ == "__main__":
    main()