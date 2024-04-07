def ODCA(beta, B, p):
    num_EUs = len(B)
    num_MECs = len(p)
    f = [[0] * num_MECs for _ in range(num_EUs)]  # Resource purchase decisions for each EU
    for i in range(num_EUs):
        Sz = set()
        j = 0
        while j <= num_MECs - 1:
            if p[j] == 0:
                f[i][j] = 0
            else:
                if j in Sz:
                    f[i][j] = 0
                else:
                    Snew = set(range(num_MECs)) - Sz
                    S_prime = {k for k in Snew if p[k] == 0}
                    Mnew = len(Snew) - len(S_prime)
                    f[i][j] = (B[i] + sum(p[k] / beta[k] for k in Snew)) / (Mnew * p[j]) - beta[j] ** -1
                    if f[i][j] < 0:
                        f[i][j] = 0
                        Sz.add(j)
                        break
                    else:
                        j = j + 1
    return f

# Example usage:
beta = [0.5, 0.3, 0.2]  # Constants for each EU
# beta = [2, 1, 0.5]  # Constants for each EU
# beta = [0.25, 0.15, 0.1, 0.1]  # Constants for each EU
B = [10, 20, 30]  # Budget of each EU used for resource purchase
p = [2, 1, 1.5]  # Prices of MECs

result = ODCA(beta, B, p)
print(result)
