def func():
    ls = [1,2,3]
    def sub(i):
        return ls[i]
    func1(sub, 1)

def func1(funct, i):
    print(funct(i))

if __name__ == "__main__":
    func()