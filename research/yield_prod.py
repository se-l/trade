def producer():
    for i in range(10):
        yield i


if __name__ == '__main__':
    for msg in producer():
        print(msg)
