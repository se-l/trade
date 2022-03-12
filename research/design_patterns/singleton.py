class Singleton:
    def __new__(cls):
        if not hasattr(cls, '_instance'):
            print('Creating the object')
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


if __name__ == '__main__':
    print(Singleton())
    print(Singleton())
