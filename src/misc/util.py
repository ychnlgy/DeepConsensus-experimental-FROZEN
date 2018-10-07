import sys

# === Constants ===

UNITTEST = "unittest_"

# === Functions ===

def hardmap(fn, *args):
    return tuple(map(fn, args))

def unittest(fname, *funcs):
    
    print("Running tests for %s..." % fname)
    
    for f in funcs:
        fn = f.__name__
        assert fn.startswith(UNITTEST)
        fn = fn[len(UNITTEST):]
        sys.stdout.write("  >> %s..." % fn)
        
        f()
        sys.stdout.write("OK\n")
    
    print("All %d tests passed." % len(funcs))

def Struct(*keys):

    class StructClass:
        def __init__(self, *args, **kwargs):
            self.__dict__.update(zip(keys, args))
            self.__dict__.update(kwargs)
    
    return StructClass

def do_nothing(*args, **kwargs):
    return

# === Tests ===

def unittest_Struct():
    Apple = Struct("calories", "color")
    a1 = Apple(120, "red")
    assert a1.calories == 120
    assert a1.color == "red"

if __name__ == "__main__":
    unittest(__file__,
        unittest_Struct
    )
