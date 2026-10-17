# [MODULE DOC]
# Test for areal.utils.wrapper: methods marked with wrapable() on a target object are
# exposed on a wrapper object (under their name or alias) and routed through a call
# interceptor that receives method metadata in kwargs.
from areal.utils.wrapper import (
    wrap,
    wrap_get_method,
    wrap_get_method_name,
    wrap_remove_meta,
    wrapable,
)


class Calculator:
    # [CLASS DOC]
    # HELPER: Target object with two wrapable methods using a constant offset/scale of 2.
    def __init__(self):
        # [FUNCTION DOC]
        # Sets remaining = 2.
        self.remaining = 2

    @wrapable()
    def add(self, a, b):
        # [FUNCTION DOC]
        # Returns a + b + remaining.
        return a + b + self.remaining

    @wrapable(name="multiply")
    def mul(self, x, y):
        # [FUNCTION DOC]
        # Exposed as "multiply"; returns x * y * remaining.
        return x * y * self.remaining


class EnhancedCalculator:
    # [CLASS DOC]
    # HELPER: Wrapper that mirrors Calculator's wrapable methods via wrap().
    def __init__(self, obj: Calculator):
        # [FUNCTION DOC]
        # Stores prefix and target and calls wrap() to install interceptor-routed methods on
        # self. The prefix attribute is never used.
        super().__init__()
        self.prefix = "Result:"
        self.calculator = obj
        wrap(self, obj, self._wrap_call)

    def _wrap_call(self, *args, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Interceptor: extracts method name and bound method from kwargs metadata,
        # strips the metadata, prints both (stdout side effect) and invokes the method.
        method_name = wrap_get_method_name(kwargs)
        method = wrap_get_method(kwargs)
        kwargs = wrap_remove_meta(kwargs)
        print("wrap method: ", method_name)
        print("wrap method: ", method)
        return method(*args, **kwargs)


def test_wrapper():
    # [FUNCTION DOC]
    # Asserts wrapped add(2,3) == 7 and multiply(4,5) == 40.
    calc = Calculator()
    enhancer = EnhancedCalculator(calc)

    assert enhancer.add(2, 3) == 7
    assert enhancer.multiply(4, 5) == 40


if __name__ == "__main__":
    test_wrapper()
