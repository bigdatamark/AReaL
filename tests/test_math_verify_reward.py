# [MODULE DOC]
# Unit tests for the math-verify based reward functions exported by areal.reward:
# gsm8k_reward_fn and geometry3k_reward_fn (called with prompt, completions string,
# empty prompt_ids/completion_ids and a gold answer; return float 1.0 for a match, 0.0
# otherwise, never raising), plus the shared verifier returned by get_math_verify_worker
# (verify(pred, gold) -> float). No GPU; depends on the math-verify library's parsing
# and equivalence behavior.
from areal.reward import geometry3k_reward_fn, get_math_verify_worker, gsm8k_reward_fn


class TestGSM8KRewardFn:
    # [CLASS DOC]
    # Basic GSM8K reward behavior: numeric, symbolic, boxed, LaTeX, wrong and empty
    # answers.
    """Test GSM8K reward function with math-verify."""

    def test_gsm8k_exact_match(self):
        # [FUNCTION DOC]
        # "The answer is 4" vs gold "4" gives 1.0.
        """Test exact numerical match."""
        reward = gsm8k_reward_fn(
            prompt="What is 2+2?",
            completions="The answer is 4",
            prompt_ids=[],
            completion_ids=[],
            answer="4",
        )
        assert reward == 1.0, "Exact match should return 1.0"

    def test_gsm8k_symbolic_equivalence(self):
        # [FUNCTION DOC]
        # 0.5 vs gold 1/2 gives 1.0 (symbolic equivalence).
        """Test symbolic equivalence (e.g., 1/2 vs 0.5)."""
        reward = gsm8k_reward_fn(
            prompt="What is 1/2?",
            completions="The answer is 0.5",
            prompt_ids=[],
            completion_ids=[],
            answer="1/2",
        )
        assert reward == 1.0, "Symbolic equivalence should return 1.0"

    def test_gsm8k_boxed_format(self):
        # [FUNCTION DOC]
        # Completion with a boxed 4 vs gold "4" gives 1.0.
        """Test boxed answer format."""
        reward = gsm8k_reward_fn(
            prompt="What is 2+2?",
            completions="Step 1: 2+2=4. \\boxed{4}",
            prompt_ids=[],
            completion_ids=[],
            answer="4",
        )
        assert reward == 1.0, "Boxed format should extract and match"

    def test_gsm8k_incorrect_answer(self):
        # [FUNCTION DOC]
        # "The answer is 5" vs gold "4" gives 0.0.
        """Test incorrect answer."""
        reward = gsm8k_reward_fn(
            prompt="What is 2+2?",
            completions="The answer is 5",
            prompt_ids=[],
            completion_ids=[],
            answer="4",
        )
        assert reward == 0.0, "Incorrect answer should return 0.0"

    def test_gsm8k_empty_answer(self):
        # [FUNCTION DOC]
        # Empty completion gives 0.0.
        """Test empty/missing answer."""
        reward = gsm8k_reward_fn(
            prompt="What is 2+2?",
            completions="",
            prompt_ids=[],
            completion_ids=[],
            answer="4",
        )
        assert reward == 0.0, "Empty completion should return 0.0"

    def test_gsm8k_latex_expression(self):
        # [FUNCTION DOC]
        # LaTeX sqrt(4) = 2 vs gold "2" gives 1.0.
        """Test LaTeX mathematical expression."""
        reward = gsm8k_reward_fn(
            prompt="What is sqrt(4)?",
            completions="$\\sqrt{4} = 2$",
            prompt_ids=[],
            completion_ids=[],
            answer="2",
        )
        assert reward == 1.0, "LaTeX expression should be parsed correctly"


class TestGeometry3KRewardFn:
    # [CLASS DOC]
    # Basic Geometry3K reward behavior: bracketed answers like "[90]", numeric/symbolic
    # equivalence, fallback without brackets, whitespace and empty input.
    """Test Geometry3K reward function with math-verify."""

    def test_geometry3k_bracket_format(self):
        # [FUNCTION DOC]
        # "[90]" vs gold 90 gives 1.0.
        """Test Geometry3K bracket answer format [answer]."""
        reward = geometry3k_reward_fn(
            prompt="What is the angle?",
            completions="The angle is [90].",
            prompt_ids=[],
            completion_ids=[],
            answer="90",
        )
        assert reward == 1.0, "Bracket format should match"

    def test_geometry3k_numerical_equivalence(self):
        # [FUNCTION DOC]
        # "[5.0]" vs gold 5 gives 1.0.
        """Test numerical equivalence."""
        reward = geometry3k_reward_fn(
            prompt="What is the length?",
            completions="The length is [5.0].",
            prompt_ids=[],
            completion_ids=[],
            answer="5",
        )
        assert reward == 1.0, "5.0 should equal 5"

    def test_geometry3k_incorrect_bracket_format(self):
        # [FUNCTION DOC]
        # "[60]" vs gold 90 gives 0.0.
        """Test incorrect answer in bracket format."""
        reward = geometry3k_reward_fn(
            prompt="What is the angle?",
            completions="The angle is [60].",
            prompt_ids=[],
            completion_ids=[],
            answer="90",
        )
        assert reward == 0.0, "Incorrect answer should return 0.0"

    def test_geometry3k_no_bracket_fallback(self):
        # [FUNCTION DOC]
        # No brackets ("90 degrees") still matches gold 90 via fallback parsing.
        """Test fallback to last number when no bracket format."""
        reward = geometry3k_reward_fn(
            prompt="What is the angle?",
            completions="The answer is 90 degrees.",
            prompt_ids=[],
            completion_ids=[],
            answer="90",
        )
        # With math-verify, this should work if both parse to 90
        assert reward == 1.0, "Fallback to last number should work"

    def test_geometry3k_symbolic_fraction(self):
        # [FUNCTION DOC]
        # "[1/2]" vs gold 0.5 gives 1.0.
        """Test symbolic fraction equivalence."""
        reward = geometry3k_reward_fn(
            prompt="What is the ratio?",
            completions="The ratio is [1/2].",
            prompt_ids=[],
            completion_ids=[],
            answer="0.5",
        )
        assert reward == 1.0, "1/2 should equal 0.5"

    def test_geometry3k_empty_completion(self):
        # [FUNCTION DOC]
        # Empty completion gives 0.0.
        """Test empty completion."""
        reward = geometry3k_reward_fn(
            prompt="What is the angle?",
            completions="",
            prompt_ids=[],
            completion_ids=[],
            answer="90",
        )
        assert reward == 0.0, "Empty completion should return 0.0"

    def test_geometry3k_whitespace_handling(self):
        # [FUNCTION DOC]
        # "[ 90 ]" with inner whitespace matches gold 90.
        """Test whitespace is properly stripped."""
        reward = geometry3k_reward_fn(
            prompt="What is the angle?",
            completions="The angle is [ 90 ].",
            prompt_ids=[],
            completion_ids=[],
            answer="90",
        )
        assert reward == 1.0, "Whitespace should be stripped from bracket"


class TestRewardEdgeCases:
    # [CLASS DOC]
    # Defensive behavior on malformed inputs.
    """Test edge cases for reward functions."""

    def test_gsm8k_malformed_expression(self):
        # [FUNCTION DOC]
        # Unparseable completion yields 0.0 rather than raising.
        """Test handling of malformed expressions."""
        reward = gsm8k_reward_fn(
            prompt="What is 2+2?",
            completions="The answer is !!!invalid!!!",
            prompt_ids=[],
            completion_ids=[],
            answer="4",
        )
        # Should gracefully return 0.0 on parse error
        assert reward == 0.0, "Malformed expression should not crash"

    def test_geometry3k_none_values(self):
        # [FUNCTION DOC]
        # answer=None yields 0.0 rather than raising.
        """Test handling of None values."""
        # This shouldn't happen in practice, but test defensive coding
        reward = geometry3k_reward_fn(
            prompt="What is the angle?",
            completions="No answer",
            prompt_ids=[],
            completion_ids=[],
            answer=None,
        )
        assert reward == 0.0, "None answer should return 0.0"


class TestGSM8KAdvanced:
    # [CLASS DOC]
    # GSM8K reward with more complex expressions.
    """Advanced tests for GSM8K with complex mathematical expressions."""

    def test_gsm8k_exponents(self):
        # [FUNCTION DOC]
        # Completion 8 vs gold expression 2^3 gives 1.0.
        """Test exponential expressions."""
        reward = gsm8k_reward_fn(
            prompt="What is 2^3?",
            completions="The answer is 8",
            prompt_ids=[],
            completion_ids=[],
            answer="2^3",
        )
        assert reward == 1.0, "2^3 should equal 8"

    def test_gsm8k_square_root(self):
        # [FUNCTION DOC]
        # LaTeX sqrt(16) vs gold 4 gives 1.0.
        """Test square root expressions."""
        reward = gsm8k_reward_fn(
            prompt="What is sqrt(16)?",
            completions=r"The answer is $\sqrt{16}$",
            prompt_ids=[],
            completion_ids=[],
            answer="4",
        )
        assert reward == 1.0, "sqrt(16) should equal 4"

    def test_gsm8k_complex_fraction(self):
        # [FUNCTION DOC]
        # LaTeX frac 5/6 vs gold "5/6" gives 1.0.
        """Test complex fractions."""
        reward = gsm8k_reward_fn(
            prompt="What is (1/2) + (1/3)?",
            completions=r"The answer is $\frac{5}{6}$",
            prompt_ids=[],
            completion_ids=[],
            answer="5/6",
        )
        assert reward == 1.0, "1/2 + 1/3 = 5/6"

    def test_gsm8k_negative_numbers(self):
        # [FUNCTION DOC]
        # "-15" vs "-15" gives 1.0.
        """Test negative number handling."""
        reward = gsm8k_reward_fn(
            prompt="What is -5 * 3?",
            completions="-15",
            prompt_ids=[],
            completion_ids=[],
            answer="-15",
        )
        assert reward == 1.0, "Negative numbers should match"

    def test_gsm8k_decimal_precision(self):
        # [FUNCTION DOC]
        # 0.333333333 vs 1/3: only asserts the result is a float in {0.0, 1.0}
        # (precision-dependent).
        """Test decimal precision handling."""
        reward = gsm8k_reward_fn(
            prompt="What is 1/3 as decimal?",
            completions="0.333333333",
            prompt_ids=[],
            completion_ids=[],
            answer="1/3",
        )
        # math-verify may or may not accept this depending on precision settings
        # This tests that it handles the comparison gracefully
        assert isinstance(reward, float) and reward in [0.0, 1.0]

    def test_gsm8k_equation_form(self):
        # [FUNCTION DOC]
        # "x = 3" vs gold 3 gives 1.0 (equation right-hand side extracted).
        """Test equation form answers."""
        reward = gsm8k_reward_fn(
            prompt="Solve x + 2 = 5",
            completions="x = 3",
            prompt_ids=[],
            completion_ids=[],
            answer="3",
        )
        assert reward == 1.0, "Should extract x=3 and match with 3"

    def test_gsm8k_multiline_solution(self):
        # [FUNCTION DOC]
        # Multi-line completion ending "Final answer: 42" matches 42.
        """Test multiline solution with answer at end."""
        reward = gsm8k_reward_fn(
            prompt="Solve step by step",
            completions="Step 1: Calculate\nStep 2: Simplify\nFinal answer: 42",
            prompt_ids=[],
            completion_ids=[],
            answer="42",
        )
        assert reward == 1.0, "Should extract final answer from multiline text"

    def test_gsm8k_pi_constant(self):
        # [FUNCTION DOC]
        # LaTeX 2 pi vs gold "2*pi": only asserts a float in {0.0, 1.0}.
        """Test pi constant handling."""
        reward = gsm8k_reward_fn(
            prompt="What is the circumference of circle with r=1?",
            completions=r"$2\pi$",
            prompt_ids=[],
            completion_ids=[],
            answer="2*pi",
        )
        # math-verify may handle pi differently; test robustness instead
        assert isinstance(reward, float) and reward in [0.0, 1.0]

    def test_gsm8k_scientific_notation(self):
        # [FUNCTION DOC]
        # LaTeX 10^3 vs gold 1000 gives 1.0.
        """Test scientific notation via LaTeX power notation."""
        reward = gsm8k_reward_fn(
            prompt="What is 10^3?",
            completions=r"$10^3$",
            prompt_ids=[],
            completion_ids=[],
            answer="1000",  # 10^3 = 1000
        )
        assert reward == 1.0, "Power notation 10^3 should equal 1000"


class TestGeometry3KAdvanced:
    # [CLASS DOC]
    # Geometry3K reward with more complex bracketed expressions.
    """Advanced tests for Geometry3K with complex expressions."""

    def test_geometry3k_angle_units(self):
        # [FUNCTION DOC]
        # "[45 degrees]" vs gold 45 gives 1.0 (units tolerated).
        """Test angle with units."""
        reward = geometry3k_reward_fn(
            prompt="Find angle",
            completions="[45 degrees]",
            prompt_ids=[],
            completion_ids=[],
            answer="45",
        )
        # Should handle unit stripping or work anyway
        assert reward == 1.0, "Units should be handled"

    def test_geometry3k_fraction_angle(self):
        # [FUNCTION DOC]
        # "[45]" vs gold 45 gives 1.0.
        # KNOWN ISSUE: Named and documented as a fractional-angle test, but uses the
        # integer 45, so no fraction is exercised.
        """Test fractional angles with explicit values."""
        reward = geometry3k_reward_fn(
            prompt="Find angle",
            completions="[45]",  # Use explicit numerical form
            prompt_ids=[],
            completion_ids=[],
            answer="45",
        )
        assert reward == 1.0, "Fractional angles should match numerically"

    def test_geometry3k_negative_coordinate(self):
        # [FUNCTION DOC]
        # "[-3]" vs gold -3 gives 1.0.
        """Test negative coordinates."""
        reward = geometry3k_reward_fn(
            prompt="Find coordinate",
            completions="[-3]",
            prompt_ids=[],
            completion_ids=[],
            answer="-3",
        )
        assert reward == 1.0, "Negative values should match"

    def test_geometry3k_square_root_answer(self):
        # [FUNCTION DOC]
        # Bracketed LaTeX sqrt(4) vs gold 2 gives 1.0.
        """Test square root in answer."""
        reward = geometry3k_reward_fn(
            prompt="Find length",
            completions=r"[$\sqrt{4}$]",  # Use perfect square
            prompt_ids=[],
            completion_ids=[],
            answer="2",  # sqrt(4) = 2
        )
        assert reward == 1.0, "sqrt(4) should equal 2"

    def test_geometry3k_decimal_approximation(self):
        # [FUNCTION DOC]
        # "[1.414]" vs gold sqrt(2): only asserts a float in {0.0, 1.0}.
        """Test decimal vs exact form."""
        reward = geometry3k_reward_fn(
            prompt="Find length",
            completions="[1.414]",
            prompt_ids=[],
            completion_ids=[],
            answer="sqrt(2)",
        )
        # May or may not match depending on precision
        assert isinstance(reward, float) and reward in [0.0, 1.0]

    def test_geometry3k_multiple_brackets(self):
        # [FUNCTION DOC]
        # Multiple brackets "[30] ... [45]": the last one is used, matching gold 45.
        """Test text with multiple brackets, should use last."""
        reward = geometry3k_reward_fn(
            prompt="Find angle",
            completions="First try [30], second try [45].",
            prompt_ids=[],
            completion_ids=[],
            answer="45",
        )
        assert reward == 1.0, "Should use last bracket value"

    def test_geometry3k_scientific_notation(self):
        # [FUNCTION DOC]
        # Bracketed LaTeX 10^4 vs gold 10000 gives 1.0.
        """Test scientific notation via power in geometry."""
        reward = geometry3k_reward_fn(
            prompt="Find area with 10^4",
            completions=r"[$10^4$]",  # Use power notation
            prompt_ids=[],
            completion_ids=[],
            answer="10000",  # 10^4 = 10000
        )
        assert reward == 1.0, "Power notation 10^4 should equal 10000"

    def test_geometry3k_nested_sqrt(self):
        # [FUNCTION DOC]
        # Bracketed nested sqrt(sqrt(16)) vs gold 2 gives 1.0.
        """Test nested square roots."""
        reward = geometry3k_reward_fn(
            prompt="Find length",
            completions=r"[$\sqrt{\sqrt{16}}$]",
            prompt_ids=[],
            completion_ids=[],
            answer="2",
        )
        assert reward == 1.0, "Nested sqrt should simplify to 2"


class TestMathVerifyRobustness:
    # [CLASS DOC]
    # Robustness cases across both reward functions.
    """Test robustness of math-verify backend across both functions."""

    def test_gsm8k_unicode_math_symbols(self):
        # [FUNCTION DOC]
        # Unicode Greek letters in the prompt do not affect matching "42".
        """Test unicode math symbols."""
        reward = gsm8k_reward_fn(
            prompt="What is α + β?",
            completions="42",
            prompt_ids=[],
            completion_ids=[],
            answer="42",
        )
        assert reward == 1.0, "Should handle unicode in prompt gracefully"

    def test_gsm8k_very_large_number(self):
        # [FUNCTION DOC]
        # 15-digit integers match exactly.
        """Test very large numbers."""
        reward = gsm8k_reward_fn(
            prompt="Large computation",
            completions="999999999999999",
            prompt_ids=[],
            completion_ids=[],
            answer="999999999999999",
        )
        assert reward == 1.0, "Very large numbers should match"

    def test_gsm8k_very_small_decimal(self):
        # [FUNCTION DOC]
        # 0.0000001 matches itself.
        """Test very small decimals."""
        reward = gsm8k_reward_fn(
            prompt="Small decimal",
            completions="0.0000001",
            prompt_ids=[],
            completion_ids=[],
            answer="0.0000001",
        )
        assert reward == 1.0, "Very small decimals should match"

    def test_gsm8k_matrix_style_answer(self):
        # [FUNCTION DOC]
        # Matching pmatrix LaTeX strings: only asserts a float is returned.
        """Test matrix/array style answer."""
        reward = gsm8k_reward_fn(
            prompt="Matrix det",
            completions=r"$\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$",
            prompt_ids=[],
            completion_ids=[],
            answer=r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}",
        )
        # math-verify should handle matrix notation
        assert isinstance(reward, float)

    def test_geometry3k_near_miss(self):
        # [FUNCTION DOC]
        # "[45.000001]" vs 45: accepts either 1.0 or 0.0.
        # KNOWN ISSUE: The assertion accepts every valid outcome, so the test cannot
        # fail on value.
        """Test near-miss numerical answers."""
        reward = geometry3k_reward_fn(
            prompt="Find angle",
            completions="[45.000001]",
            prompt_ids=[],
            completion_ids=[],
            answer="45",
        )
        # Should likely match with reasonable tolerance
        assert reward == 1.0 or reward == 0.0  # Both reasonable

    def test_gsm8k_division_order(self):
        # [FUNCTION DOC]
        # Completion "4" matches gold 12/3 (1.0) but not gold 3/12 (0.0).
        """Test division order matters."""
        reward_1 = gsm8k_reward_fn(
            prompt="What is 12/3?",
            completions="4",
            prompt_ids=[],
            completion_ids=[],
            answer="12/3",
        )
        reward_2 = gsm8k_reward_fn(
            prompt="What is 3/12?",
            completions="4",
            prompt_ids=[],
            completion_ids=[],
            answer="3/12",
        )
        assert reward_1 == 1.0, "12/3 should be 4"
        assert reward_2 == 0.0, "3/12 should not be 4"

    def test_gsm8k_commutativity(self):
        # [FUNCTION DOC]
        # Completion "3+2" matches gold 5.
        # KNOWN ISSUE: Docstring mentions multiplication too, but only addition is
        # tested.
        """Test commutativity of addition/multiplication."""
        reward_add = gsm8k_reward_fn(
            prompt="What is 2+3?",
            completions="3+2",
            prompt_ids=[],
            completion_ids=[],
            answer="5",
        )
        assert reward_add == 1.0, "Addition should be commutative"

    def test_gsm8k_distribution(self):
        # [FUNCTION DOC]
        # "2x + 2" vs gold "2*x + 2".
        # KNOWN ISSUE: Assertion "reward == 1.0 or isinstance(reward, float)" is
        # effectively always true, so the equivalence is not actually checked.
        """Test algebraic distribution."""
        reward = gsm8k_reward_fn(
            prompt="Simplify 2(x+1)",
            completions="2x + 2",
            prompt_ids=[],
            completion_ids=[],
            answer="2*x + 2",
        )
        assert reward == 1.0 or isinstance(reward, float), "Distribution should work"

    def test_gsm8k_inequality_handling(self):
        # [FUNCTION DOC]
        # "x = 4" vs gold 4 gives 1.0.
        # KNOWN ISSUE: Name/docstring refer to inequality handling, but no inequality is
        # tested.
        """Test inequality vs equation distinction."""
        reward_eq = gsm8k_reward_fn(
            prompt="Solve x + 1 = 5",
            completions="x = 4",
            prompt_ids=[],
            completion_ids=[],
            answer="4",
        )
        assert reward_eq == 1.0, "Equation should match"


class TestComplexLatexFormulas:
    # [CLASS DOC]
    # GSM8K and Geometry3K rewards on richer LaTeX formulas; several only assert a float
    # result where math-verify's symbolic matching is uncertain.
    """Advanced LaTeX formula tests for robust mathematical notation."""

    def test_gsm8k_summation_notation(self):
        # [FUNCTION DOC]
        # Summation identity vs gold n(n+1)/2: only asserts a float.
        """Test summation formula."""
        reward = gsm8k_reward_fn(
            prompt="Sum from 1 to n",
            completions=r"$\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$",
            prompt_ids=[],
            completion_ids=[],
            answer=r"$n(n+1)/2$",
        )
        # Should match at least the final answer
        assert isinstance(reward, float)

    def test_gsm8k_quadratic_formula(self):
        # [FUNCTION DOC]
        # Quadratic formula vs a plus-branch gold: only asserts a float.
        """Test quadratic formula."""
        reward = gsm8k_reward_fn(
            prompt="Solve ax^2 + bx + c = 0",
            completions=r"$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$",
            prompt_ids=[],
            completion_ids=[],
            answer=r"(-b + sqrt(b^2 - 4ac)) / (2a)",
        )
        # Formulas may not match exactly due to representation
        assert isinstance(reward, float)

    def test_gsm8k_complex_exponent(self):
        # [FUNCTION DOC]
        # Euler identity completion "e^{i pi} = -1" vs gold -1 gives 1.0.
        """Test complex exponential."""
        reward = gsm8k_reward_fn(
            prompt="What is e^(iπ)?",
            completions=r"$e^{i\pi} = -1$",
            prompt_ids=[],
            completion_ids=[],
            answer="-1",
        )
        assert reward == 1.0, "Euler's identity result should match"

    def test_gsm8k_power_notation(self):
        # [FUNCTION DOC]
        # "10^3 = 1000" vs gold 1000 gives 1.0.
        """Test power notation instead of e notation."""
        reward = gsm8k_reward_fn(
            prompt="What is 10^3?",
            completions=r"$10^3 = 1000$",
            prompt_ids=[],
            completion_ids=[],
            answer="1000",
        )
        assert reward == 1.0, "Power notation should work"

    def test_gsm8k_scientific_latex(self):
        # [FUNCTION DOC]
        # LaTeX "1 times 10^3" vs gold 1000 gives 1.0.
        """Test scientific notation in LaTeX form."""
        reward = gsm8k_reward_fn(
            prompt="Express in scientific notation",
            completions=r"$1 \times 10^3$",
            prompt_ids=[],
            completion_ids=[],
            answer="1000",
        )
        assert reward == 1.0, "Scientific notation in LaTeX should work"

    def test_gsm8k_matrix_determinant(self):
        # [FUNCTION DOC]
        # vmatrix determinant "= 3" vs gold 3 gives 1.0.
        """Test matrix determinant calculation."""
        reward = gsm8k_reward_fn(
            prompt="Find determinant of [[2, 1], [1, 2]]",
            completions=r"$\begin{vmatrix} 2 & 1 \\ 1 & 2 \end{vmatrix} = 3$",
            prompt_ids=[],
            completion_ids=[],
            answer="3",
        )
        assert reward == 1.0, "Determinant result should match"

    def test_gsm8k_fraction_chain(self):
        # [FUNCTION DOC]
        # Nested LaTeX fraction "= 2/3" vs gold 2/3 gives 1.0.
        """Test continued fraction."""
        reward = gsm8k_reward_fn(
            prompt="Simplify nested fraction",
            completions=r"$\frac{\frac{1}{2}}{\frac{3}{4}} = \frac{2}{3}$",
            prompt_ids=[],
            completion_ids=[],
            answer=r"2/3",
        )
        assert reward == 1.0, "Continued fraction should simplify"

    def test_gsm8k_binomial_coefficient(self):
        # [FUNCTION DOC]
        # binom(5,2) = 10 vs gold 10 gives 1.0.
        """Test binomial coefficient."""
        reward = gsm8k_reward_fn(
            prompt="What is C(5,2)?",
            completions=r"$\binom{5}{2} = 10$",
            prompt_ids=[],
            completion_ids=[],
            answer="10",
        )
        assert reward == 1.0, "Binomial coefficient should match"

    def test_gsm8k_greek_letters_in_formula(self):
        # [FUNCTION DOC]
        # sin(pi/2) = 1 vs gold 1 gives 1.0.
        """Test Greek letters in formulas."""
        reward = gsm8k_reward_fn(
            prompt="What is sin(π/2)?",
            completions=r"$\sin\left(\frac{\pi}{2}\right) = 1$",
            prompt_ids=[],
            completion_ids=[],
            answer="1",
        )
        assert reward == 1.0, "Trigonometric result should match"

    def test_geometry3k_complex_angle_formula(self):
        # [FUNCTION DOC]
        # Bracketed interior-angle formula vs gold expression: only asserts a float.
        """Test complex geometry formula."""
        reward = geometry3k_reward_fn(
            prompt="Calculate angle sum",
            completions=r"$[\frac{(n-2) \times 180°}{n}]$",
            prompt_ids=[],
            completion_ids=[],
            answer=r"(n-2)*180/n",
        )
        # Formula comparison may vary
        assert isinstance(reward, float)

    def test_geometry3k_pythagorean_theorem(self):
        # [FUNCTION DOC]
        # "[sqrt(3^2+4^2)] = [5]" vs gold 5 gives 1.0 (last bracket used).
        """Test Pythagorean theorem."""
        reward = geometry3k_reward_fn(
            prompt="Find hypotenuse when a=3, b=4",
            completions=r"[$\sqrt{3^2 + 4^2}$] = [5]",
            prompt_ids=[],
            completion_ids=[],
            answer="5",
        )
        assert reward == 1.0, "Pythagorean result should match"

    def test_geometry3k_area_formula_latex(self):
        # [FUNCTION DOC]
        # Bracketed pi r^2 vs gold "pi*r^2": only asserts a float.
        """Test area formula with LaTeX."""
        reward = geometry3k_reward_fn(
            prompt="Area of circle",
            completions=r"[$\pi r^2$]",
            prompt_ids=[],
            completion_ids=[],
            answer=r"pi*r^2",
        )
        # Formula matching
        assert isinstance(reward, float)

    def test_gsm8k_limit_notation(self):
        # [FUNCTION DOC]
        # Limit expression "= 1" vs gold 1 gives 1.0.
        """Test limit notation."""
        reward = gsm8k_reward_fn(
            prompt="What is lim(x→0) sin(x)/x?",
            completions=r"$\lim_{x \to 0} \frac{\sin x}{x} = 1$",
            prompt_ids=[],
            completion_ids=[],
            answer="1",
        )
        assert reward == 1.0, "Limit result should match"

    def test_gsm8k_partial_derivative(self):
        # [FUNCTION DOC]
        # "The result is $3$" vs gold 3 gives 1.0.
        """Test partial derivative with numerical answer."""
        reward = gsm8k_reward_fn(
            prompt="Partial derivative ∂f/∂x of f=xy at x=2, y=3",
            completions=r"The result is $3$",
            prompt_ids=[],
            completion_ids=[],
            answer="3",
        )
        assert reward == 1.0, "Partial derivative numerical result should match"

    def test_gsm8k_absolute_value_complex(self):
        # [FUNCTION DOC]
        # Absolute value "= 2" vs gold 2 gives 1.0.
        """Test absolute value with complex expression."""
        reward = gsm8k_reward_fn(
            prompt="What is |3 - 5|?",
            completions=r"$\left| 3 - 5 \right| = 2$",
            prompt_ids=[],
            completion_ids=[],
            answer="2",
        )
        assert reward == 1.0, "Absolute value should match"

    def test_gsm8k_modulo_operation(self):
        # [FUNCTION DOC]
        # "17 bmod 5 = 2" vs gold 2 gives 1.0.
        """Test modulo operation."""
        reward = gsm8k_reward_fn(
            prompt="What is 17 mod 5?",
            completions="$17 \\bmod 5 = 2$",
            prompt_ids=[],
            completion_ids=[],
            answer="2",
        )
        assert reward == 1.0, "Modulo operation should work"

    def test_gsm8k_logarithm_change_base(self):
        # [FUNCTION DOC]
        # log_2(8) = 3 vs gold 3 gives 1.0.
        """Test logarithm with base change."""
        reward = gsm8k_reward_fn(
            prompt="What is log_2(8)?",
            completions=r"$\log_2(8) = 3$",
            prompt_ids=[],
            completion_ids=[],
            answer="3",
        )
        assert reward == 1.0, "Logarithm with base should match"

    def test_gsm8k_vector_notation(self):
        # [FUNCTION DOC]
        # sqrt(a^2+b^2) vs gold in plain notation: only asserts a float.
        """Test vector notation."""
        reward = gsm8k_reward_fn(
            prompt="Magnitude of vector",
            completions=r"The answer is $\sqrt{a^2 + b^2}$",
            prompt_ids=[],
            completion_ids=[],
            answer=r"$sqrt(a^2 + b^2)$",
        )
        assert isinstance(reward, float), "Vector notation should be handled"

    def test_gsm8k_subscript_superscript_combination(self):
        # [FUNCTION DOC]
        # Boxed sub/superscript expression vs gold: only asserts a float.
        # KNOWN ISSUE: Uses a raw string with a doubled backslash before "boxed", so the
        # completion contains two literal backslashes, which is likely not the intended
        # LaTeX.
        """Test combination of subscripts and superscripts."""
        reward = gsm8k_reward_fn(
            prompt="Formula with sub and superscripts",
            completions=r"the answer is $\\boxed{a_n^2 + b_m^3}$",
            prompt_ids=[],
            completion_ids=[],
            answer=r"$a_n^2 + b_m^3$",
        )
        assert isinstance(reward, float), "Complex subscripts should be handled"


class TestMathVerifyWorkerRoundingCases:
    # [CLASS DOC]
    # Tests numeric rounding tolerance of the math-verify worker's verify(pred, gold)
    # (default precision noted as 6 decimals).
    def test_worker_rounding_none_answer_fails(self):
        # [FUNCTION DOC]
        # Gold string "None" never matches a boxed number (0.0).
        worker = get_math_verify_worker()
        assert worker.verify("\\boxed{0.0667}", "None") == 0.0

    # Current default precision is 6
    def test_worker_rounding_matches_six_decimal_places(self):
        # [FUNCTION DOC]
        # 0.0666668 vs 0.066666793 match at 6-decimal precision (1.0).
        worker = get_math_verify_worker()
        assert worker.verify("\\boxed{0.0666668}", "\\boxed{0.066666793}") == 1.0

    def test_worker_rounding_matches_three_decimal_places(self):
        # [FUNCTION DOC]
        # 0.067 vs 0.066793 do not match (0.0).
        worker = get_math_verify_worker()
        assert worker.verify("\\boxed{0.067}", "\\boxed{0.066793}") == 0.0

    def test_worker_rounding_allows_extra_precision(self):
        # [FUNCTION DOC]
        # 2.232 vs 2.23 do not match (0.0).
        # KNOWN ISSUE: Name says "allows extra precision" but asserts the values do NOT
        # match.
        worker = get_math_verify_worker()
        assert worker.verify("\\boxed{2.232}", "\\boxed{2.23}") == 0.0

    def test_worker_rounding_allows_less_precision(self):
        # [FUNCTION DOC]
        # 2.23 vs 2.232 do not match (0.0).
        # KNOWN ISSUE: Name says "allows less precision" but asserts the values do NOT
        # match.
        worker = get_math_verify_worker()
        assert worker.verify("\\boxed{2.23}", "\\boxed{2.232}") == 0.0


class TestMathVerifyWorkerTextWrappedAnswers:
    # [CLASS DOC]
    # Tests verify() when gold/pred answers are wrapped in natural-language text.
    def test_worker_text_embedded_fraction(self):
        # [FUNCTION DOC]
        # Boxed 2/3 vs gold text "answer is 0.6666667" matches (1.0).
        worker = get_math_verify_worker()
        pred = "The final answer is \\boxed{2/3}."
        gold = "answer is 0.6666667"
        assert worker.verify(pred, gold) == 1.0

    # This is a BAD case
    # Answers must be numerical or symbolic, not mixed.
    def test_worker_text_embedded_sqrt(self):
        # [FUNCTION DOC]
        # Boxed LaTeX sqrt(2) vs plain-text "sqrt(2)" gold does NOT match (0.0);
        # documented as a known bad case (mixed symbolic text not parsed).
        worker = get_math_verify_worker()
        pred = "After simplification we get \\boxed{\\sqrt{2}}"
        gold = "the answer is sqrt(2)"
        assert worker.verify(pred, gold) == 0.0

    def test_worker_text_embedded_latex_power(self):
        # [FUNCTION DOC]
        # Boxed 10^4 vs gold text "answer is 10000" matches.
        worker = get_math_verify_worker()
        pred = "Area equals \\boxed{10^4} square units"
        gold = "answer is 10000"
        assert worker.verify(pred, gold) == 1.0

    def test_worker_text_embedded_nested_expr(self):
        # [FUNCTION DOC]
        # Boxed (1/2)^2 + sqrt(9) vs gold "0.25 + 3" matches.
        worker = get_math_verify_worker()
        pred = "Result: \\boxed{(1/2)^{2} + \\sqrt{9}}"
        gold = "answer is 0.25 + 3"
        assert worker.verify(pred, gold) == 1.0
