examples = {
    "permutation":{"""
from manim import *

class PermutationsScene(Scene):
    def construct(self):
        # Create the title text
        title = Text('Permutations of ABC').scale(0.9)
        
        # Write the title text to the scene
        self.play(Write(title))
        self.wait(1)
        
        # Move the title text to the top edge of the scene
        self.play(title.animate.to_edge(UP))
        
        # Create and display the initial item text
        items = Text('ABC').scale(1.2)
        self.play(Write(items))
        self.wait(1)
        
        # Define the permutations of 'ABC'
        permutations = ['ABC', 'ACB', 'BAC', 'BCA', 'CAB', 'CBA']
        
        # Create a group of Text objects for each permutation
        perm_texts = VGroup(*[Text(perm).scale(0.8) for perm in permutations]).arrange_in_grid(rows=2, cols=3, buff=1)
        
        # Transform the initial items text into the grid of permutations
        self.play(Transform(items, perm_texts))
        self.wait(2)
        
        # Create and display the total count of permutations
        count_text = Text(f'Total permutations = {len(permutations)}').scale(0.8).to_edge(DOWN)
        self.play(Write(count_text))
        self.wait(2)
        
        # Create and display the mathematical explanation of permutations
        explanation_text = MathTex(r'P(n) = n! = 3! = 6').to_edge(DOWN)
        self.play(ReplacementTransform(count_text, explanation_text))
        self.wait(2)
"""},
      "binomial": {"""
from manim import *

class BinomialTheoremScene(Scene):
    def construct(self):
        # Create and write the title
        title = Text('Binomial Theorem: (a + b)^2').scale(0.9)
        self.play(Write(title))
        self.wait(1)
        
        # Move the title to the upper edge of the screen
        self.play(title.animate.to_edge(UP))
        
        # Display the expansion formula of (a+b)^2
        expansion_formula = MathTex('(a + b)^2 = a^2 + 2ab + b^2')
        self.play(Write(expansion_formula))
        self.wait(1)
        
        # Create and arrange Pascal's Triangle below the expansion formula
        pascals_triangle = VGroup(
            MathTex('1'),
            MathTex('1\\quad 1'),
            MathTex('1\\quad 2\\quad 1'),
            MathTex('1\\quad 3\\quad 3\\quad 1')
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).next_to(expansion_formula, DOWN, buff=0.5)
        
        # Slightly adjust the position of the last row for better alignment
        pascals_triangle[-1].shift(UP * 0.2)
        self.play(Write(pascals_triangle))
        self.wait(1)
        
        # Highlight the third row of Pascal's Triangle
        highlight = SurroundingRectangle(pascals_triangle[2], color=YELLOW)
        self.play(Create(highlight))
        self.wait(2)
        
        # Display an explanatory text regarding coefficients
        explanation_text = Text('Coefficients come from Pascal\'s Triangle', font_size=24).next_to(pascals_triangle, DOWN)
        self.play(Write(explanation_text))
        self.wait(2)
"""},
        "plot": {"""
from manim import *
import numpy as np

class Surface3DPlot(ThreeDScene):
    def construct(self):
        # Create axes
        axes = ThreeDAxes(
            x_range=[-3, 3, 1],
            y_range=[-3, 3, 1],
            z_range=[-1, 1, 0.5]
        )

        # Define the surface function z = sin(x) * cos(y)
        surface = Surface(
            lambda u, v: axes.c2p(u, v, np.sin(u) * np.cos(v)),
            u_range=[-PI, PI],
            v_range=[-PI, PI],
            color=BLUE
        )

        # Set up the camera orientation
        self.set_camera_orientation(phi=45 * DEGREES, theta=45 * DEGREES)
        self.add(axes)
        self.play(Create(surface))
        self.begin_ambient_camera_rotation(rate=0.2)  # Optional: Rotate the camera
        self.wait(4)
"""},
        "graph": {"""
from manim import *

class GraphMultipleFunctions(Scene):
    def construct(self):
        # Create axes
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-3, 9, 1],
            axis_config={'color': BLUE}
        )

        # Function y = x
        linear_function = axes.plot(lambda x: x, color=GREEN)
        linear_label = MathTex('y = x').next_to(linear_function, UP+LEFT)

        # Function y = x^2
        quadratic_function = axes.plot(lambda x: x**2, color=RED)
        quadratic_label = MathTex('y = x^2').next_to(quadratic_function.get_top(), UP)

        # Add all elements to the scene
        self.add(axes, linear_function, quadratic_function, linear_label, quadratic_label)
        self.play(Create(linear_function), Create(quadratic_function), Write(linear_label), Write(quadratic_label))
        self.wait(1)
"""}
}