Approach 1: Interactive Textual and Visual Representation

Workflow:
User Input: User submits a query like "What is the Pythagorean Theorem?".
Fetch Textual Explanation: SLATE sends this query to the Gemini API and receives a textual explanation.
User Decision: SLATE asks the user if they would like a visual representation.
Conditional Visual Representation:
If the user answers "Yes", SLATE sends a modified prompt to the Vertex AI (e.g., "Give a visual representation of the Pythagorean Theorem.") to generate the Manim code.
If the user answers "No", the process ends.
Render Video: If the user responded "Yes", SLATE renders the Manim code into a video.
Display Both: If visual representation was chosen, SLATE displays both the textual explanation and the video to the user.


Approach 2: Automated Textual and Visual Representation

User Input: User submits a query like "What is the Pythagorean Theorem?".
Fetch Textual Explanation: SLATE sends this query to the Gemini API and receives a textual explanation.
Generate Visual Representation: Simultaneously, SLATE sends a modified prompt to the Vertex AI (e.g., "Give a visual representation of the Pythagorean Theorem.") to generate the Manim code.
Render Video: SLATE takes the received Manim code, renders it into a video.
Display Both: SLATE displays both the textual explanation and the video to the user.
