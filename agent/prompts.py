task_template_prompt = """You are an agent that is trained to analyse and retell the object movement and camera motion based on a text and an image.

First, you should analyse the given text and image. 
Then, you shoule seperate the given text in to two parts, including the object movement and the camera motion.
Finally, you should reply me these two parts, where the first part is object movement and the second part is camera motion.
Please note that If the text is oppsite with the image content, you should follow the text rather than the image.

Here are some examples for you:

If there are object movement and camera motion in the text, you should divide the text into two part.
Text: "a close up of a blue and orange liquid, camera tilts down."
Reply: <a close up of a blue and orange liquid.>, <camera tilts down.>

If there is no camera motion in the text, you can select a simple camera motion according to the image content from: pan left/right, tilt up/down, zoom in/out.
Text: "a man in a mask is walking through a crowd of people."
Reply: <a man in a mask is walking through a crowd of people.>, <camera zooms in.>

If there is neither object movement or camera motion in the text, you can imagine a object movement based on the image content and select a simple camera motion from: pan left/right, tilt up/down, zoom in/out.
Text: ""
Reply: <a man riding a bike down a street.>, <camera zooms in.>

The text is: <short_text>.
Next, I will give you an image.

Your output should include four parts in the given format:
Observation: <Describe what you observe in the image.>
Thought: <To complete the given task, what you should do.>
Reply: <Reply me two parts with the required format.>
Summary: <Summarize your actions in one or two sentences.>

This is a reply template:
Observation: The image shows two hot-air balloons in the sky. The larger hot-air balloon is on the right, closer to the foreground, while the smaller one is on the left, further in the distance.
Thought: Based on the text, I need to separate the object's movement and the camera motion described. If there is any discrepancy between the text and the image, I must prioritize the text.
Reply: <object movement: The bigger hot-air balloon flies toward the right, the smaller hot-air balloon flies upward.>, <camera static.>
Summary: The task involved analyzing the given text and image, prioritizing the text's details over the image when there was a conflict, and dividing the information into object movement and camera motion as directed."""

task_template_description = """You are an agent that is trained to describe the object movement based on a text and an image.

First, you should analyse the given text and image.
Then, you should divide the text into several single sentences. Each sentence is used to describe an individual object movement with direction.
Next, you can polish and extend every single sentence.
Finally, you should reply me these sentences.
Please note that If the text is oppsite with the image content, you should follow the text rather than the image.

Here are some examples for you:

If there is specific object movement described in the text, you should divide the long given into several single sentences without creating new object movement that do not exist in the given text. 
This example is the case that you already know the objects with movement, and you should divide the long sentence into short sentence directly.
Text: "A boy shakes his face from side to side, a horse goes forward, and a man walks to the right."
Reply: <A boy shakes his face from side to side.>, <A horse goes forward.>, <A man walks to the right.>
This example is the case that different parts of an object move separately, you should use different sentences to describe independent movement of different parts in the same object.
Text: "A little girl is raising her right hand up and turning her face to left."
Reply: <A little girl is raising her right hand up.>, <A little girl is turning her face to left.>
This example is the case that different objects are doing the same movement, you should use different sentence to describe the same movement of different objects.
Text: "A child and a man are nodding."
Reply: <A child is nodding.>, <A man is nodding.>
This example is the case that different objects are doing the same movement, but the different object use the same name, you should distinguish them according to the given image content (such as: location) and divide them into different sentences.
Text: "These two women are shaking their faces."
Reply: <The woman on the left is shaking her face.>, <The woman on the right is shaking her face.>
This example is the case that an object is moving with multiple directions (such as: first, then) or performing an action repeatedly, you should not divide the sentence and should desribe the complete object movement in a sentence.
Text: "A woman is turning her face first up then down."
Reply: <A woman is turning her face first up then down.>

If there is no specific movement but some objects described in the text, you can create new movement for each object according to the text and image, but cannot create new object that do not exist in the given text.
Text: "a boy, a horse, and a man."
Reply: <A boy turns his face to the right.>, <A horse goes forward.>, <A man walks to the right.>

If the given text don't include any object or no text is given, you can create new object and corresponding movement according to the content of image, and reply me with no more than five sentences.
Text: "A view of landscape."
Reply: <A boy turns his face to the right.>, <A horse goes forward.>, <A man walks to the right.>

The text is: <short_text>.
Next, I will give you an image.

Your output should include four parts in the given format:
Observation: <Describe what you observe in the image.>
Thought: <To complete the given task, what you should do.>
Reply: <Reply me no more than five sentences with the required format without "\n", such as: <*****>, <*****>, <*****>, <*****>, <*****>.>
Summary: <Summarize your actions in one or two sentences.>

This is a reply template:
Observation: The image shows two sheep in a grassy area. The sheep on the left appears larger, and both are lowering their heads, seemingly eating grass.                                                                                          
Thought: The text describes specific movements of the sheep eating grass. I will split the description into two sentences, one for each sheep, and follow the text instructions.                                                               
Reply: <The sheep on the left is eating grass.>, <The sheep on the right is eating grass.>                                                                                                                                                   
Summary: I described the movement of both sheep eating grass as per the text, dividing the action into individual sentences."""

task_template_object = """You are an agent that is trained to find the dynamic objects based on a text and an image.

First, you should analyse the given text and image.
Then, you should find one or more dynamic objects desribed in the text according to the image.
Next, you should reply me these objects.

Here are some examples for you:

If the given text indentifies the dynamic objects, you should reply me these objects.
example 1:
Text: "A boy shakes his face from side to side."
Reply: <face>
example 2:
Text: "an eagle is flying backward."
Reply: <eagle>
example 3:
Text: "A little girl is raising her right hand up."
Reply: <right hand>
example 4:
Text: "a sailboat is drifting to the left."
Reply: <sailboat>

If the given text does not identify any dynamic object, you should analyse the text based on the given image, and reply me the analysed dynamic objects.
Text: "A man is nodding."
Reply: <head>

The text is: <short_text>.
Next, I will give you an image.

Your output should include four parts in the given format:
Observation: <Describe what you observe in the image.>
Thought: <To complete the given task, what you should do.>
Reply: <Reply me with the required format without "\n", such as: <*****>.>
Summary: <Summarize your actions in one or two sentences.>

This is a reply template:
Observation: The image shows two hot-air balloons in the sky. One is larger and closer to the camera, while the smaller one appears in the distance over the landscape.                                                                            
Thought: I need to identify the dynamic object "The smaller hot-air balloon" described in the text as flying downward.                                                                                                                         
Reply: <smaller hot-air balloon>                                                                                                                                                                                                             
Summary: Based on the text and the image, I identified the dynamic object as the smaller hot-air balloon flying downward."""

task_template_set_start_point = """You are an agent that is trained to set a point on an image based on a text. You will be given an image overlaid by a grid. The grid divides the image into small square areas. Each area is labeled with an integer in the top-left corner.

You can call the following function to set a point:

1. set_point(area: int, subarea: str)
This function is used to set a point.
"area" is the integer label assigned to the grid area which marks the location of the point. "subarea" is a string representing the exact location of the point within the grid area.
The subarea's parameters can take one of the nine values: center, top-left, top, top-right, left, right, bottom-left, bottom, and bottom-right.
A simple use case can be set_point(21, "center"), which sets the point at the center of grid area 21.

The task you need to complete is to set a point on an image according to the text description: <task_description>.
First, you should find the object on the given image that is mentioned in the text description.
Then, you should set a point at the center of the object.
Finally, you should call the function with the correct parameters to return the result.

Your output should include four parts in the given format:
Observation: <Describe what you observe in the image>
Thought: <To complete the given task, what is the step you should do>
Action: <The function call with the correct parameters to proceed with the task. You can only output a function call in this field. Such as: set_point(21, "center")>
Summary: <Summarize your actions in one or two sentences.>"""

task_template_judge_start_point = """You are an agent that is trained to jugde the precision of a point plotted on an image based on a text. You will be given an image overlaid by a grid and a point. The grid divides the image into small square areas. Each area is labeled with an integer in the top-left corner. The plotted point is represented by a circle.

You can call the following function to judge a point:

1. judge_point(judgement: str, area: int, subarea: str)
This function is used to jugde the precision of a point.
"judgement" is the bool label to indicate the precision of the plotted point, which can take two values: "True" or "False". "area" is the integer label assigned to the grid area which marks the location of the point. "subarea" is a string representing the exact location of the point within the grid area.
The subarea's parameters can take one of the nine values: center, top-left, top, top-right, left, right, bottom-left, bottom, and bottom-right.
If you choose "True" in the "judgement", which means you believe the plotted point is presice enough to describe the text, you should return the same "area" and "subarea" with the plotted point's location.
If you choose "False" in the "judgement", which indicates you think the plotted point is inaccurate to represent the text, you should return a new "area" and "subarea" to represent the more precise point's location.
A simple use case can be judge_point("True", 21, "center"), which means you think the plotted point is accurate to describe the text and return the same location of the plotted point.

The task you need to complete is to jugde the precision of a point plotted on an image based on the text description: <task_description>.
The location of the plotted point is: <point_location>.
First, you should find the object on the given image that is mentioned in the text description.
Then, you should judge the point's location is accurate enough to represent the object's center.
Finally, you should call the function with the correct parameters to return the judgement result.

Your output should include four parts in the given format:
Observation: <Describe what you observe in the image>
Thought: <To complete the given task, what is the step you should do>
Action: <The function call with the correct parameters to proceed with the task. You can only output a function call in this field. Such as: judge_point("True", 21, "center")>
Summary: <Summarize your actions in one or two sentences.>"""

task_template_set_start_point_ground = """You are an agent that is trained to select a object on an image based on a text. You will be given an image overlaid by some detection results that is represented by some bounding-boxes and semi-transparent masks. Each bounding-box is labeled by an integer at the top-left corner.

You can call the following function to select a object:

1. select_object(index: int)
This function is used to select a object covered by a bounding-box.
"index" is the integer label assigned to the bounding-box.
A simple use case can be select_object(0), which selects the object labeled 0.

The task you need to complete is to select a object on an image according to the text description: <task_description>.
First, you should find the object covered by a bounding-box on the given image, which is mentioned in the text description.
Then, you should call the function with the correct parameters to return the result.
Please note that if there is only one object in the given image, you should choose label 0.

Your output should include four parts in the given format:
Observation: <Describe what you observe in the image.>
Thought: <To complete the given task, what is the step you should do.>
Action: <The function call with the correct parameters to proceed with the task. You can only output a function call in this field. Such as: select_object(0).>
Summary: <Summarize your actions in one or two sentences.>

This is a reply template:
Observation: The image shows a large pyramid in the background with three objects labeled by bounding boxes. The labels are 0 (a man on a camel highlighted in pink), 1, and 2 (both appear to show camels and possibly riders, highlighted in blue). The man labeled 0 is in the front, while the man on the camel labeled 1 is slightly behind and appears to be moving forward toward the pyramid.                                                                                    
Thought: Based on the description, I need to select the bounding box of "another man on a camel (label 1)" who is moving toward the pyramid.                                                                                                   
Action: select_object(1)                                                                                                                                                                                                                      
Summary: I selected the object labeled 1, which represents another man on a camel moving toward the pyramid as described."""

task_template_trajectory_given_start_long = """You are an agent that is trained to plot a trajectory on an image based on a text and a starting point. You will be given an image overlaid by a grid and a starting point. The grid divides the image into small square areas. Each area is labeled with an integer in the top-left corner. The starting point of the trajectory is represented by a circle.

You can call the following functions to plot a trajectory:

1. set_1_point(start_area: int, start_subarea: str)
This function is used to set a starting point of a trajectory, which represents a static state, such as buildings or landscape.
"start_area" is the integer label assigned to the grid area which marks the starting location of the trajectory. "start_subarea" is a string representing the exact location to begin the trajectory within the grid area.
The subarea's parameters can take one of the nine values: center, top-left, top, top-right, left, right, bottom-left, bottom, and bottom-right.
A simple use case can be set_1_point(21, "center"), which sets the starting point of the trajectory at the center of grid area 21.

2. set_2_points(start_area: int, start_subarea: str, end_area: int, end_subarea: str)
This function is used to set a starting point and an end point of a trajectory, which represents a simple linear trajectory.
"start_area" is the integer label assigned to the grid area which marks the starting location of the trajectory. "start_subarea" is a string representing the exact location to begin the trajectory within the grid area.
"end_area" is the integer label assigned to the grid area which marks the end location of the trajectory. "end_subarea" is a string representing the exact location to end the trajectory within the grid area.
The two subareas' parameters can take one of the nine values: center, top-left, top, top-right, left, right, bottom-left, bottom, and bottom-right.
A simple use case can be set_2_points(21, "center", 25, "right"), which sets the starting point of the trajectory at the center of grid area 21 and the end point of the trajectory at the right part of grid area 25.

3. set_3_points(start_area: int, start_subarea: str, mid_area: int, mid_subarea: str, end_area: int, end_subarea: str)
This function is used to set a starting point, a mid-point and an end point of a trajectory, which represents a little complex trajectory.
"start_area" is the integer label assigned to the grid area which marks the starting location of the trajectory. "start_subarea" is a string representing the exact location to begin the trajectory within the grid area.
"mid_area" is the integer label assigned to the grid area which marks the middle location of the trajectory. "mid_subarea" is a string representing the exact location of the mid-point within the grid area.
"end_area" is the integer label assigned to the grid area which marks the end location of the trajectory. "end_subarea" is a string representing the exact location to end the trajectory within the grid area.
The three subareas' parameters can take one of the nine values: center, top-left, top, top-right, left, right, bottom-left, bottom, and bottom-right.
A simple use case can be set_3_points(21, "center", 25, "right", 28, "bottom-left"), which sets the starting point of the trajectory at the center of grid area 21, the mid-point of the trajectory at the right part of grid area 25, and the end point of the trajectory at the bottom-left part of grid area 28.

4. set_4_points(start_area: int, start_subarea: str, mid_1_area: int, mid_1_subarea: str, mid_2_area: int, mid_2_subarea: str, end_area: int, end_subarea: str)
This function is used to set a starting point, two mid-points and an end point of a trajectory, which represents a complex trajectory.
"start_area" is the integer label assigned to the grid area which marks the starting location of the trajectory. "start_subarea" is a string representing the exact location to begin the trajectory within the grid area.
"mid_<*>_area" is the integer label assigned to the grid area which marks a middle location of the trajectory. "mid_<*>_subarea" is a string representing the exact location of a mid-point within the grid area.
"end_area" is the integer label assigned to the grid area which marks the end location of the trajectory. "end_subarea" is a string representing the exact location to end the trajectory within the grid area.
The four subareas' parameters can take one of the nine values: center, top-left, top, top-right, left, right, bottom-left, bottom, and bottom-right.
A simple use case can be set_4_points(21, "center", 25, "right", 28, "bottom-left", 30, "top-right"), which sets the starting point of the trajectory at the center of grid area 21, the first mid-point of the trajectory at the right part of grid area 25, the second mid-point of the trajectory at the bottom-left part of grid area 28, and the end point of the trajectory at the top-right part of grid area 30.

5. set_5_points(start_area: int, start_subarea: str, mid_1_area: int, mid_1_subarea: str, mid_2_area: int, mid_2_subarea: str, mid_3_area: int, mid_3_subarea: str, end_area: int, end_subarea: str)
This function is used to set a starting point, three mid-points and an end point of a trajectory, which represents a much more complex trajectory.
"start_area" is the integer label assigned to the grid area which marks the starting location of the trajectory. "start_subarea" is a string representing the exact location to begin the trajectory within the grid area.
"mid_<*>_area" is the integer label assigned to the grid area which marks a middle location of the trajectory. "mid_<*>_subarea" is a string representing the exact location of a mid-point within the grid area.
"end_area" is the integer label assigned to the grid area which marks the end location of the trajectory. "end_subarea" is a string representing the exact location to end the trajectory within the grid area.
The five subareas' parameters can take one of the nine values: center, top-left, top, top-right, left, right, bottom-left, bottom, and bottom-right.
A simple use case can be set_5_points(21, "center", 25, "right", 28, "bottom-left", 30, "top-right", 34, "left"), which sets the starting point of the trajectory at the center of grid area 21, the first mid-point of the trajectory at the right part of grid area 25, the second mid-point of the trajectory at the bottom-left part of grid area 28, and the third mid-point of the trajectory at the top-right part of grid area 30, and the end point of the trajectory at the left part of grid area 34.

6. set_6_points(start_area: int, start_subarea: str, mid_1_area: int, mid_1_subarea: str, mid_2_area: int, mid_2_subarea: str, mid_3_area: int, mid_3_subarea: str, mid_4_area: int, mid_4_subarea: str, end_area: int, end_subarea: str)
This function is used to set a starting point, four mid-points and an end point of a trajectory, which represents the most complex trajectory.
"start_area" is the integer label assigned to the grid area which marks the starting location of the trajectory. "start_subarea" is a string representing the exact location to begin the trajectory within the grid area.
"mid_<*>_area" is the integer label assigned to the grid area which marks a middle location of the trajectory. "mid_<*>_subarea" is a string representing the exact location of a mid-point within the grid area.
"end_area" is the integer label assigned to the grid area which marks the end location of the trajectory. "end_subarea" is a string representing the exact location to end the trajectory within the grid area.
The six subareas' parameters can take one of the nine values: center, top-left, top, top-right, left, right, bottom-left, bottom, and bottom-right.
A simple use case can be set_6_points(21, "center", 25, "right", 28, "bottom-left", 30, "top-right", 34, "left", 36, "center"), which sets the starting point of the trajectory at the center of grid area 21, the first mid-point of the trajectory at the right part of grid area 25, the second mid-point of the trajectory at the bottom-left part of grid area 28, and the third mid-point of the trajectory at the top-right part of grid area 30, the fourth mid-point of the trajectory at the left part of grid area 34, and the end point of the trajectory at the center part of grid area 36.

The task you need to complete is to plot a trajectory to discribe: <task_description>.
The location of the trajectory starting point is: <start_point_location>.
Now, given the following labeled image, you need to think and call the function needed to proceed with the task.

First, you should find the location of the object according to the task description, and set the same starting point with the given one.
Next, you should select N mid-points to extend the trajectory.
Finally, you should select an end point to complete the trajectory.
Please note that the distance between two continuous points in the trajectory should be less than 1-2 grids.

Your output should include four parts in the given format:
Observation: <Describe what you observe in the image.>
Thought: <To complete the given task, what is the step you should do.>
Action: <The function call with the correct parameters to proceed with the task. You can only output a function call in this field. Such as: set_4_points(21, "center", 25, "right", 28, "bottom-left", 30, "top-right").>
Summary: <Summarize your actions in one or two sentences.>

This is a reply template:
Observation: The image shows a woman interacting with a child while engaging in an activity. Based on the given starting point, (36, bottom-left), the task is to describe the motion corresponding to a nod (a small repetitive motion).          
Thought: To represent a "nodding" motion, I will create a trajectory starting at (36, bottom-left), move to the top part of the adjacent area (56, top), and then return to the initial area (36, center). This simulates the slight vertical movement of a nod.                                                                                                                                                                                                                     
Action: set_3_points(36, "bottom-left", 56, "top", 36, "center")                                                                                                                                                                              
Summary: The trajectory starts at the bottom-left of grid area 36, moves to the top of grid area 56 to simulate the upward movement of a nod, and returns to the center of grid area 36, representing the completion of a nodding motion."""

task_template_trajectory_given_start = """You are an agent that is trained to plot a trajectory on an image based on a text and a starting point. You will be given an image overlaid by a grid and a starting point. The grid divides the image into small square areas. Each area is labeled with an integer in the top-left corner. The starting point of the trajectory is represented by a circle.

You can call the following functions to plot a trajectory:

1. set_1_point(start_area: int, start_subarea: str)
This function is used to set a starting point of a trajectory, which represents a static state, such as buildings or landscape.
"start_area" is the integer label assigned to the grid area which marks the starting location of the trajectory. "start_subarea" is a string representing the exact location to begin the trajectory within the grid area.
The subarea's parameters can take one of the nine values: center, top-left, top, top-right, left, right, bottom-left, bottom, and bottom-right.
A simple use case can be set_1_point(21, "center"), which sets the starting point of the trajectory at the center of grid area 21.

2. set_2_points(start_area: int, start_subarea: str, end_area: int, end_subarea: str)
This function is used to set a starting point and an end point of a trajectory, which represents a simple linear trajectory.
"start_area" is the integer label assigned to the grid area which marks the starting location of the trajectory. "start_subarea" is a string representing the exact location to begin the trajectory within the grid area.
"end_area" is the integer label assigned to the grid area which marks the end location of the trajectory. "end_subarea" is a string representing the exact location to end the trajectory within the grid area.
The two subareas' parameters can take one of the nine values: center, top-left, top, top-right, left, right, bottom-left, bottom, and bottom-right.
A simple use case can be set_2_points(21, "center", 25, "right"), which sets the starting point of the trajectory at the center of grid area 21 and the end point of the trajectory at the right part of grid area 25.

3. set_3_points(start_area: int, start_subarea: str, mid_area: int, mid_subarea: str, end_area: int, end_subarea: str)
This function is used to set a starting point, a mid-point and an end point of a trajectory, which represents a little complex trajectory.
"start_area" is the integer label assigned to the grid area which marks the starting location of the trajectory. "start_subarea" is a string representing the exact location to begin the trajectory within the grid area.
"mid_area" is the integer label assigned to the grid area which marks the middle location of the trajectory. "mid_subarea" is a string representing the exact location of the mid-point within the grid area.
"end_area" is the integer label assigned to the grid area which marks the end location of the trajectory. "end_subarea" is a string representing the exact location to end the trajectory within the grid area.
The three subareas' parameters can take one of the nine values: center, top-left, top, top-right, left, right, bottom-left, bottom, and bottom-right.
A simple use case can be set_3_points(21, "center", 25, "right", 28, "bottom-left"), which sets the starting point of the trajectory at the center of grid area 21, the mid-point of the trajectory at the right part of grid area 25, and the end point of the trajectory at the bottom-left part of grid area 28.

4. set_4_points(start_area: int, start_subarea: str, mid_1_area: int, mid_1_subarea: str, mid_2_area: int, mid_2_subarea: str, end_area: int, end_subarea: str)
This function is used to set a starting point, two mid-points and an end point of a trajectory, which represents a complex trajectory.
"start_area" is the integer label assigned to the grid area which marks the starting location of the trajectory. "start_subarea" is a string representing the exact location to begin the trajectory within the grid area.
"mid_<*>_area" is the integer label assigned to the grid area which marks a middle location of the trajectory. "mid_<*>_subarea" is a string representing the exact location of a mid-point within the grid area.
"end_area" is the integer label assigned to the grid area which marks the end location of the trajectory. "end_subarea" is a string representing the exact location to end the trajectory within the grid area.
The four subareas' parameters can take one of the nine values: center, top-left, top, top-right, left, right, bottom-left, bottom, and bottom-right.
A simple use case can be set_4_points(21, "center", 25, "right", 28, "bottom-left", 30, "top-right"), which sets the starting point of the trajectory at the center of grid area 21, the first mid-point of the trajectory at the right part of grid area 25, the second mid-point of the trajectory at the bottom-left part of grid area 28, and the end point of the trajectory at the top-right part of grid area 30.

The task you need to complete is to plot a trajectory to discribe: <task_description>.
The location of the trajectory starting point is: <start_point_location>.
Now, given the following labeled image, you need to think and call the function needed to proceed with the task.

First, you should find the location of the object according to the task description, and set the same starting point with the given one.
Next, you should select N mid-points to extend the trajectory.
Finally, you should select an end point to complete the trajectory.
Please note that the distance between two continuous points in the trajectory should be less than 1-2 grids.

Your output should include four parts in the given format:
Observation: <Describe what you observe in the image.>
Thought: <To complete the given task, what is the step you should do.>
Action: <The function call with the correct parameters to proceed with the task. You can only output a function call in this field. Such as: set_4_points(21, "center", 25, "right", 28, "bottom-left", 30, "top-right").>
Summary: <Summarize your actions in one or two sentences.>

This is a reply template:
Observation: The image shows a woman interacting with a child while engaging in an activity. Based on the given starting point, (36, bottom-left), the task is to describe the motion corresponding to a nod (a small repetitive motion).          
Thought: To represent a "nodding" motion, I will create a trajectory starting at (36, bottom-left), move to the top part of the adjacent area (56, top), and then return to the initial area (36, center). This simulates the slight vertical movement of a nod.                                                                                                                                                                                                                     
Action: set_3_points(36, "bottom-left", 56, "top", 36, "center")                                                                                                                                                                              
Summary: The trajectory starts at the bottom-left of grid area 36, moves to the top of grid area 56 to simulate the upward movement of a nod, and returns to the center of grid area 36, representing the completion of a nodding motion."""

task_template_camera = """You are an agent that is trained to generate a camera motion based on a text and an image. please note that the world coordinate is opencv's one, which is x-axis rightwards (camera pans right), y-axis downwards (camera tilts down), and z-axis frontwards (camera zooms in).

You can call the following functions to generate a camera motion:

1. set_camera_motion(x_translation: float, y_translation: float, z_translation: float, x_rotation: int, y_rotation: int, z_rotation: int, motion_type: str)
This function sets a simple camera motion, such as pan down, that is represented by the shifting distance and rotation degrees of the camera optical center on the x-axis, y-axis, and z-axis.
"x_translation" is a floating point ranged from (-1.00, 1.00), which represents the shift distance in the x axis.
"y_translation" is a floating point ranged from (-1.00, 1.00), which represents the shift distance in the y axis.
"z_translation" is a floating point ranged from (-1.00, 1.00), which represents the shift distance in the z axis.
"x_rotation" is a integer ranged from (0, 360), which represents the degrees rotated alone the x axis.
"y_rotation" is a integer ranged from (0, 360), which represents the degrees rotated alone the y axis.
"z_rotation" is a integer ranged from (0, 360), which represents the degrees rotated alone the z axis.
"motion_speed" is a string representing the camera motion type, and the parameters can take one of the three values: uniform, decrement, increment.
A simple use case can be set_camera_motion(0.52, 0.27, -0.85, 0, 0, 0,"uniform"), which represents the uniform camera moves about 0.52 in x axis (camera pans right), 0.27 in y axis (camera tilts down), and -0.85 in z axis (camera zooms out) and no rotation.

2. set_camera_motion_complex(x_translation_1: float, x_translation_2: float, y_translation_1: float, y_translation_2: float, z_translation_1: float, z_translation_2: float, x_rotation: int, y_rotation: int, z_rotation: int)
This function sets a complex camera motion, such as first fast tilt up then slowly zoom in, that is represented by the shifting distance and rotation degrees of the camera optical center on the x-axis, y-axis, and z-axis.
"x_translation_1" is a floating point ranged from (-0.50, 0.50), which represents the first-step shift distance in the x axis.
"x_translation_2" is a floating point ranged from (-0.50, 0.50), which represents the second-step shift distance in the x axis.
"y_translation_1" is a floating point ranged from (-0.50, 0.50), which represents the first-step shift distance in the y axis.
"y_translation_2" is a floating point ranged from (-0.50, 0.50), which represents the second-step shift distance in the y axis.
"z_translation_1" is a floating point ranged from (-0.50, 0.50), which represents the first-step shift distance in the z axis.
"z_translation_2" is a floating point ranged from (-0.50, 0.50), which represents the second-step shift distance in the z axis.
"x_rotation" is a integer ranged from (0, 360), which represents the degrees rotated alone the x axis.
"y_rotation" is a integer ranged from (0, 360), which represents the degrees rotated alone the y axis.
"z_rotation" is a integer ranged from (0, 360), which represents the degrees rotated alone the z axis.
A simple use case can be set_camera_motion_complex(0.16, -0.39, 0.25, -0.25, -0.40, 0.10, 0, 0, 0), which represents the camera moves about first 0.16 then -0.39 in x axis (camera first slowly pans right then fast pans left), first 0.25 then -0.25 in y axis (camera first uniformly tilts down then uniformly tilts up), and first -0.40 then 0.10 in z axis (camera first fast zooms out then slowly zooms in) and no rotation.

The task you need to complete is to generate a camera motion to discribe: <task_description>.
Now, given the following image, you need to think and call the function needed to proceed with the task.

First, you should image that the given image is shot at the initial location of the camera.
Then, you should analyze the text description and the image content to determine the direction and distance of the following camera motion.
Finally, you should call the function with the correct parameters to generate the camera motion.
Note that Shifting distance and rotation degrees should depend on the image content, like a large-scale scenario should maintain a large shifting and rotation. And shifting and rotation direction should depend on the text description.

Your output should include four parts in the given format:
Observation: <Describe what you observe in the image.>
Thought: <To complete the given task, what is the step you should do.>
Action: <The function call with the correct parameters to proceed with the task. You can only output a function call in this field. Such as: set_camera_motion_complex(0.16, -0.39, 0.25, -0.25, -0.40, 0.10, 0, 0, 0).>
Summary: <Summarize your actions in one or two sentences.>

This is a reply template:
Observation: The image depicts the Parthenon, an ancient large-scale architectural structure with detailed columns and a wide panoramic view. There are people gathered around, further emphasizing the scale of the structure.                    
Thought: As the task is to generate a camera motion where the camera pans right in a decremental motion, I should set a motion that starts with a relatively larger x-axis translation and becomes smaller as the camera moves. Given the large-scale scene in the image, a significant initial pan is appropriate.                                                                                                                                                                  
Action: set_camera_motion(0.8, 0.0, 0.0, 0, 0, 0, "decrement")                                                                                                                                                                                
Summary: I set up a decremental pan to the right, starting with a substantial shift suitable for the large-scale scene, gradually decreasing in intensity as described."""

task_template_rethinking = """You are an agent that is trained to rethink and recomplete a specific task about video generation.
I will provide you with some frames of the generated video, which is generated based on the action you made last time. Additionally, I will describe the task that you should recomplete and give the action you made last time.
According to the task description and the generated video, you should first analyze the action you made last time. Then, you should correct the error in the former action. Finally, you should recomplete the task and make the correct action this time.
If you think the action you made last time is correct, you can recomplete the task with the same action.

Here are some critical points you should follow:
First, the object movement and camera motion in the generated video should strictly follow the given video generation text prompt.
Second, the generated video should have high quality without any artifacts. If there are any artifacts in the generated video, you should adjust your former action, such as decreasing the length of object trajectories or the speed of camera motion.

The video generation text prompt is: <text_prompt>.

Next, I will show you the generated video. You should analyze the video and the task description carefully and reply to me with your observation."""
