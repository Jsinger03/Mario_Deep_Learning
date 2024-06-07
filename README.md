For a long time I wanted to tackle creating a model to play a game, and with the help of ChatGPT4o I was finally able to accomplish it.


To start, I initialized a context in GPT4o with "You are a programmer who specializes in ML, specifically deep learning through python and Gymnasium. You are experienced with debugging and upgrading environments from the old gym format to the new Gymnasium format."


The first issue I encountered was the game environment uses gym, which is no longer maintained, and much of the nes-py code and the mario code which invokes it relies on gym, in a way which is incompatable with the new version, Gymnasium.
Gymnasium has built in wrappers for transforming the step and reset functions, but these were not working as intended.
By having a conversation with GPT4o, I was able to debug and found that the initial creation of the game environment included calls to the old gym versions of several wrappers (specifically TineLimit).
To address this, I returned to GPT4o and asked it if it could create for me a custom wrapper to remove other wrappers. With the output from this prompt, I was able to clear away the old wrappers and advance.
I added several other wrappers to my environment from the ones available in Gymnasium, and I did encounter some issues which, when I presented the error log to GPT4o were quickly resolved (a recurring error was the issue of render_mode, which I set to human when runnign locally but required being set to None when running remotely on Google Colab).
The prompt I used to generate the custom wrapper was "Create a wrapper which takes an environment and strips it back to just its first layer. This will be iterated upon to resolve any issues, include an explanation of how it works and the different elements used."


The next set of errors came up in preprocessing. In this stage, I was adding additional wrappers not from Gymnasium to change the way my model would interpret the environment.
When converting the observation space to grayscale, Gymnasium requires the type as Gymnasium.spaces.Box, however the mario space is gym.spaces.box.Box, so I asked GPT4o to make a helper function to convert it to the correct type.

prompt: "Using the Gymnasium GrayScaleWrapper as a base, create a custom wrapper that addresses problematic behavior related to the difference between gym's gym.spaces.box.Box and Gymnasium's Gymnasium.spaces.Box"

Another error occurred with the vectorizing wrapper. Using the DummyVectorization wrapper from stable_baselines3, the num_envs variable is missing, so I again asked GPT4o to design a wrapper to set that variable.

prompt: "Create a wrapper which takes an environment with the layers we have added thus far and sets the num_envs variable."

Once the DummyVecEnv wrapper was in place, I discovered that it's step functions were producing output that worked with the gym format expected from a step, but not Gymnasium's. This required me to ask GPT4o to create a custom version upgrading its step functions.

prompt: "Based off the DummyVecEnv from stable_baselines3, create a custom wrapper which defines the step_Wait, step_async, and init functions so that they handle inputs and outputs the way Gymnasium requires."

After this, I had GPT4o design a custom reset helper function to call after the stacking in order to get the propper formatted reset function, as the existing one didnt properly interact with Gymnasiums format.
The last issue in this segment was the VecFrameStack wrapper from stable_baselines3, which didn't have a set step function and kept on returning 4 outputs when Gymnasium needs 5. This one took the most iterations with GPT4o to resolve, but in the end the custom wrapper was created and adds the rules for a step function which works.
prompt: "I am encountering a similar problem with the VecFrameStack wrapper as I was with the DummyVecEnv wrapper. Design a custom wrapper which defines the step and step_wait functions for VecFrameStack so that it aligns with the Gymnasium format."


The last set of errors I encountered were when I was training the model.
I encountered numerous errors related to the custom wrappers I had setup and their inability to handle all the areas contained in the "info" variable Gymnasium creates.
Specifically, the field "seed" was unable to be handled for the custom DummyVecEnv custom VecFrameStack.
To address this, I asked GPT4o to iterate on these custom wrappers to handle additional fields in "info".
GPT4o required multiple iterations of testing these changes until all errors were resolved, as each change brought about new errors to debug.

prompt: "While training the model, I am finding that the custom wrappers we created cannot handle all the fields in the info variable. Given the following error logs, redefine the functions to handle all possible fields in info." (logs not included for length)


One of the major issues I encountered when doing this project was GPT4o's habit of getting stuck on one or two ideas and circling between them. 
For instance, this occured when I was addressing the handling of "seed" in the "info" variable in my custom wrappers, where GPT4o decided that storing all the fields should be done in an array of dictionaries. This format, however, brought about errors related to how the compiler saw the array. When I attemopted to address that, GPT4o attempted to solve it with changing the way the dictionaries were called, and started to cycle between its original idea and an alternative which was logically equivalent. When I came back to it with one not working, it would suggest the other, and then continue that cycle. After a few iterations of that, I had to remind GPT4o to remember the conversation history and not to repeat ideas that had failed, and suggested it broaden its search to similar issues online. After this, another couple of iterations occurred before I suggested we change the way the data was stored, and this resulted in GPT4o producing the final version of that wrapper which correctly handled all fields.

Of note, in order for the program to execute, it is neccessary to install the approproate version of torch for ones system. Mine is using the Mac installation with GPU.
On google Colab, it is also necessary to set the environment rendering mode to None
