# SEL_Navel
Social-Emotional Learning (SEL) role-play study with the Navel robot, exploring how children process worries about the transition to middle school through guided role-play with a social robot.
 
This repository contains the robot interaction software used to run the sessions, together with the data-collection materials (questionnaires) and the role-play scripts (stories), in both Italian and English.


## About the study

The study uses the Navel robot in role-play activities with primary school children to support their understanding of emotions around a stressful transition — starting middle school. Children take on the roles of Navel and classmates in short scripted scenes with blanks they complete themselves, guided by a teacher. Four story variants are included (A, B, C – "Robot", D – "Control"), each built around a different worry (not knowing anyone, strict teachers, friends changing, harder subjects).

Data is collected through short questionnaires completed by students (after each session and at the end of the programme) and by teachers (end of programme), using 5-point Likert-scale items plus open-ended questions on engagement and perceived emotional understanding.

This material supports an ongoing study toward an HRI 2027 submission.

## `robot_scripts/`

Python-based interaction stack that runs a live session with the robot:

- **`InteractionHub.py`** - the Interaction Manager :coordinates the overall interaction flow between the Speech Recognizer, the Robot Manager, and the human operator. The robot first presents the activity, then waits for the operator to start the listening loop. For each transcribed utterance, the Hub asks the operator whether a response should be generated; if so, it queries the LLM, retrieves the generated response and its associated emotional behaviour, and forwards both to the Robot Manager, otherwise the utterance is discarded and listening resumes.
- **`RobotManager.py`** — robot movement controller (keyboard-driven) and manager of speech and facial-expression output. Maps the selected emotion to one of Navel's six built-in facial states.
- **`SpeakerRecognizer.py`** — Voice Activity Detection (Silero VAD) combined with a silence threshold to segment incoming audio into chunks, each transcribed via the Microsoft Azure Speech Recognition API. Transcription starts before the speaker finishes (0.5 s silence threshold for mid-utterance pauses, 6 s max chunk duration) to reduce latency; end-of-utterance is detected via a 2 s silence threshold reinforced by VAD.
- **`OpenAIClient.py`** — client class that queries the OpenAI API and returns the generated text response together with the selected emotion. Response generation uses GPT-4o. The model acts as the conversational component of the social robot, with scenario-specific context provided at inference time, and additionally selects an emotional display from Ekman's six basic emotions (Anger, Disgust, Fear, Happiness, Sadness, Surprise) plus Neutrality.
- **`prompt.yaml`** — prompt templates and configuration used to drive the robot's responses.
- **`stories.json`** — contains information about the specific activity to perform: welcome message, story to fill, story related knowledge to add to the prompt. 


## Questionnaires

- **`student_questionnaire_ITA.pdf`** / **`student_questionnaire_ENG.pdf`** — student questionnaire (end of session and end of programme) :
  - *End of session*: how the child felt during the role-play, comfort playing their character, how easy it was to understand Navel's feelings.
  - *End of programme *: whether hearing Navel express its feelings helped, what stood out, differences between talking to Navel vs. the teacher, and what the child learned.

## Stories

Located in `Storie ITA/` and `Storie ENG/`: four role-play scripts (A, B, C – Robot, D – Control) with four characters (Navel, Student 1, Student 2, Teacher) and blank spaces for children to complete in character.


## Requirements
 
- Python 3.x
- Key dependencies (see `requirements.txt`): `openai`, `omegaconf`, `python-dotenv`, `torch`, `pyaudio`, `requests`, `azure-cognitiveservices-speech`
- A `.env` file with the required API keys (OpenAI, Azure Speech) loaded via `python-dotenv`

 
