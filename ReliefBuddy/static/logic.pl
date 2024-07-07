% logic.pl
% Define some self-care activities
activity(reading, relaxing).
activity(exercise, energizing).
activity(meditation, calming).

% Define a rule to suggest an activity based on mood
suggest_activity(Mood, Activity) :- activity(Activity, Mood).
