from state_machine import State, StateMachine
from ml_microservice_wrapper import get_sentiment, get_type_of_sentence, get_entities
import random
import sys

bot_indicator = "BOT: "
DEBUG = False

def yes_no_unsure(user_input,yes_state, no_state, unsure_state):
    sentiment = get_sentiment(user_input)
    sentence_type = get_type_of_sentence(user_input)

    if (DEBUG):
        print ("[DEBUG] Sentiment:", sentiment)
        print ("[DEBUG] Sentence Type:", sentence_type)

    # ordering would change depending on best indicators
    if sentence_type == "interrogative":
        return unsure_state
    elif sentiment['polarity'] < -0.25 or sentence_type == "negative":
        return no_state
    elif sentiment['polarity'] > 0.25 or sentence_type == "assertive" :
        return yes_state
    else:
        return unsure_state

class SendGreeting(State):
    def run(self):
        print(bot_indicator,"Hello I'm Luke from Zenreach, are you interested in receiving free wifi at your restaurant?")

    def transition(self, user_input):
        return yes_no_unsure(user_input,GivePitch(),AttemptToContinue(),ProvideMoreInfo())

class GivePitch(State):
    def run(self):
        print(bot_indicator,"Zenreach helps businesses boost their marketing. If you're interested we can schedule an on-site")

    def transition(self, user_input):
        return yes_no_unsure(user_input,ScheduleFollowUp(),EndCall(),ScheduleFollowUp())

class AttemptToContinue(State):
    def run(self):
        print(bot_indicator,"I don't want you to miss out, do you want to hear a quick pitch?")

    def transition(self, user_input):
        return yes_no_unsure(user_input,GivePitch(),EndCall(),ProvideMoreInfo())

class ProvideMoreInfo(State):
    def run(self):
        print(bot_indicator,"Zenreach is a game-changer for brick and mortar businesses. You’ll get the tools you need to collect contacts faster and engage customers more effectively.")

    def transition(self, user_input):
        return yes_no_unsure(user_input,GivePitch(),EndCall(),ScheduleFollowUp())

class EndCall(State):
    def run(self):
        print(bot_indicator,"Ok then bye!")
        sys.exit()

    def transition(self, user_input):
        return yes_no_unsure(EndCall(),EndCall(),EndCall())

class ScheduleFollowUp(State):
    def run(self):
        print(bot_indicator,"When is the best time for us to talk (Date or Time - Ex. 5pm on Decemeber 15th)?")

    def transition(self, user_input):
        entities = get_entities(user_input);
        date_or_time = [x['text'] for x in entities if x['label'] == 'TIME' or x['label'] == 'DATE']
        if len(date_or_time) > 0:
            print(bot_indicator, "great we will talk then (" + " ".join(date_or_time) + ")");
        else:
            print (bot_indicator, "Sorry I couldn't find a Date or Time in there")
        return EndCall()


class CallStateMachine(StateMachine):
    def __init__(self, initial_state=SendGreeting()):
        self.state = initial_state
    def run(self, debug=DEBUG):
        if (debug):
            print ('[DEBUG] Current State', str(self.state))
        self.state.run()

        user_input = input()
        self.state = self.state.transition(user_input)
        self.run()


if __name__ == '__main__':
    if (len(sys.argv) > 1) and (sys.argv[1] == "debug"):
        DEBUG = True
    M = CallStateMachine()
    M.run()
