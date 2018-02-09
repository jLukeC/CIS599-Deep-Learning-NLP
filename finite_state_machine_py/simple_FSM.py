
# StateMachine
# State
# transition


from state_machine import State, StateMachine
import sys


class Greeting(State):
    def run(self):
        print("Hello!")

    def transition(self, user_input):
        if user_input:
            return EndCall() if user_input == "Go away" else GivePitch()
        return Greeting()

class EndCall(State):
    def run(self):
        print("Peace Out!")
        sys.exit(1)

    def transition(self, user_input):
        return EndCall()

class GivePitch(State):
    def run(self):
        print("I'd like to offer you a 5% discount")

    def transition(self, user_input):
        return GivePitch() if "What" in user_input else EndCall()


class CallStateMachine(StateMachine):
    def __init__(self, initial_state=Greeting()):
        self.state = initial_state
    def run(self, debug=True):
        if (debug):
            print ('Current State', str(self.state))
        self.state.run()
        user_input = input()
        self.state = self.state.transition(user_input)
        self.run()
