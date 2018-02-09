
class State(object):
    def run(self):
        pass
    def transition(self):
        pass
    def __str__(self):
        return self.__class__.__name__

class StateMachine(object):
    def __init__(self, initial_state):
        pass
    def start(self):
        pass
