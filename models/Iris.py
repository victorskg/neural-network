class Iris(object):
    def __init__(self, inputs, expected_type):
        self.inputs = inputs
        self.expected_type = 1 if inputs[4] == expected_type else -1

    def __repr__(self):
        return "Inputs: {0}, Expected_Type: {1}".format(self.inputs, self.expected_type)

    def __str__(self):
        return "Expected_Type: {0}, {1}".format(self.expected_type, self.inputs[4])
