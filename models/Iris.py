class Iris(object):
    def __init__(self, inputs, expected_type):
        self.inputs = inputs
        self.expected_type = 1 if inputs[len(inputs)-1] == expected_type else 0

    def __repr__(self):
        return "Expected_Type: {0}, {1}".format(self.expected_type, self.inputs[len(self.inputs)-1])

    def __str__(self):
        return "Expected_Type: {0}, {1}".format(self.expected_type, self.inputs[len(self.inputs)-1])
