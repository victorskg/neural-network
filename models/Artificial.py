class Artificial(object):
    def __init__(self, inputs, expected_type):
        self.inputs = inputs
        self.expected_type = expected_type

    def __repr__(self):
        return "Expected_Type: {0}".format(self.expected_type)

    def __str__(self):
        return "Expected_Type: {0}".format(self.expected_type)
