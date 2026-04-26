class Generator:
    def __init__(self, model_fn):
        self.model_fn = model_fn

    def generate(self, prompt):
        return self.model_fn(prompt)
