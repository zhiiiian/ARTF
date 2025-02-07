from models.ARTF import ARTF

def get_model(model_name, args):
    name = model_name.lower()
    if name == 'artf':
        return ARTF(args)

    else:
        assert 0
