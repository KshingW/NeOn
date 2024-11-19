from ..models.tiramisuComplex import tiramisuComplex
from ..models.xfeatComplex import xfeatComplex
from ..models.tiramisuAndXfeatComplex import tiramisuAndXfeatComplex
from ..models.smallDispModelComplex import smallDispModelComplex

def getModel(opt):

    model_name = opt['model']
    nkwargs = opt['nkwargs']
    model = None

    if 'tiramisuComplex' in model_name:
        model = tiramisuComplex(**nkwargs)
    elif 'xfeatComplex' in model_name:
        model = xfeatComplex(**nkwargs)
    elif 'tiramisuAndXfeatComplex' in model_name:
        model = tiramisuAndXfeatComplex(**nkwargs)
    elif 'smallDispModelComplex' in model_name:
        model = smallDispModelComplex(**nkwargs)
    else:
        raise ValueError("Model %s not recognized." % model_name)

    model = model.cuda()
    print("----->>>> Model %s is built ..." % model_name)

    return model