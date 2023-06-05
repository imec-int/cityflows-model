def interpolate(factor, val_1, val_2, disable_extrapolation=True):
    if (disable_extrapolation and (factor < 0 or factor > 1)):
        raise Exception('Provided interpolation factor %f is not between 0 and 1' % factor)
    
    return factor * (val_2 - val_1) + val_1
