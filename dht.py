import numpy as np

def dht(x:np.array):
    """ Compute the DHT for a sequence x of length n using the FFT.
    """
    X = np.fft.fft(x)
    X = np.real(X)-np.imag(X)
    return X


def idht(X:np.array):
    """ Compute the IDHT for a sequence x of length n using the FFT. 
    
    Since the DHT is involutory, IDHT(x) = 1/n DHT(H) = 1/n DHT(DHT(x))
    """
    n = len(X)
    x = dht(X)
    x = 1.0/n*x
    return x

def flip_periodic(x:np.array):
    return np.concatenate((x[0], np.flip(x[1:])), axis=None)


def dht_conv(x:np.array, y:np.array):
    """ Computes the DHT of the convolution of x and y, sequences of length n, using FFT.

    This is a straightforward implementation of the convolution theorem for the
    DHT. See https://en.wikipedia.org/wiki/Discrete_Hartley_transform#Properties

    """  
    
    err_msg = "x and y must be 1D sequences of the same length"
    assert (x.shape == y.shape) & (np.squeeze(x).ndim < 2), err_msg

    X = dht(x)
    Y = dht(y)
    Xflip = flip_periodic(X)
    Yflip = flip_periodic(Y)
    Yeven = 0.5*  (Y + Yflip)
    Yodd  = 0.5 * (Y - Yflip)
    Z = X * Yeven + Xflip * Yodd
    return Z    


def conv(x:np.array, y:np.array):
    """ Computes the convolution of x and y, sequences of length n, using the DHT.

    Once the DHT of the convolution has benn computed using dht_conv(), 
    computing the convolution just requires a IDHT.
    """  
    
    err_msg = "x and y must be 1D sequences of the same length"
    assert (x.shape == y.shape) & (np.squeeze(x).ndim < 2), err_msg

    Z = dht_conv(x, y)
    z = idht(Z)
    return z
