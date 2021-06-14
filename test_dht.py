from dht import dht, idht, flip_periodic, dht_conv, conv
import numpy as np
import inspect

def test_dht():
    # DHT transform of a constant sequence, based on DFT transform of the same
    N = 20
    x = np.ones((N, ))
    X = dht(x)
    X1 = np.zeros((N, ))
    X1[0] = N
    msg = f"{inspect.stack()[0][3]}: constant sequence test failed"
    np.testing.assert_allclose(X, X1, atol=1e-08, err_msg=msg)
    # DHT transform of a cosine, based on DFT transform of the same
    k = 3
    c = 2.*np.pi*k/N
    i = np.arange(N)
    x = np.cos(c * i)
    X = dht(x)
    X1 = np.zeros((N, ))
    X1[k]  = N/2.
    X1[-k] = X1[-k] + N/2.
    msg = f"{inspect.stack()[0][3]}: cosine sequence test failed"
    np.testing.assert_allclose(X, X1, atol=1e-08, err_msg=msg)
    # DHT transform of a cosine: corner cases
    k = 0
    c = 2.*np.pi*k/N
    i = np.arange(N)
    x = np.cos(c * i)
    X = dht(x)
    X1 = np.zeros((N, ))
    X1[k]  = N/2.
    X1[-k] = X1[-k] + N/2.
    msg = f"{inspect.stack()[0][3]}: cosine corner case #1 test failed"
    np.testing.assert_allclose(X, X1, atol=1e-08, err_msg=msg)
    N = 20
    k = 19
    c = 2.*np.pi*k/N
    i = np.arange(N)
    x = np.cos(c * i)
    X = dht(x)
    X1 = np.zeros((N, ))
    X1[k]  = N/2.
    X1[-k] = X1[-k] + N/2.
    msg = f"{inspect.stack()[0][3]}: cosine corner case #2 test failed"
    np.testing.assert_allclose(X, X1, atol=1e-08, err_msg=msg)


def test_idht():
    # IDHT transform of an impulse at 0, based on IDFT transform of the same
    N = 20
    X = np.zeros((N, ))
    X[0] = N
    x = idht(X)
    x1 = np.ones((N, ))
    msg = f"{inspect.stack()[0][3]}: impulse test failed"
    np.testing.assert_allclose(x, x1, atol=1e-08, err_msg=msg)
    # IDHT transform of sum of impulses at k and N-k,  based on IDFT transform of the same
    k = 3
    X = np.zeros((N, ))
    X[k]  = N/2.
    X[-k] = X[-k] + N/2.
    x = idht(X)
    c = 2.*np.pi*k/N
    i = np.arange(N)
    x1 = np.cos(c * i)
    msg = f"{inspect.stack()[0][3]}: sum of impulses test failed"
    np.testing.assert_allclose(x, x1, atol=1e-08, err_msg=msg)
    # IDHT transform of sum of impulses at k and N-k,  corner cases
    k = 0
    X = np.zeros((N, ))
    X[k]  = N/2.
    X[-k] = X[-k] + N/2.
    x = idht(X)
    c = 2.*np.pi*k/N
    i = np.arange(N)
    x1 = np.cos(c * i)
    msg = f"{inspect.stack()[0][3]}: sum of impulses corner case #1 test failed"
    np.testing.assert_allclose(x, x1, atol=1e-08, err_msg=msg)
    # IDHT transform of sum of impulses at k and N-k,  corner cases
    k = 19
    X = np.zeros((N, ))
    X[k]  = N/2.
    X[-k] = X[-k] + N/2.
    x = idht(X)
    c = 2.*np.pi*k/N
    i = np.arange(N)
    x1 = np.cos(c * i)
    msg = f"{inspect.stack()[0][3]}: sum of impulses corner case #2 test failed"
    np.testing.assert_allclose(x, x1, atol=1e-08, err_msg=msg)


def test_dht_conv():
    N = np.random.randint(1, 20)
    x = np.ones((N, ))
    y = np.copy(x)
    Z = dht_conv(x, y)
    Z1 = np.real(np.fft.fft(x)*np.fft.fft(y))
    msg = f"{inspect.stack()[0][3]} test failed"
    np.testing.assert_allclose(Z, Z1, err_msg=msg)

def test_conv():
    N = np.random.randint(1, 20)
    x = np.ones((N, ))
    y = np.copy(x)
    z = conv(x, y)
    z1 = np.real(np.fft.ifft(np.fft.fft(x)*np.fft.fft(y)))
    msg = f"{inspect.stack()[0][3]} test failed"
    np.testing.assert_allclose(z, z1, err_msg=msg)

if (__name__=='__main__'):
  test_dht()
  print("test_dht() passed")
  test_idht()
  print("test_idht() passed")
  test_dht_conv()
  print("test_dht_conv() passed")
  test_conv()
  print("test_conv() passed")