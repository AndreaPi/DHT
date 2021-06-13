# Discrete Hartley Transform in Python
A naive implementation of the [Discrete Hartley Transform](https://en.wikipedia.org/wiki/Discrete_Hartley_transform)(DHT) and Inverse Discrete Hartley Transform (IDHT) in Python. The code has not been optimized for speed. All computations are based on formulas listed [here](https://en.wikipedia.org/wiki/Discrete_Hartley_transform#Properties)

 - To compute the DHT we use the well-known relationship between FFT and DHT. 
 - To compute the IDHT we use the fact that the DHT is involutory, i.e., its own inverse up to a scale factor equal to the sequence length
 - To compute the cyclic convolution of two sequences based on the DHT, we first compute the DHT of the convolution of two sequences using the convolution theorem of the DHT, and then we compute the IDHT of the resulting sequence
 
## Tests
To execute tests, in the same directory where `dht.py` is, run `python test_dht.py`.

## Limitations
Currently, only the 1D DHT is computed.