from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import load_pem_private_key, load_pem_public_key
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding


def gen_key():
    """Generate new private key
    """
    private_key = rsa.generate_private_key(
        public_exponent=65537, key_size=2048, backend=default_backend()
    )
    return private_key


def save_private_key(pk, filename):
    """ Save the private key in the file
    
    Parameters
    ----------
    pk :
        private key
    filename : 
        file name
    """
    pem = pk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    )
    with open(filename, 'wb') as pem_out:
        pem_out.write(pem)

def save_public_key(pk,filename):
    """ Save public key
    
    Parameters
    ----------
    pk : 
        public key
    filename :
        file name 
    """
    pem = pk.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.PKCS1
    )
    with open(filename, 'wb') as pem_out:
        pem_out.write(pem)


def load_private_key(filename):
    """Import the private key
    
    Parameters
    ----------
    filename : 
        file name 
    
    Returns
    -------
        private key
    """
    with open(filename, 'rb') as pem_in:
        pemlines = pem_in.read()
    private_key = load_pem_private_key(pemlines, None, default_backend())
    return private_key

def load_public_key(filename):
    """ Import the public key
    
    Parameters
    ----------
    filename : 
        file name
    
    Returns
    -------
        public key
    """
    with open(filename, 'rb') as pem_in:
        pemlines = pem_in.read()
    public_key = load_pem_public_key(pemlines, default_backend())
    return public_key


def sign_data(private_key,plain_text):
    """Sign the data by using a private key
    
    Parameters
    ----------
    private_key : 
        private key
    plain_text : string
        message for singing
    
    Returns
    -------
        signed message
    """
    # SIGN DATA/STRING
    signature = private_key.sign(
        data=plain_text,
        padding=padding.PSS(
            mgf=padding.MGF1(hashes.MD5()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        algorithm=hashes.MD5()
    )
    return signature

def verify_data(public_key,plain_text,signature):
    """ Verify the message and signature
    
    Parameters
    ----------
    public_key : 
        public key
    plain_text : 
        message
    signature : 
        signed message
    
    Returns
    -------
        result of verification
    """
    try:
        public_key.verify(
            signature=signature,
            data=plain_text,
            padding=padding.PSS(
                mgf=padding.MGF1(hashes.MD5()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            algorithm=hashes.MD5()
        )
        is_signature_correct = True
    except InvalidSignature:
        is_signature_correct = False
    return is_signature_correct
