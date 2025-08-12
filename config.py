import os

# Try to import secret.py if it exists
try:
    import secret
    SECRETS_AVAILABLE = True
except ImportError:
    SECRETS_AVAILABLE = False

def get_credential(var_name, default=None):
    """
    Get a credential from environment variables or secret.py.
    
    Args:
        var_name (str): Name of the variable to get
        default: Default value if not found in either source
        
    Returns:
        The credential value or raises an exception if not found
        
    Raises:
        ValueError: If the credential is not found in environment or secret.py
    """
    # First check environment variables
    env_value = os.getenv(var_name)
    if env_value is not None:
        return env_value
        
    # Then check secret.py if available
    if SECRETS_AVAILABLE and hasattr(secret, var_name):
        return getattr(secret, var_name)
        
    # If default is provided, use it
    if default is not None:
        return default
        
    # Otherwise, raise an error
    raise ValueError(
        f"Credential '{var_name}' not found in environment variables "
        f"or secret.py. Please set the {var_name} environment variable "
        f"or add it to secret.py."
    )

# Polygon.io API key
# Currently requires Stocks and Options Developer Plans (or higher)
API_KEY = get_credential("API_KEY")

# PostgreSQL credentials
PG_HOST = get_credential("PG_HOST")
PG_PORT = get_credential("PG_PORT")
PG_DATABASE = get_credential("PG_DATABASE")
PG_USER = get_credential("PG_USER")
PG_PASSWORD = get_credential("PG_PASSWORD") 