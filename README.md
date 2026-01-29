# TR-APC

## Furax addition

Experimental operators were added locally to the FURAX library as not all operations were possible with the current version. It is therefore necessary to replace these files (with the ones located in `furax_changes` folder) when installing FURAX in order to run this code : 

- `src/furax/core/_fourier.py`
- `src/furax/obs/operators/_hwp.py`
- `src/furax/obs/operators/_polarizers.py`  

And also adjust the `__init__.py` files :
- `src/furax/core/__init__.py` : import **HWPDemodOperator**, **ChopperDemodOperator**, and add them to the `__all__`list
- `src/furax/obs/operators/__init__.py` : import **WPOperator**, **RealisticHWPOperator**, **ActualLinearPolarizerOperator** and add them to the list
- `src/furax/obs/__init__.py`: same as previous line