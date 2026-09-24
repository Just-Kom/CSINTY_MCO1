This module handles the EXAMINING of a state. That is:
1. Checks to see if the state is VALID;
     * "Is my move valid?"
2. Checks to see if the state is in a DEADLOCK;
     * "Will my move lead to a fail state?"
3. Checks to see if the state is in a PRIORITY;
     * "How much does my move cost?"

Do note that all methods are modularized, so any method MUST be segregated accordingly to avoid confusion.
* Kindly do identify the 

ONLY USE examine function in examine.py for other directories.
* Other functions MUST be inquired first before calling them.

