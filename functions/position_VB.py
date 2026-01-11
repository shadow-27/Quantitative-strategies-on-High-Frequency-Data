# function returns a position for
# the Volatility breakout approach

import numpy as np

def positionVB(signal,   # signal
             lower,    # lower threshold
             upper,    # upper threshold
             pos_flat, # vector indicating when position should be 0 (flat)
             strategy): # type of the strategy: "mom" or "mr"
    # Checks it the strategy is correct
    if strategy not in ["mom", "mr"]:
        print("Incorrect strategy parameter. Use 'mom' or 'mr'!")
        return None
    
    # Creates vectors with zeros
    position = np.zeros(len(signal))
    
    # Going through signals
    for i in range(1, len(signal)):
        if pos_flat[i] == 1:
            position[i] = 0
        else:
            # Checks if there are any missings, starting from the second one
            if not np.isnan(signal[i-1]) and not np.isnan(upper[i-1]) and not np.isnan(lower[i-1]):
                # If the previous position is 0
                if position[i-1] == 0:
                    if signal[i-1] > upper[i-1]:
                        position[i] = -1
                    elif signal[i-1] < lower[i-1]:
                        position[i] = 1
                # If the previous position is -1
                elif position[i-1] == -1:
                    if signal[i-1] > lower[i-1]:
                        position[i] = -1
                    elif signal[i-1] < lower[i-1]:
                        position[i] = 1
                # If the previous position is 1
                elif position[i-1] == 1:
                    if signal[i-1] < upper[i-1]:
                        position[i] = 1
                    elif signal[i-1] > upper[i-1]:
                        position[i] = -1
            else:
                # If there are missing values, keep the current position
                position[i] = position[i-1]
    
    # Reverse position for momentum strategy
    if strategy == "mom":
        position = -position
    
    return position