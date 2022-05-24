import random

def address_incr(cfg, location):
    '''
    Helper function to determine incrementing address for the circular buffer
    '''
    length = cfg['BUFFER_SIZE']
    if (location + 1) == length:
        return 0
    else:
        return location + 1

def address_decr(cfg, location):
    '''
    Helper function to determine decrementing address for the circular buffer
    '''
    length = cfg['BUFFER_SIZE']
    if (location - 1) < 0:
        return length - 1
    else:
        return location - 1

def twos(value, bits):
    '''
    Helper function for two's complement since python integer precision is limited only by hardware
    Returns:
        value - unsigned value of "bits" precision
    '''
    if value < 0:
        value = (1<<bits) + value
    return value

def r_twos(value, bits):
    '''
    Helper function to reverse two's complement to get back the signed integger, since python integer precision is limited only by hardware
    Returns:
        value - signed value
    '''
    if (value >> (bits-1) & 1) == 1:
        value = value - (1<<bits)
    return value

def inv(value, bits):
    '''
    The most negative number is the value which we are using for INV symbol.
    This function checks to see if input value is the most negative number.
    '''
    if (-1* value) == r_twos(value, bits):
        return True
    return False

def ovf(value, bits):
    '''
    Checks for overflow
    '''
    too_big = 1 << (bits-1)
    if (value < 0):
        value = value * -1
    return value >= too_big

def packer(cfg, data):
    '''
    Packs data coming in.
    Inputs:
        data - input data to compress
    Returns:
        precision_buffer - 1 bit array indicating full or low precision for that particular addr
        packed_buffer - N bit array of packed data
        packed_buffer_vis - N bit array visualization of what is actually packed (low/full/INV)
        stop_location - Address where we "stop" according to # of clock cycles defined within config
    '''
    precision_buffer = [0] * cfg['BUFFER_SIZE']
    packed_buffer = [0] * cfg['BUFFER_SIZE']
    packed_buffer_vis = [0] * cfg['BUFFER_SIZE']
    value_arr_vis = []
    stop_location = 0
    value = 0
    delta_size = int(cfg['WIDTH'] / cfg['PRECISION'])
    bitmask = 0
    j = -1
    for i in range(delta_size):
        bitmask += (1 << i)
    invalid = 1 << (delta_size-1)

    if cfg['DEBUG'] == 1:
        print(f'--------------------------------\n------------PACKER------------\n--------------------------------')

    for i in range(len(data)):
        if (i == 0):
            precision_buffer[stop_location] = 1
            packed_buffer[stop_location] = data[i]
            packed_buffer_vis[stop_location] = data[i]
            stop_location = address_incr(cfg, stop_location)
        else:
            # overflow. Accounts for INVALID sign (max neg number in delta_size bits)
            if (ovf(data[i-1] - data[i], delta_size)):
                # Write what we have first to current addr if previous was not an overflow and data was being compressed
                if cfg['DEBUG'] == 1:
                    print('Inside overflow')
                # Commit what is inside packed values first
                if value_arr_vis:
                    precision_buffer[stop_location] = 0
                    # Better to just write as full precision if just one packed value inside the total width and previous value was full precision
                    if j == cfg['PRECISION']-2 and precision_buffer[address_decr(cfg, stop_location)] == 1:
                        precision_buffer[stop_location] = 1
                        if cfg['DEBUG'] == 1:
                            print('Writing delta as full precision to save space ')
                        pass
                    elif j != -1:
                        # Need to invalidate empty data, set to invalid (max neg number)
                        while (j != -1):
                            value = value | (invalid << (j * delta_size))
                            value_arr_vis.append('INV')
                            j-=1
                        packed_buffer[stop_location] = value
                        packed_buffer_vis[stop_location] = value_arr_vis.copy()
                        precision_buffer[stop_location] = 0
                        value_arr_vis = []
                        stop_location = address_incr(cfg, stop_location)
                        precision_buffer[stop_location] = 0
                    packed_buffer[stop_location] = data[i-1]
                    packed_buffer_vis[stop_location] = data[i-1]
                    stop_location = address_incr(cfg, stop_location)
                # Then write current overflow to next addr
                precision_buffer[stop_location] = 1
                packed_buffer[stop_location] = data[i]
                packed_buffer_vis[stop_location] = data[i]
                stop_location = address_incr(cfg, stop_location)
                # Reset value, j, etc
                j = cfg['PRECISION'] - 1
                value = 0
                value_arr_vis = []
            # No overflow, write/pack deltas
            else:
                if j == -1:
                    value = 0
                    value_arr_vis = []
                    j = cfg['PRECISION'] - 1
                bitshifted = twos(data[i-1] - data[i], delta_size) << (j * delta_size)
                value_arr_vis.append((data[i-1] - data[i]))
                value = value | bitshifted
                if cfg['DEBUG'] == 1:
                    print('Inside delta packing')
                # print(f'bitshifted {bitshifted}, j {j}, value {value}')
                if (j == 0):
                    precision_buffer[stop_location] = 0
                    packed_buffer[stop_location] = value
                    packed_buffer_vis[stop_location] = value_arr_vis.copy()
                    stop_location = address_incr(cfg, stop_location)
                    # value_arr_vis = []
                j -= 1
        if cfg['DEBUG'] == 1:
            print(f'------')
            print(f'packing: {packed_buffer}')
            print(f'packing_vis: {packed_buffer_vis}')
            print(f'value_array vis: {value_arr_vis}')
            print(f'addr: {stop_location}')
            print(f'precision {precision_buffer}')
            print(f'delta: {data[i-1] - data[i]}')
            # print(f'max deltasize bits: {delta_size}')
            print(f'Ovf? {ovf(data[i-1] - data[i], delta_size)}')
            print(f'------')
    # If there is any packed data left we must commit it
    if value_arr_vis and j != -1:
        if cfg['DEBUG'] == 1:
            print(f'Committing the unfinished packed data: {value_arr_vis}')
        if j < (cfg['PRECISION'] - 2) and j != -1:
            # Need to invalidate empty data, set to invalid (max neg number)
            while (j != -1):
                value = value | (invalid << (j * delta_size))
                value_arr_vis.append('INV')
                j-=1
            packed_buffer[stop_location] = value
            if cfg['DEBUG'] == 1:
                print(f'-> {value}')
            packed_buffer_vis[stop_location] = value_arr_vis.copy()
            value_arr_vis = []
            precision_buffer[stop_location] = 0
        else:
            packed_buffer[stop_location] = data[i]
            packed_buffer_vis[stop_location] = data[i]
            if cfg['DEBUG'] == 1:
                print(f'-> {data[i]}')
            precision_buffer[stop_location] = 1
        stop_location = address_incr(cfg, stop_location)
    stop_location = address_decr(cfg, stop_location)

    if cfg['DEBUG'] == 1:
        print(f'Precision Buffer: {precision_buffer}\nPacked buffer: {packed_buffer}\nPacked Vis: {packed_buffer_vis}\nDataIn: {dataIn}')
    return precision_buffer, packed_buffer, stop_location, packed_buffer_vis

def unpacker(cfg, precision_buffer, packed_buffer, stop_location):
    '''
    Unpacks/reconstructs data. Worst case: additional 1xBUFFER_SIZE bits of precision buffer memory
    Inputs:
        stop_location - Address where we "stop" according to # of clock cycles defined within config
        packed_buffer - Compressed data for this function to unpack!
        precision_buffer - 1 bit array storing full or low precision flag
    # of cycles isn't known by unpacker
    Returns:
        output_buffer - Unpacked buffer
    '''
    full_value = 0
    current_location = -1
    first_full_precision = 0
    output_buffer = []
    delta_size = int(cfg['WIDTH'] / cfg['PRECISION'])
    bitmask = 0
    inv_count = 0
    for i in range(delta_size):
        bitmask += (1 << i)
    invalid = 1 << (delta_size-1)

    if cfg['DEBUG'] == 1:
        print(f'--------------------------------\n------------UNPACKER------------\n--------------------------------')

    # Assert precision buffer is same size as packed buffer
    assert len(precision_buffer) == len(packed_buffer), 'Buffer size mismatch!'
    # Assert stop_location is within the size of buffer
    assert (stop_location < len(precision_buffer)) and (stop_location > -1), 'Stop location not within size of buffer!'

    # Start location is where unpacking starts, which is stop location + 1
    start_location = address_incr(cfg, stop_location)

    while (current_location != start_location):
        # Initialize current_location as the loop variable
        if (current_location == -1):
            current_location = start_location
        prev_location = address_decr(cfg, current_location)
        # **********************************************************
        # 0 -> 1 precision: if current location is full precision, we need to backtrack
        # by deleting the previous 'PRECISION'# values and replacing with COUNT 
        # (last known value), only if previous location is 0.
        # If first full precision hasn't been encountered yet we backfill up to the start location
        # **********************************************************
        if (precision_buffer[current_location] == 1 and precision_buffer[prev_location] == 0 and stop_location != prev_location and precision_buffer[address_decr(cfg, prev_location)] != 1):
            if first_full_precision == 1:
                if cfg['DEBUG'] == 1:
                    print('Backtracking')
                if cfg['DEBUG'] == 1:
                    temp = cfg['PRECISION']
                    print(f'Removing {temp+inv_count} values from output buffer')
                # Remove previous 'PRECISION' # values
                for i in range(cfg['PRECISION']+inv_count):
                    output_buffer.pop()
                inv_count = 0
            else:
                if cfg['DEBUG'] == 1:
                    print('Backtracking from first full precision')
                og_full_value = packed_buffer[current_location]
                output_buffer.insert(0, og_full_value)
                if cfg['DEBUG'] == 1:
                    print(f'First full value: {og_full_value}')
                current_location = address_decr(cfg, current_location)
                full_value = packed_buffer[current_location]
                output_buffer.insert(0, full_value)
                if cfg['DEBUG'] == 1:
                    print(f'Second full value: {full_value}')
                first_full_precision = 1
                if (start_location == current_location):
                    current_location = address_incr(cfg, address_incr(cfg, prev_location))
                    continue
                current_location = address_decr(cfg, current_location)
                while(start_location != current_location):
                    # Ignore first value which was the current value
                    for i in range(cfg['PRECISION']):
                        bitshifted = packed_buffer[current_location] >> (i * delta_size)
                        # # Skip if invalid symbol is encountered
                        if(bitshifted & bitmask == invalid):
                            if cfg['DEBUG'] == 1:
                                print(f'Invalid found, skipping')
                            continue
                        full_value = full_value + r_twos(bitshifted & bitmask, delta_size)
                        if cfg['DEBUG'] == 1:
                            print(f'delta: {r_twos(bitshifted & bitmask, delta_size)}')

                        output_buffer.insert(0, full_value)
                        if cfg['DEBUG'] == 1:
                            print(f'index: {i}')
                            print(f'inserting {full_value}')
                    current_location = address_decr(cfg, current_location)
                current_location = address_incr(cfg, address_incr(cfg, prev_location))
                full_value = og_full_value
                continue
        # **********************************************************
        # Basic case of full precision and low precision
        # **********************************************************
        # If current location is a full precision, unpack it to output
        if precision_buffer[current_location] == 1:
            first_full_precision = 1
            full_value = packed_buffer[current_location]
            output_buffer.append(full_value)
        # Otherwise unpack low precision deltas if first full precision has happened
        elif first_full_precision == 1:
            for i in range(cfg['PRECISION']-1, -1, -1):
                bitshifted = packed_buffer[current_location] >> (i * delta_size)
                full_value = full_value - r_twos(bitshifted & bitmask, delta_size)
                if(((bitshifted & bitmask) == invalid) and ((precision_buffer[address_incr(cfg, current_location)] is not 1) or (address_incr(cfg, current_location) == start_location))):
                    inv_count +=1
                output_buffer.append(full_value)

        if cfg['DEBUG'] == 1:
            print(f'Current Location: {current_location}, Output Buffer: {output_buffer}')

        # Increment current location
        current_location = address_incr(cfg, current_location)

    # Any remaining inv_count must be decremented
    for i in range(inv_count):
        output_buffer.pop()

    return output_buffer
