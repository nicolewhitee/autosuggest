import os
import sys
import pprint
import dawg

try: 
    import termios
except Exception:
    terminos = fcntl = None

# Waits for a single keypress on stdin and returns the character of the key that was pressed
def read_single_keypress():
    if fcntl is None or termios is None:
        raise ValueError('termios and/or fcntl packages are not available in your system. This is possible because you are not on a Linux Distro.')
    fd = sys.stdin.fileno()
    # save old state
    flags_save = fcntl.fcntl(fd, fcntl.F_GETFL)
    attrs_save = termios.tcgetattr(fd)
    # make raw - the way to do this comes from the termios(3) man page.
    attrs = list(attrs_save)  # copy the stored version to update
    # iflag
    attrs[0] &= ~(termios.IGNBRK | termios.BRKINT | termios.PARMRK |
                  termios.ISTRIP | termios.INLCR | termios.IGNCR |
                  termios.ICRNL | termios.IXON)
    # oflag
    attrs[1] &= ~termios.OPOST
    # cflag
    attrs[2] &= ~(termios.CSIZE | termios. PARENB)
    attrs[2] |= termios.CS8
    # lflag
    attrs[3] &= ~(termios.ECHONL | termios.ECHO | termios.ICANON |
                  termios.ISIG | termios.IEXTEN)
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    # turn off non-blocking
    fcntl.fcntl(fd, fcntl.F_SETFL, flags_save & ~os.O_NONBLOCK)
    # read a single keystroke
    try:
        ret = sys.stdin.read(1)  # returns a single character
    except KeyboardInterrupt:
        ret = 0
    finally:
        # restore old state
        termios.tcsetattr(fd, termios.TCSAFLUSH, attrs_save)
        fcntl.fcntl(fd, fcntl.F_SETFL, flags_save)
    return ret

def demo(running_modules, max_cost, size):
    word_list = []

    running_modules = running_modules if isinstance(running_modules, dict) else {running_modules.__class__.__name__: running_modules}

    print('AUTOSUGGEST DEMO')
    print('Press any key to search for. Press ctrl+c to exit')

    while True: 
        pressed = read_single_keypress()
        if pressed == '\x7f':
            if word_list:
                word_list.pop()
        elif pressed == '\x03':
            break
        else:
            word_list.append(pressed)

        joined = ''.join(word_list)
        print(chr(27) + "[2J")
        print(joined)
        results = {}
        for module_name, module in running_modules.items():
            results[module_name] = module.search(word=joined, max_cost=max_cost, size=size)
        pprint(results)
        print('')

