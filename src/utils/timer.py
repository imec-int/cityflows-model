import time
from .bcolors import bcolors

class Timer:
    def __init__(self, name):
        self._name = name
        self._start_time = None
        self._end_time = None

    @property
    def _prefix(self):
        return f'Timer "{self._name}"'

    @property
    def _is_running(self):
        return self._start_time is not None and self._end_time is None

    @property
    def _started(self):
        return self._start_time is not None

    @property
    def _completed(self):
        return self._end_time is not None

    def start(self):
        if self._started:
            print(f'{self._prefix}: timer already started')
            return
        
        if self._completed:
            print(f'{self._prefix}: timer already completed')
            return

        self._start_time = time.time()

    def stop(self):
        if not self._started:
            print(f'{self._prefix}: timer not started yet')
            return
        
        if self._completed:
            print(f'{self._prefix}: timer already completed')
            return

        self._end_time = time.time()

    @property
    def duration(self):
        if not self._completed:
            return None

        elapsed_time = self._end_time - self._start_time
        return elapsed_time

    def __str__(self):
        if not self._started:
            return f'{self._prefix}: timer not started yet'
        if self._is_running:
            return f'{self._prefix}: timer still running'

        return f'{self._prefix} elapsed time: %.2f seconds' % self.duration

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.stop()
        print(self)

class PausableTimer:
    def __init__(self, name):
        self._base_name = name
        self._completed_timers = []
        self._current_timer = None

    @property
    def _prefix(self):
        return f'Timer "{self._base_name}"'

    @property
    def _is_running(self):
        return self._current_timer._is_running

    def resume(self):
        name = f'{self._base_name}-{len(self._completed_timers)}'
        self._current_timer = Timer(name)
        self._current_timer.start()

    def pause(self):
        self._current_timer.stop()
        self._completed_timers.append(self._current_timer)

    @property
    def duration(self):
        return sum([t.duration for t in self._completed_timers])

    def __str__(self):
        if self._is_running:
            return f'{self._prefix}: timer is running'

        return f'{self._prefix} elapsed time: %.2f seconds' % self.duration

class UseTimer:
    def __init__(self, timer):
        self._timer = timer

    def __enter__(self):
        if isinstance(self._timer, PausableTimer):
            self._timer.resume()
        elif isinstance(self._timer, Timer):
            self._timer.start()
        else:
            raise Exception('Unknown timer class')

    def __exit__(self, exc_type, exc_value, exc_tb):
        if isinstance(self._timer, PausableTimer):
            self._timer.pause()
        elif isinstance(self._timer, Timer):
            self._timer.stop()
        else:
            raise Exception('Unknown timer class')

def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f'{bcolors.HEADER}Function {func.__name__!r} executed in {(t2-t1):.4f}s{bcolors.ENDC}')
        return result
    return wrap_func

if __name__ == '__main__':
    t = Timer('Test 1')
    t.start()
    time.sleep(0.1)
    t.stop()
    print(t)

    with Timer('Test 2'):
        time.sleep(0.2)

    t = Timer('Test 3')
    t.start()
    t.start()
    time.sleep(0.3)
    t.stop()

    t = Timer('Test 4')
    t.stop()

    t = Timer('Test 5')
    t.start()
    time.sleep(0.5)
    t.stop()
    t.stop()

    t = PausableTimer('Test 6')
    t.resume()
    time.sleep(0.1)
    t.pause()
    print(t)

    t = PausableTimer('Test 7')
    t.resume()
    time.sleep(0.1)
    
    t.pause()
    time.sleep(0.1)
    
    t.resume()
    time.sleep(0.1)
    
    t.pause()
    time.sleep(0.1)
    
    t.resume()
    time.sleep(0.1)
    
    t.pause()
    print(t)
