import time
import sys


class AbortedSequenceException(Exception):
    '''
    Raised when the error will not be solved by trying again, hence aborting the sequence
    '''
    pass


class RetryException(Exception):
    '''
    Raised when all tries failed
    '''

    def __init__(self, message, errors):
        message = message + "\n" + "Individual error messages were: \n\t" + \
            "\n\t".join(map(lambda e: str(e), errors))
        super().__init__(message)


def make_execute_with_retry(n_tries=5, wait_time=1, verbose=False):

    my_print = print if verbose else lambda *a, **k: None

    def execute_with_retry(func, *args, **kwargs):
        errors = []
        for i in range(n_tries):
            try:
                res = func(*args, **kwargs)
                if i > 0:
                    my_print('Success!')
                return res
            except AbortedSequenceException as e:
                raise e
            except Exception as e:
                errors.append(e)
                if i < n_tries - 1:
                    my_print(
                        f"Error: {e}, trying again in {wait_time} seconds...", end=' ')
                    sys.stdout.flush()
                else:
                    my_print(f"Error: {e}")
                time.sleep(wait_time)
        raise RetryException(f"Error: all {n_tries} tries failed", errors)

    return execute_with_retry
