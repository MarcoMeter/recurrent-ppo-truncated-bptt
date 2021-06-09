import multiprocessing
import multiprocessing.connection
from utils import create_env

def worker_process(remote: multiprocessing.connection.Connection, env_name:str, env_path:str, worker_id:int=0):
    """Executes the threaded interface to the environment.
    Args:
        remote (multiprocessing.connection.Connection) -- Parent thread
        env_name (str) -- Name of the to be instantiated environment
        env_path (str): If a Unity environment is used, a path to the executable has to be provided.
        worker_id (int, optional): If a Unity environment is used, each one has to communicate via its distinct port. Defaults to 0.
    """
    # Spawn environment
    try:
        env = create_env(env_name)
    except KeyboardInterrupt:
        pass

    # Communication interface of the environment thread
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(env.step(data))
            elif cmd == "reset":
                remote.send(env.reset())
            elif cmd == "close":
                remote.send(env.close())
                remote.close()
                break
            else:
                raise NotImplementedError
        except:
            break

class Worker:
    """A worker that runs one environment on one thread."""
    child: multiprocessing.connection.Connection
    process: multiprocessing.Process
    
    def __init__(self, env_name:str, env_path:str, worker_id:int=0):
        """
        Args:
            env_name (str) -- Name of the to be instantiated environment
            env_path (str): If a Unity environment is used, a path to the executable has to be provided.
            worker_id (int, optional): If a Unity environment is used, each one has to communicate via its distinct port. Defaults to 0.
        """
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process, args=(parent, env_name, env_path, worker_id))
        self.process.start()