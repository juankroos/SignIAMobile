import os, time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class SmartWatchdog(FileSystemEventHandler):
    def __init__(self, target_path, delay, callback):
        self.path = target_path
        self.delay = delay
        self.buffer = []
        self.timer = None
        self.callback = callback

    def on_created(self, event):
        if event.src_path == self.path:
            print(f"Fichier créé: {self.path}")
            self.watch_modifications()

    def on_modified(self, event):
        if event.src_path == self.path:
            with open(self.path, 'r') as f:
                lines = f.readlines()
                new_data = lines[len(self.buffer):]
                self.buffer += new_data
                print(f"Nouvelles entrées détectées: {new_data}")
                self.reset_timer()

    def reset_timer(self):
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.delay, self.trigger_callback)
        self.timer.start()

    def trigger_callback(self):
        print("Envoi des données après délai...")
        self.callback(self.buffer)

def send_to_correction(data):
    # Ton pipeline ici
    print("Pipeline de correction reçoit:", data)

# Usage
watcher = SmartWatchdog('data.csv', delay=10, callback=send_to_correction)
observer = Observer()
observer.schedule(watcher, path='.', recursive=False)
observer.start()