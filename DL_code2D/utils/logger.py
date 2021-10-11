from datetime import datetime
from pytz import timezone
import pytz

class logger(object):
   def __init__(self, stream, path, desc):
       self.stream = stream
       t = datetime.now(tz=pytz.utc).astimezone(timezone('US/Pacific')).strftime('%m%d%H%M%S_')
       self.name = t+desc
       self.path = path+self.name+'.txt'
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
       open(self.path, 'a').write(data)
       return
   def name(self):
       name = self.name
       return name
   def flush(self):
       pass
