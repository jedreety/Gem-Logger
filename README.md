# Gem::Logger

> **Gem::Logger is a single-header, zero-dependency, thread-safe C++23 logger. Include `logger.h`, compile with `-std=c++23`.**

### Quick-Start
Copy paste the file into your project
```cpp
#include "logger.h" 
```

## Why Gem::Logger?
- **Near-Zero CPU Impact** - By using an asynchronous Lock-Free Queue, your application threads never "stall" while waiting for disk I/O. They simply drop the message and continue execution instantly.
  
- **Native Flexibility** - Simultaneously broadcast logs to the console (with ANSI colors), plain text files, and structured JSON for analysis tools like ELK or Datadog.

Happy logging!
